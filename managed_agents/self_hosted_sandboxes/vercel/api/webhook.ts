/**
 * Vercel Function: Anthropic session webhook → drain environment work queue →
 * spin up a Vercel Sandbox per item.
 *
 * The webhook is a wake-up signal only. Each delivery drains *all* pending
 * work items (not just the one that triggered it), so a single arriving
 * webhook recovers any earlier missed deliveries. A bad work item is logged
 * and skipped; it stays un-acked and reclaims on the next webhook, so it
 * can't wedge the rest of the queue.
 *
 * Flow per webhook delivery:
 *   1. Verify the Standard Webhooks signature via `client.beta.webhooks.unwrap()`.
 *   2. Only act on data.type == "session.status_run_started".
 *   3. Loop work.poll() until empty; per item: ack, then create a Vercel
 *      Sandbox running runner/runner.mjs.
 *
 * Env (vercel env add ...):
 *   ANTHROPIC_WEBHOOK_SECRET   - issued by Anthropic at webhook registration
 *   ANTHROPIC_ENVIRONMENT_ID   - the self-hosted environment id
 *   ANTHROPIC_ENVIRONMENT_KEY  - the environment key: the single credential for
 *                                work poll/ack/stop and the runner's session calls
 *   ANTHROPIC_BASE_URL         - optional, default https://api.anthropic.com
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import Anthropic from "@anthropic-ai/sdk";
import { Sandbox } from "@vercel/sandbox";

const SDK_VERSION = "^0.97.0";
const SANDBOX_TIMEOUT_MS = 30 * 60 * 1000;
const MAX_DRAIN = 25;
// Where the runner's package.json + runner.mjs live and where `npm install` runs.
const RUNNER_DIR = "/vercel/sandbox/runner";
// The agent's working tree. Created at sandbox start; matches the workdir the
// other demos use (Modal mounts a volume at /mnt/session, the CF container's
// `ant beta:worker run --workdir /workspace`). Vercel Sandboxes have no
// persistent volumes — these are fresh per sandbox.
const AGENT_DIRS = ["/mnt/session", "/workspace"];
const WORKDIR = "/workspace";

// Read at module load so Vercel's file tracer bundles it. vercel.json
// includeFiles is a belt-and-suspenders for builds where tracing misses
// the readFileSync.
const RUNNER_SOURCE = readFileSync(
  fileURLToPath(new URL("../runner/runner.mjs", import.meta.url)),
  "utf-8",
);

// The runner's package.json — installed inside the sandbox at spawn time.
// Pinned to the same SDK version the webhook handler uses so the
// toolRunner() API surface matches.
const RUNNER_PACKAGE_JSON = JSON.stringify(
  {
    name: "cma-self-hosted-sandbox-runner",
    private: true,
    type: "module",
    dependencies: { "@anthropic-ai/sdk": SDK_VERSION, minimatch: "^9.0.5" },
  },
  null,
  2,
);

function requireEnv(name: string): string {
  const v = process.env[name];
  if (!v) throw new Error(`${name} is not set`);
  return v;
}

function baseURL(): string {
  return process.env.ANTHROPIC_BASE_URL ?? "https://api.anthropic.com";
}

/** Scrub credential shapes before mirroring sandbox output to the function log. */
function redact(s: string): string {
  return s
    .replace(/sk-ant-[A-Za-z0-9._-]+/g, "sk-ant-[REDACTED]")
    .replace(/whsec_[A-Za-z0-9+/=_-]+/g, "whsec_[REDACTED]")
    .replace(/Bearer\s+\S{8,}/gi, "Bearer [REDACTED]");
}

interface ProcessedItem {
  session_id: string;
  work_id: string;
  sandbox_id?: string;
  reused?: boolean;
  error?: string;
}

/**
 * session_id → sandbox_id mapping, optional. Vercel Sandbox has no native
 * tag/label/get-by-name API and `Sandbox.list()` returns no per-sandbox
 * metadata, so reuse needs external state. Uses Vercel KV when configured;
 * falls back to fresh-per-run otherwise so the demo works without provisioning.
 */
type KVLike = {
  get<T>(key: string): Promise<T | null>;
  set(key: string, value: unknown, opts?: { px?: number }): Promise<unknown>;
  del(key: string): Promise<unknown>;
};
let _kv: KVLike | null | undefined;
async function kvStore(): Promise<KVLike | null> {
  if (_kv !== undefined) return _kv;
  if (!process.env.KV_REST_API_URL || !process.env.KV_REST_API_TOKEN) return (_kv = null);
  try {
    const { kv } = await import("@vercel/kv");
    return (_kv = kv as unknown as KVLike);
  } catch {
    return (_kv = null);
  }
}

const sandboxKey = (sessionId: string) => `cma:sandbox:${sessionId}`;

/**
 * Find an existing running sandbox for this session and renew its lifetime.
 * Returns `null` when there's no mapping, the mapped sandbox is no longer
 * running, or KV isn't configured.
 */
async function findLiveSandbox(sessionId: string): Promise<Sandbox | null> {
  const kv = await kvStore();
  if (!kv) return null;
  const sandboxId = await kv.get<string>(sandboxKey(sessionId));
  if (!sandboxId) return null;
  try {
    const sb = await Sandbox.get({ sandboxId });
    if (sb.status !== "running") {
      await kv.del(sandboxKey(sessionId));
      return null;
    }
    // The sandbox VM has its own clock independent of the runner — renew it on
    // each new run so an active session doesn't get killed mid-turn. Once the
    // runner exits (60s after end_turn idle) renewals stop and the sandbox
    // dies at the next timeout boundary.
    await sb.extendTimeout(SANDBOX_TIMEOUT_MS).catch(() => {});
    return sb;
  } catch {
    await kv.del(sandboxKey(sessionId)).catch(() => {});
    return null;
  }
}

/** Diagnostics we author and control — safe to surface in logs and the
 * webhook response (key names, never values). */
class WorkError extends Error {}

/** Ack and create a sandbox for one work item.
 * Raises on ack failure or Vercel API errors — the caller treats any raise
 * as "skip this item and keep draining".
 */
async function processWorkItem(
  anthropic: Anthropic,
  work: { id: string; data: { id: string } },
  environmentId: string,
  environmentKey: string,
): Promise<ProcessedItem> {
  const sessionId = work.data.id;

  // `anthropic` is authed with the environment key — the single credential for
  // poll / ack / stop — so no per-call Authorization header is needed.
  await anthropic.beta.environments.work.ack(work.id, { environment_id: environmentId });

  // Reuse a live sandbox for the same session if KV has a mapping for it. The
  // runner inside is still streaming and dispatching; it just needs the
  // already-extended VM timeout. Without KV every delivery creates a fresh
  // sandbox — SessionToolRunner dedups via seen/answered so a duplicate runner
  // is wasteful, not wrong.
  const existing = await findLiveSandbox(sessionId);
  if (existing) {
    console.log(`[webhook] work=${work.id} session=${sessionId} sandbox=${existing.sandboxId} (reused)`);
    return { session_id: sessionId, work_id: work.id, sandbox_id: existing.sandboxId, reused: true };
  }

  const sandbox = await Sandbox.create({
    runtime: "node24",
    timeout: SANDBOX_TIMEOUT_MS,
    networkPolicy: {
      // npm install + Anthropic API. Add github.com etc. if your tools need it.
      allow: [new URL(baseURL()).hostname, "registry.npmjs.org"],
    },
  });
  // Record session_id → sandbox_id with a TTL matching the VM timeout so stale
  // mappings self-evict. Best-effort — a write failure just means the next
  // delivery creates a duplicate sandbox.
  await (await kvStore())?.set(sandboxKey(sessionId), sandbox.sandboxId, { px: SANDBOX_TIMEOUT_MS }).catch(() => {});

  await sandbox.writeFiles([
    { path: `${RUNNER_DIR}/runner.mjs`, content: Buffer.from(RUNNER_SOURCE) },
    { path: `${RUNNER_DIR}/package.json`, content: Buffer.from(RUNNER_PACKAGE_JSON) },
  ]);

  // Root-owned paths — create them as root, chown to the sandbox user so the
  // runner (which runs unprivileged) can write the agent's working tree.
  const mkdirs = await sandbox.runCommand({
    cmd: "sh",
    args: ["-c", `mkdir -p ${AGENT_DIRS.join(" ")} && chown -R "$(stat -c %u:%g ${RUNNER_DIR})" ${AGENT_DIRS.join(" ")}`],
    sudo: true,
  });
  if (mkdirs.exitCode !== 0) {
    console.warn(`[webhook] sandbox=${sandbox.sandboxId} mkdir rc=${mkdirs.exitCode}: ${redact((await mkdirs.output("both")).trim())}`);
  }

  // Install runner deps and surface the result in the *function* logs
  // (`vercel logs <project>`) — sandbox stdout isn't visible without the
  // Sandboxes observability tab, which not every plan has.
  const install = await sandbox.runCommand({
    cmd: "sh",
    args: ["-c", `cd ${RUNNER_DIR} && npm install --no-audit --no-fund 2>&1`],
  });
  const installOut = await install.output("both");
  console.log(
    `[webhook] sandbox=${sandbox.sandboxId} npm install rc=${install.exitCode}: ${redact(installOut.slice(-500).trim())}`,
  );
  if (install.exitCode !== 0) throw new WorkError(`npm install failed rc=${install.exitCode}`);

  // Runner detached — its heartbeat must outlive this function invocation.
  const runner = await sandbox.runCommand({
    cmd: "node",
    args: ["runner.mjs"],
    cwd: RUNNER_DIR,
    // Same env contract as `ant beta:worker poll --on-work`: runner.mjs reads these
    // to build the client and run EnvironmentWorker.handleItem().
    // ANTHROPIC_ENVIRONMENT_KEY is the runner's single credential.
    env: {
      ANTHROPIC_BASE_URL: baseURL(),
      ANTHROPIC_ENVIRONMENT_KEY: environmentKey,
      ANTHROPIC_SESSION_ID: sessionId,
      ANTHROPIC_ENVIRONMENT_ID: environmentId,
      ANTHROPIC_WORK_ID: work.id,
      WORKDIR,
    },
    detached: true,
  });

  // Mirror the runner's first ~30s of stdout/stderr into the function log so
  // crash-on-start (bad import, 401, missing env) and the first tool dispatch
  // are debuggable from `vercel logs` alone. The runner keeps going after we
  // stop listening.
  const tail: string[] = [];
  const tailAbort = new AbortController();
  const tailTimeout = setTimeout(() => tailAbort.abort(), 30_000);
  try {
    for await (const log of runner.logs({ signal: tailAbort.signal })) {
      tail.push(log.data);
      if (tail.join("").length > 8_000) break;
    }
  } catch {
    // aborted by timeout — expected
  } finally {
    clearTimeout(tailTimeout);
  }
  console.log(`[webhook] sandbox=${sandbox.sandboxId} runner head:\n${redact(tail.join("").trim()) || "(no output yet)"}`);

  console.log(`[webhook] acked work=${work.id} session=${sessionId} sandbox=${sandbox.sandboxId} (created)`);
  return { session_id: sessionId, work_id: work.id, sandbox_id: sandbox.sandboxId };
}

/**
 * Drain the work queue. The TS `WorkPoller` long-polls and never returns on
 * an empty queue, but a serverless handler must respond — so poll → ack until
 * empty here. The runner sandbox owns the lease (heartbeat + force-stop), so
 * the webhook never posts `stop`.
 */
async function drainWork(
  anthropic: Anthropic,
  environmentId: string,
  environmentKey: string,
): Promise<ProcessedItem[]> {
  const results: ProcessedItem[] = [];
  for (let i = 0; i < MAX_DRAIN; i++) {
    let work;
    try {
      // SDK auto-sends `anthropic-beta: managed-agents-2026-04-01` — no extra betas needed.
      work = await anthropic.beta.environments.work.poll(environmentId, {
        reclaim_older_than_ms: 2000,
      });
    } catch (e) {
      // /work/poll can 404 when it dequeues a Redis entry whose session is
      // gone. The stale entry is consumed server-side, so retrying moves past
      // it; anything else is a config/transient failure — stop and recover on
      // the next webhook.
      const { status, requestID } = (e ?? {}) as { status?: number; requestID?: string | null };
      console.warn(
        `[webhook] poll failed status=${status ?? "?"} request_id=${requestID ?? "?"} — ${status === 404 ? "skipping" : "stopping drain"}`,
      );
      if (status === 404) continue;
      break;
    }
    if (!work) break;
    if (work.data.type !== "session") {
      console.log(`[webhook] skipping work=${work.id} type=${work.data.type}`);
      continue;
    }
    try {
      results.push(await processWorkItem(anthropic, work, environmentId, environmentKey));
    } catch (e) {
      // Only WorkError carries our own controlled diagnostic text. SDK / Vercel
      // exceptions can embed request context, so log type only.
      const detail = e instanceof WorkError ? e.message : e?.constructor?.name ?? "unknown";
      console.error(`[webhook] FAILED work=${work.id} session=${work.data.id}: ${detail}`);
      results.push({ session_id: work.data.id, work_id: work.id, error: detail });
    }
  }
  return results;
}

// Vercel only hands Node.js `api/` functions the Web Standard `Request` when the
// handler is exported as a named HTTP method — `export default` gets the legacy
// `(req: IncomingMessage, res: ServerResponse)` shape, where `req.text()` doesn't exist.
export async function POST(req: Request): Promise<Response> {
  const anthropic = new Anthropic({
    authToken: requireEnv("ANTHROPIC_ENVIRONMENT_KEY"),
    baseURL: baseURL(),
    // Pass the whsec_ secret as-is: the SDK decodes its URL-safe base64 internally.
    webhookKey: requireEnv("ANTHROPIC_WEBHOOK_SECRET"),
  });

  const body = await req.text();
  let event: ReturnType<typeof anthropic.beta.webhooks.unwrap>;
  try {
    event = anthropic.beta.webhooks.unwrap(body, { headers: Object.fromEntries(req.headers) });
  } catch (e) {
    console.error(`[webhook] signature reject: ${e instanceof Error ? e.constructor.name : "Error"}`);
    return new Response("signature verification failed", { status: 401 });
  }

  console.log(`[webhook] event=${event.data.type} session_id=${event.data.id}`);
  if (event.data.type !== "session.status_run_started") {
    return Response.json({ status: "ignored", event_type: event.data.type });
  }

  const spawned = await drainWork(
    anthropic,
    requireEnv("ANTHROPIC_ENVIRONMENT_ID"),
    requireEnv("ANTHROPIC_ENVIRONMENT_KEY"),
  );
  return Response.json({ status: "ok", event_type: event.data.type, spawned });
}

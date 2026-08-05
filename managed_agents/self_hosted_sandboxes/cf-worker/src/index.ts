/**
 * Cloudflare Worker (pure-Worker variant): webhook → drain environment work
 * queue → spin up a per-session Durable Object running the TS tool dispatcher
 * with an in-isolate fake filesystem. No container, no real shell.
 *
 * The webhook is a wake-up signal only — each delivery drains *all* pending
 * work items, so a single arriving webhook recovers any earlier missed ones.
 */
import Anthropic from "@anthropic-ai/sdk";
import type { SandboxRunner } from "./runner";

export { SandboxRunner } from "./runner";

export interface Env {
  RUNNER: DurableObjectNamespace<SandboxRunner>;
  ANTHROPIC_BASE_URL: string;
  ANTHROPIC_ENVIRONMENT_ID: string;
  ANTHROPIC_WEBHOOK_SECRET: string;
  ANTHROPIC_ENVIRONMENT_KEY: string;
}

const MAX_DRAIN = 25;

/** Scrub credential shapes before mirroring an error message to the function log. */
function redact(s: string): string {
  return s
    .replace(/sk-ant-[A-Za-z0-9._-]+/g, "sk-ant-[REDACTED]")
    .replace(/whsec_[A-Za-z0-9+/=_-]+/g, "whsec_[REDACTED]")
    .replace(/Bearer\s+\S{8,}/gi, "Bearer [REDACTED]");
}

/**
 * Structured, redacted error detail for log lines. Surfaces the SDK's
 * `status` + `requestID` when present so an API failure is correlatable to a
 * server-side trace, and the message (with credential shapes stripped) for
 * everything else (DO RPC failures, container start errors).
 */
function errDetail(e: unknown): string {
  if (e instanceof Error) {
    const { status, requestID } = e as Error & { status?: number; requestID?: string | null };
    const parts = [e.constructor.name];
    if (status !== undefined) parts.push(`status=${status}`);
    if (requestID) parts.push(`request_id=${requestID}`);
    if (e.message) parts.push(redact(e.message));
    return parts.join(" ");
  }
  return String(e);
}

/**
 * Drain the work queue. The TS `WorkPoller` long-polls and never returns on an
 * empty queue, but a Worker fetch handler must respond — so poll → ack until
 * empty here. The runner DO owns the lease (heartbeat + force-stop), so the
 * webhook never posts `stop`.
 */
async function drainWork(env: Env, anthropic: Anthropic): Promise<object[]> {
  const spawned: object[] = [];
  for (let i = 0; i < MAX_DRAIN; i++) {
    let work;
    try {
      // SDK auto-sends `anthropic-beta: managed-agents-2026-04-01` — no extra betas needed.
      work = await anthropic.beta.environments.work.poll(env.ANTHROPIC_ENVIRONMENT_ID, {
        reclaim_older_than_ms: 2000,
      });
    } catch (e) {
      // /work/poll can 404 ("EnvironmentInstance not found for session …")
      // when it dequeues a Redis entry whose session is gone. The stale entry
      // is consumed server-side, so retrying moves past it; anything else is a
      // config/transient failure — stop and recover on the next webhook.
      const { status, requestID } = (e ?? {}) as { status?: number; requestID?: string | null };
      console.warn(
        `[webhook] poll failed status=${status ?? "?"} request_id=${requestID ?? "?"} — ${status === 404 ? "skipping" : "stopping drain"}`,
      );
      if (status === 404) continue;
      break;
    }
    if (!work) break;
    // Log the claimed work item — explicit allowlist so `actor` (PII) and
    // `metadata` (user-provided) never reach the log.
    console.log(
      `[webhook] polled work=${JSON.stringify({
        id: work.id,
        environment_id: work.environment_id,
        data: { type: work.data.type, id: work.data.id },
        created_at: work.created_at,
        acknowledged_at: work.acknowledged_at,
        latest_heartbeat_at: work.latest_heartbeat_at,
      })}`,
    );
    if (work.data.type !== "session") continue;

    const sessionId = work.data.id;
    try {
      // The client is authed with the environment key — the single credential
      // for poll / ack / stop — so no per-call Authorization header is needed.
      await anthropic.beta.environments.work.ack(work.id, {
        environment_id: env.ANTHROPIC_ENVIRONMENT_ID,
      });

      const stub = env.RUNNER.get(env.RUNNER.idFromName(sessionId));
      const wasLive = await stub.isLive();
      if (!wasLive) {
        await stub.start({
          sessionId,
          environmentKey: env.ANTHROPIC_ENVIRONMENT_KEY,
          workId: work.id,
          environmentId: env.ANTHROPIC_ENVIRONMENT_ID,
        });
      }
      spawned.push({ session_id: sessionId, work_id: work.id, created: !wasLive });
    } catch (e) {
      // Skip and keep draining; the lease lapses and the next webhook reclaims.
      const detail = errDetail(e);
      console.warn(`[webhook] FAILED work=${work.id} session=${sessionId}: ${detail}`);
      spawned.push({ session_id: sessionId, work_id: work.id, error: detail });
    }
  }
  return spawned;
}

export default {
  async fetch(req: Request, env: Env): Promise<Response> {
    const anthropic = new Anthropic({
      authToken: env.ANTHROPIC_ENVIRONMENT_KEY,
      baseURL: env.ANTHROPIC_BASE_URL,
      // Pass the whsec_ secret as-is: the SDK decodes its URL-safe base64 internally.
      webhookKey: env.ANTHROPIC_WEBHOOK_SECRET,
    });

    const body = await req.text();
    let event: ReturnType<typeof anthropic.beta.webhooks.unwrap>;
    try {
      event = anthropic.beta.webhooks.unwrap(body, { headers: Object.fromEntries(req.headers) });
    } catch (e) {
      console.warn(`[webhook] signature reject: ${e instanceof Error ? e.constructor.name : "Error"}`);
      return new Response("signature verification failed", { status: 401 });
    }

    if (event.data.type !== "session.status_run_started") {
      return Response.json({ status: "ignored", event_type: event.data.type });
    }

    const spawned = await drainWork(env, anthropic);
    return Response.json({ status: "ok", spawned });
  },
};

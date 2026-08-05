/**
 * Runs inside the Vercel Sandbox — TS analogue of sandbox_runner.py.
 *
 * `client.beta.environments.work.worker(...).handleItem()` is the whole runner:
 * it builds the per-session `AgentToolContext` at `workdir` and downloads the
 * agent's skills into `{workdir}/skills/<name>/`, then runs a `SessionToolRunner`
 * (heartbeat + reconcile + event stream + tool dispatch + result posting) for
 * the session, and force-stops the work item on exit. With no per-item arguments
 * it reads the same `ANTHROPIC_*` env vars `ant beta:worker poll --on-work` sets,
 * which the webhook injects when it creates the sandbox.
 *
 * Idle policy is the SDK default: the runner stays alive for as long as the
 * session has activity and exits `DEFAULT_MAX_IDLE_MS` (60s) after
 * `session.status_idle` with `stop_reason: end_turn`; any other event —
 * including `requires_action` idle, where the agent is blocked on the sandbox —
 * resets the clock.
 *
 * Env vars (set per-command by the webhook, not on the sandbox default env):
 *   ANTHROPIC_BASE_URL        - API base URL
 *   ANTHROPIC_ENVIRONMENT_KEY - the environment key: the runner's single
 *                               credential. Authenticates the client (skill
 *                               download) and, threaded through
 *                               EnvironmentWorker, every poll / heartbeat /
 *                               event-stream / force-stop call.
 *   ANTHROPIC_SESSION_ID      - session id
 *   ANTHROPIC_ENVIRONMENT_ID  - environment id
 *   ANTHROPIC_WORK_ID         - work item id
 *   WORKDIR                   - agent working tree (default /workspace)
 */
import Anthropic from "@anthropic-ai/sdk";

// Created + chown'd by the webhook before the runner starts.
const WORKDIR = process.env.WORKDIR ?? "/workspace";

const ctrl = new AbortController();
process.once("SIGTERM", () => ctrl.abort());
process.once("SIGINT", () => ctrl.abort());

// The environment key is the only credential: the client uses it (skill
// download), and the worker threads it through every poll / heartbeat /
// event-stream / force-stop call. baseURL from ANTHROPIC_BASE_URL.
const environmentKey = process.env.ANTHROPIC_ENVIRONMENT_KEY;
const client = new Anthropic({
  authToken: environmentKey,
  logLevel: "info", // surface worker lifecycle (start/idle/heartbeat shutdown) in sandbox logs
});

console.log(`[runner] attaching session=${process.env.ANTHROPIC_SESSION_ID} work=${process.env.ANTHROPIC_WORK_ID}`);
try {
  await client.beta.environments.work
    .worker({
      environmentKey,
      workdir: WORKDIR,
      unrestrictedPaths: true,
      signal: ctrl.signal,
    })
    .handleItem();
} finally {
  console.log("[runner] done");
}

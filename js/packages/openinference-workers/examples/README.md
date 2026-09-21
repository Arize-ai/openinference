# Cloudflare AI, Workers, and Durable Objects examples

The AI routes call real Cloudflare-hosted models using the instrumented `env.AI` binding. They demonstrate:

```text
worker.request [CHAIN]
└── Workers AI.run [LLM]

worker.request [CHAIN]
└── ai.object [CHAIN]
    └── Workers AI.run [LLM]
```

`/ai` calls a small text model; `/ai/stream` returns its SSE stream unchanged;
`/ai/object` calls the model from a Durable Object; `/ai/tools` requests a weather
tool call from a tool-capable model. `/ai/masked` and `/ai/masked/stream` hide both
input and output. `/ai/error` exercises a nonexistent model and `/ai/suppressed`
makes a real model call without exporting spans. These routes use fixed prompts
and bounded output tokens. Running the verifier makes eight model calls and can
incur Cloudflare usage charges. The tool example verifies the generated tool call;
it does not execute a weather service.

Optionally set `AI_GATEWAY_ID` in Wrangler vars to route these calls through an
existing AI Gateway. Gateway routing is not required by the default example.
Other providers' SDK instrumentation is outside this example's scope.

The same application also runs a counter under Wrangler/workerd and celld. A Worker calls a SQLite-backed Durable Object that increments a persisted counter. Each normal request exports:

```text
worker.request [CHAIN]
└── counter.request [CHAIN]
    └── counter.increment [TOOL]
```

The Worker attaches a session, user, metadata, and tags. Inputs are masked. `/error` records an exception and returns HTTP 500; `/suppressed` runs the local handler without recording a span. Suppression and application metadata are local context, not an HTTP propagation protocol.

## Build

From `js/`:

```sh
pnpm install --frozen-lockfile
pnpm --filter @arizeai/openinference-workers... run build
```

Start Phoenix at `http://localhost:6006`. For example, use `uvx arize-phoenix serve`. The verification command requires the Phoenix `px` CLI on PATH.

## Cloudflare Workers and Durable Objects locally

From this package:

```sh
pnpm run dev:cloudflare --port 8787
```

In another terminal:

```sh
curl 'http://localhost:8787/?session=review&counter=review'
PHOENIX_HOST=http://localhost:6006 node examples/verify.mjs http://localhost:8787 openinference-workers-cloudflare
```

The script makes 12 concurrent requests to the same Durable Object, verifies persisted counter updates, and checks the spans read back from Phoenix, including exact parent relationships, isolated sessions, masking, suppression, and error recording. It prints the run ID and trace IDs. Normal verification expects 41 spans: 36 from concurrent requests, two error spans, and three spans from a request with an inbound remote parent. Runs use unique session IDs, so repeated runs can accumulate in the same Phoenix project.

## Verify LLM tracing

Both verification scripts default to `PHOENIX_PROJECT` when it is set. An explicit
project argument overrides it. Reuse the same project for local, deployed, and
celld testing; the exporter configuration must use that same name. For credentials
and settings stored in the package's ignored `.env` file, load them with Node:

```sh
node --env-file=.env examples/verify-ai.mjs https://YOUR-WORKER.workers.dev
```

Do not append timestamps to the project name. Each verification run already uses
unique session IDs to distinguish its spans from previous runs.

With the Worker running (locally or deployed), run:

```sh
PHOENIX_HOST=http://localhost:6006 node examples/verify-ai.mjs http://localhost:8787 openinference-workers-cloudflare
```

For a deployed Worker, replace the URL and Phoenix project and configure `px` for
the public Phoenix instance. This checks seven LLM spans, actual returned output,
stream assembly, model, token counts, tools, masking, suppression, error status,
session propagation, and parent relationships. Local Wrangler also makes real
remote inference calls and requires Cloudflare authentication. Consume streamed
responses fully (for example, `curl -N http://localhost:8787/ai/stream`).

## celld

From this package:

```sh
pnpm run build:example
docker compose -p openinference-workers-example -f examples/celld/compose.yaml up -d
PHOENIX_HOST=http://localhost:6006 node examples/verify.mjs http://localhost:9877 openinference-workers-celld
```

For LLM verification, first deploy the Cloudflare example. Set `AI_WORKER_URL` in
`examples/celld/wrangler.jsonc` to that deployment's origin. Configure celld and the
deployed Worker to export to the same publicly reachable Phoenix project (including
authorization via ignored local configuration when required). Recreate the celld
container and run `node examples/verify-ai.mjs http://localhost:9877 <project>`.
The local application forwards W3C trace headers and the example session to the
Cloudflare Worker, where the native AI binding runs. This demonstrates one trace
across celld and Cloudflare; it does not emulate an AI binding inside celld.

The pinned celld image runs the compiled Worker bundle. Docker reaches Phoenix at `host.docker.internal:6006`; adjust the celld config if your collector is elsewhere. Counter state lives in a Docker volume and survives container restarts.

```sh
docker compose -p openinference-workers-example -f examples/celld/compose.yaml down
```

## Deployed Cloudflare verification

Authenticate with `pnpm exec wrangler login`. Deploy the same config with a publicly reachable collector base URL and the Phoenix project named in your environment:

```sh
pnpm exec wrangler deploy --config examples/cloudflare/wrangler.jsonc \
  --var OTEL_ENDPOINT:https://YOUR-COLLECTOR \
  --var PHOENIX_PROJECT:$PHOENIX_PROJECT
node examples/verify.mjs https://YOUR-WORKER.workers.dev
```

Set `PHOENIX_HOST` for `px` to the corresponding Phoenix server. Never use localhost as the collector URL for a deployed Worker. For an authenticated collector, run `pnpm exec wrangler secret put OTEL_AUTHORIZATION --config examples/cloudflare/wrangler.jsonc` and enter the complete authorization value (for example, `Bearer ...`). For local runs, copy `cloudflare/.dev.vars.example` to `cloudflare/.dev.vars`. Do not commit credentials.

Cloudflare authentication and a reachable collector are required for this check. Local workerd execution alone does not verify a Cloudflare deployment.

## Native runtime telemetry

The package manages its own OpenTelemetry context and exports application spans. celld native telemetry can run alongside it, but native spans use a separate context on the pinned runtime and can include the exporter’s HTTP requests. This example does not promise automatic parentage between native and application spans.

## Verification evidence

Verified with Wrangler 4.135.0 (workerd 1.20260918.1), a deployed Cloudflare Worker and SQLite Durable Object, and celld 0.5.0. The deployed run exported 41 application spans to authenticated public Phoenix: 12 concurrent three-span traces, two error spans, and a three-span trace with a remote parent. Context isolation, input masking, suppression, error status, and parent relationships passed read-back assertions.

Focused regression tests cover the runtime findings. Run `pnpm test` from the package. The HTTP error boundary is exercised by the real-runtime verification script: on celld, allowing the exception to escape the handler can cancel export before the error span reaches Phoenix.

Native AI verification also passed against local workerd (with remote inference),
a deployed Worker and Durable Object, and celld forwarding to the deployed AI
binding. Each run verified seven LLM spans for text generation, streaming, tools,
masking, and errors, plus a suppressed call. The direct runs produced 15 application
spans; the celld run produced 22 across the runtime boundary. The package has 31
regression tests, including stream cancellation, bounded capture, fragmented UTF-8
and tool deltas, final usage, masking, and export failure containment.

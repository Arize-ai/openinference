# Workers and Durable Objects examples

The same application runs under Wrangler/workerd and celld. A Worker calls a SQLite-backed Durable Object that increments a persisted counter. Each normal request exports:

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

The script makes 12 concurrent requests to the same Durable Object, verifies persisted counter updates, and checks the spans read back from Phoenix, including exact parent relationships, isolated sessions, masking, suppression, and error recording. It prints the run ID and trace IDs. Normal verification expects 41 spans: 36 from concurrent requests, two error spans, and three spans from a request with an inbound remote parent. Give each invocation an otherwise quiet project if running alongside other examples.

## celld

From this package:

```sh
pnpm run build:example
docker compose -p openinference-workers-example -f examples/celld/compose.yaml up -d
PHOENIX_HOST=http://localhost:6006 node examples/verify.mjs http://localhost:9877 openinference-workers-celld
```

The pinned celld image runs the compiled Worker bundle. Docker reaches Phoenix at `host.docker.internal:6006`; adjust the celld config if your collector is elsewhere. Counter state lives in a Docker volume and survives container restarts.

```sh
docker compose -p openinference-workers-example -f examples/celld/compose.yaml down
```

## Deployed Cloudflare verification

Authenticate with `pnpm exec wrangler login`. Deploy the same config with a publicly reachable collector base URL and a separate Phoenix project:

```sh
pnpm exec wrangler deploy --config examples/cloudflare/wrangler.jsonc \
  --var OTEL_ENDPOINT:https://YOUR-COLLECTOR \
  --var PHOENIX_PROJECT:openinference-workers-deployed
node examples/verify.mjs https://YOUR-WORKER.workers.dev openinference-workers-deployed
```

Set `PHOENIX_HOST` for `px` to the corresponding Phoenix server. Never use localhost as the collector URL for a deployed Worker. For an authenticated collector, run `pnpm exec wrangler secret put OTEL_AUTHORIZATION --config examples/cloudflare/wrangler.jsonc` and enter the complete authorization value (for example, `Bearer ...`). For local runs, copy `cloudflare/.dev.vars.example` to `cloudflare/.dev.vars`. Do not commit credentials.

Cloudflare authentication and a reachable collector are required for this check. Local workerd execution alone does not verify a Cloudflare deployment.

## Native runtime telemetry

The package manages its own OpenTelemetry context and exports application spans. celld native telemetry can run alongside it, but native spans use a separate context on the pinned runtime and can include the exporter’s HTTP requests. This example does not promise automatic parentage between native and application spans.

## Verification evidence

Verified with Wrangler 4.135.0 (workerd 1.20260918.1), a deployed Cloudflare Worker and SQLite Durable Object, and celld 0.5.0. The deployed run exported 41 application spans to authenticated public Phoenix: 12 concurrent three-span traces, two error spans, and a three-span trace with a remote parent. Context isolation, input masking, suppression, error status, and parent relationships passed read-back assertions.

Focused regression tests cover the runtime findings. Run `pnpm test` from the package. The HTTP error boundary is exercised by the real-runtime verification script: on celld, allowing the exception to escape the handler can cancel export before the error span reaches Phoenix.

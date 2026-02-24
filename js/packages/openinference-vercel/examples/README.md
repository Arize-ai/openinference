# Examples

## Prereqs

- Local Phoenix running and accepting OTLP traces.
- `OPENAI_API_KEY` exported in your shell.

Phoenix defaults used by these examples:

- `PHOENIX_COLLECTOR_ENDPOINT` (default: `http://localhost:6006/v1/traces`)
- `PHOENIX_PROJECT_NAME` (default: `openinference-vercel-examples`)

## Run

From `packages/openinference-vercel`:

```bash
pnpm i
pnpx tsx examples/ai-sdk-stream-tools.ts
pnpx tsx examples/ai-sdk-tool-loop-agent.ts
```

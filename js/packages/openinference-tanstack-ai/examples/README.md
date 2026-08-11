# Examples

## Prereqs

- Local Phoenix running and accepting OTLP traces.
- `OPENAI_API_KEY` exported in your shell.
- `ANTHROPIC_API_KEY` exported in your shell for the Anthropic example.

Phoenix defaults used by these examples:

- `PHOENIX_COLLECTOR_ENDPOINT` (default: `http://localhost:6006/v1/traces`)
- `PHOENIX_PROJECT_NAME` (default: `openinference-tanstack-ai-examples`)

## Run

From `packages/openinference-tanstack-ai`:

```bash
pnpm i
pnpx tsx examples/chat-with-tools.ts
```

Anthropic multi-tool example:

```bash
pnpx tsx examples/anthropic-multi-tool.ts
```

Anthropic non-streaming single-turn example:

```bash
pnpx tsx examples/non-streaming-chat.ts
```

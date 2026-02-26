# AGENTS.md

## Commands

```bash
pnpm install --frozen-lockfile -r   # MUST use pnpm, not npm
pnpm run -r build                   # includes prebuild (version files + symlinks)
pnpm run -r test
pnpm run type:check
pnpm run lint
pnpm run fmt:check
pnpm changeset                      # before merging PRs
```

Cross-package dependencies require the prebuild step — always build after changes.

## Instrumentor Pattern

All instrumentors extend `InstrumentationBase` and use `OITracer` (never a raw OTel tracer):

```typescript
export class MyInstrumentation extends InstrumentationBase {
  private oiTracer: OITracer;

  constructor({ traceConfig }: { traceConfig?: TraceConfigOptions } = {}) {
    super(/* instrumentation config */);
    this.oiTracer = new OITracer({ tracer: this.tracer, traceConfig });
  }
}
```

Every instrumentor **must**:
1. Respect `isTracingSuppressed()` from `@opentelemetry/core`
2. Propagate context attributes via `OITracer` (session ID, user ID, metadata, tags)
3. Support data masking via `TraceConfig`

## Testing

- Vite with **manual module mocking**: `instrumentation._modules[0].moduleExports = module`
- Tests must cover: suppress tracing, context propagation, trace config

## Core Helpers

Use `@arizeai/openinference-core` attribute helpers instead of setting semantic conventions by hand:

| Helper | Purpose |
| --- | --- |
| `getInputAttributes(input)` | `{ INPUT_VALUE, INPUT_MIME_TYPE }` — accepts `string` or `{ value, mimeType }` |
| `getOutputAttributes(output)` | `{ OUTPUT_VALUE, OUTPUT_MIME_TYPE }` — same overloads |
| `getToolAttributes({ name, parameters })` | `{ TOOL_NAME, TOOL_PARAMETERS }` — stringifies parameters |

## Typing Conventions

- **Use SDK types directly** via `import type` — no hand-rolled structural duplicates
- **Named object args**: `{ original, oiTracer }` not positional
- **No `any`**: strong types from the instrumented SDK only
- **Type guards** narrow to actual SDK types, not structural equivalents

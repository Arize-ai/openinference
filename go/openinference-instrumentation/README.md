# openinference-instrumentation (Go)

Customer-facing config layer for the OpenInference Go instrumentors. Provides suppression, context attribute propagation, and PII / sensitive-data masking. Mirrors the Python [`openinference-instrumentation`](../../python/openinference-instrumentation) package.

## Install

```bash
go get github.com/Arize-ai/openinference/go/openinference-instrumentation@latest
```

Requires Go 1.25+.

## Suppression

Skip OpenInference instrumentation for any LLM call descended from the returned context. Useful for evaluator / grader code that itself calls an LLM but should not appear in the customer's product trace:

```go
import "github.com/Arize-ai/openinference/go/openinference-instrumentation"

suppressedCtx := instrumentation.WithSuppression(ctx)
_, _ = evalClient.CreateChatCompletion(suppressedCtx, req)  // no span emitted
```

## Context attribute propagation

OTel span attributes do NOT inherit from parent to child — a customer setting `session.id` on a CHAIN span does NOT cause child LLM spans to carry it. The provider instrumentors (`openinference-instrumentation-openai`, `openinference-instrumentation-anthropic`) read these helpers from OTel baggage and auto-apply them to every LLM span descended from the context:

```go
ctx = instrumentation.WithSession(ctx, "session-abc")
ctx = instrumentation.WithUser(ctx, "user-xyz")
ctx = instrumentation.WithMetadata(ctx, `{"team":"platform"}`)
ctx = instrumentation.WithTags(ctx, "prod", "canary")  // []string, matches the spec

resp, _ := client.CreateChatCompletion(ctx, req)
// span carries session.id, user.id, metadata, tag.tags
```

For manual spans you author yourself, call `instrumentation.ApplyContextAttributes(ctx, span)` right after `tracer.Start`.

## Masking sensitive data

`TraceConfig` controls which OpenInference attributes the provider instrumentors emit. Configurable via env vars (the canonical OpenInference convention) or programmatically:

```bash
export OPENINFERENCE_HIDE_INPUTS=true
export OPENINFERENCE_HIDE_OUTPUTS=true
```

Or in code:

```go
import "github.com/Arize-ai/openinference/go/openinference-instrumentation"

cfg := instrumentation.TraceConfig{
    HideInputs:  true,   // drops input.value AND llm.input_messages.* AND llm.tools.*
    HideOutputs: true,   // drops output.value AND llm.output_messages.*
}
// Pass to the provider instrumentor:
// openaiotel.NewTransport(..., openaiotel.WithTraceConfig(cfg))
// anthropicotel.Middleware(..., anthropicotel.WithTraceConfig(cfg))
```

Full env-var table:

| Env var | What it does |
|---|---|
| `OPENINFERENCE_HIDE_INPUTS` | Strongest input-side flag. Replaces `input.value` with `__REDACTED__` AND drops `llm.input_messages.*` AND drops `llm.tools.*`. |
| `OPENINFERENCE_HIDE_OUTPUTS` | Strongest output-side flag. Replaces `output.value` with `__REDACTED__` AND drops `llm.output_messages.*`. |
| `OPENINFERENCE_HIDE_INPUT_MESSAGES` | Drops `llm.input_messages.*`; `input.value` still set. |
| `OPENINFERENCE_HIDE_OUTPUT_MESSAGES` | Drops `llm.output_messages.*`; `output.value` still set. |
| `OPENINFERENCE_HIDE_INPUT_TEXT` / `_OUTPUT_TEXT` | Keeps message structure (role, indices), redacts only `.content` with `__REDACTED__`. |
| `OPENINFERENCE_HIDE_LLM_INVOCATION_PARAMETERS` | Omits `llm.invocation_parameters`. |
| `OPENINFERENCE_HIDE_LLM_TOOLS` | Omits the `llm.tools.*` advertised-tools list. (Implied by `HIDE_INPUTS`.) |

Top-level values (`input.value` / `output.value`) are replaced with the `__REDACTED__` sentinel rather than omitted, so downstream consumers can distinguish "hidden" from "never recorded". Structural attribute families are dropped wholesale — the wire-format keys do not appear on the span at all. Token counts, model name, `llm.finish_reason`, and timing are never affected. Matches Python `TraceConfig.mask` behavior exactly.

## Send the trace to Arize AX or Phoenix

This package is exporter-agnostic — it's pure context/config plumbing. Wire it into your tracer-provider setup of choice:

- **Arize AX**: [`arize-otel-go`](https://github.com/Arize-ai/arize-otel-go) for the one-line OTLP/HTTP setup to `otlp.arize.com`.
- **Phoenix (self-hosted)**: standard OTel OTLP/HTTP exporter pointed at your Phoenix host (`http://localhost:6006/v1/traces` by default).

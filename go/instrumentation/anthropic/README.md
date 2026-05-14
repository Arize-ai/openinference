# openinference-instrumentation-anthropic (Go)

OTel middleware that traces calls made through [`anthropics/anthropic-sdk-go`](https://github.com/anthropics/anthropic-sdk-go) with OpenInference LLM spans.

## Install

```bash
go get github.com/Arize-ai/openinference/go/instrumentation/anthropic
```

## Use

```go
import (
    "github.com/anthropics/anthropic-sdk-go"
    "github.com/anthropics/anthropic-sdk-go/option"
    "go.opentelemetry.io/otel"

    anthropicotel "github.com/Arize-ai/openinference/go/instrumentation/anthropic"
)

client := anthropic.NewClient(
    option.WithMiddleware(anthropicotel.Middleware(otel.Tracer("my-app"))),
)

resp, err := client.Messages.New(ctx, anthropic.MessageNewParams{
    Model:     "claude-3-5-sonnet-latest",
    MaxTokens: 100,
    Messages: []anthropic.MessageParam{
        anthropic.NewUserMessage(anthropic.NewTextBlock("hi")),
    },
})
```

Every `/v1/messages` call now emits an LLM-kind span with the following attributes:

| Attribute | Source |
|-----------|--------|
| `openinference.span.kind` | `LLM` |
| `llm.system` / `llm.provider` | `anthropic` |
| `llm.model_name` | request `model` field, then overwritten by the canonical model in the response |
| `llm.invocation_parameters` | JSON with `max_tokens`, `temperature`, `top_p`, `top_k` |
| `llm.input_messages.{i}.message.role` / `.content` | system prompt (if any) + each message; tool-use blocks are skipped |
| `input.value` | last user message text |
| `output.value` | concatenated `content[].text` from the response |
| `llm.output_messages.0.message.role` / `.content` | `assistant` + response text |
| `llm.finish_reason` | response `stop_reason` |
| `llm.token_count.prompt` / `.completion` / `.total` | from response `usage` |
| `llm.token_count.prompt_details.cache_read` / `.cache_write` | when present |

## Streaming

Streaming responses (SSE `text/event-stream`) pass through unchanged so the caller's stream consumer keeps working. The middleware wraps the response body in a small adapter so the span's `End()` fires when the caller closes (or fully reads) the body — the span's duration reflects the actual time-to-last-token, not just the HTTP handshake. Output attributes (`output.value`, `llm.token_count.*`) are not populated for streaming spans today; future versions may parse the SSE delta stream to fill them in.

## Tool calls

The middleware ignores `tool_use` content blocks in the request and response — wrap those in dedicated TOOL spans yourself (see [openinference docs](https://github.com/Arize-ai/openinference/tree/main/spec)). Auto-instrumenting tool execution is out of scope because the tool runs in the caller's process, not in the SDK.

## Suppression and context attributes

The sibling `openinference/go/instrumentation` package gives customers control over what shows up on LLM spans without setting attributes manually on each one:

```go
import "github.com/Arize-ai/openinference/go/instrumentation"

// Suppression: evaluator/grader code that itself calls an LLM but
// should not appear in the customer's product trace.
ctx := instrumentation.WithSuppression(ctx)
resp, _ := client.Messages.New(ctx, params)        // no span emitted

// Context attributes propagate from baggage to every LLM span
// descended from ctx, even when the call is several layers deep.
ctx = instrumentation.WithSession(ctx, "session-abc")
ctx = instrumentation.WithUser(ctx, "user-xyz")
ctx = instrumentation.WithMetadata(ctx, `{"team":"platform"}`)
ctx = instrumentation.WithTags(ctx, "prod", "canary")  // typed []string, matching the spec
resp, _ := client.Messages.New(ctx, params)        // span has session.id, user.id, …
```

These use OTel baggage, so they cross goroutine boundaries the way customers already expect for distributed-tracing context.

## Masking sensitive data

The middleware honors the canonical OpenInference `OPENINFERENCE_HIDE_*` environment variables for PII / sensitive-data protection. Set any of these to `true` to redact the corresponding attribute family:

| Env var | What it does |
|---|---|
| `OPENINFERENCE_HIDE_INPUTS` | Replaces `input.value` with `__REDACTED__` AND drops `llm.input_messages.*` entirely. Strongest input-side flag. |
| `OPENINFERENCE_HIDE_OUTPUTS` | Replaces `output.value` with `__REDACTED__` AND drops `llm.output_messages.*` entirely. Strongest output-side flag. |
| `OPENINFERENCE_HIDE_INPUT_MESSAGES` | Drops `llm.input_messages.*` entirely; `input.value` still set. |
| `OPENINFERENCE_HIDE_OUTPUT_MESSAGES` | Drops `llm.output_messages.*` entirely; `output.value` still set. |
| `OPENINFERENCE_HIDE_INPUT_TEXT` / `_OUTPUT_TEXT` | Keeps message structure (role, indices) and redacts only the `.content` field with `__REDACTED__`. |
| `OPENINFERENCE_HIDE_LLM_INVOCATION_PARAMETERS` | Omits `llm.invocation_parameters`. |
| `OPENINFERENCE_HIDE_LLM_TOOLS` | Omits the `llm.tools.*` advertised-tools list. (Implied by `HIDE_INPUTS`.) |

Top-level values (`input.value` / `output.value`) are replaced with the `__REDACTED__` sentinel rather than omitted, so downstream consumers can distinguish "hidden" from "never recorded". Structural attribute families (`llm.input_messages.*`, `llm.output_messages.*`, `llm.tools.*`) are dropped wholesale — the wire-format keys do not appear on the span at all. Token counts, model name, `llm.finish_reason`, and timing are never affected.

To override the env-driven config programmatically:

```go
import "github.com/Arize-ai/openinference/go/instrumentation"

mw := anthropicotel.Middleware(
    otel.Tracer("my-app"),
    anthropicotel.WithTraceConfig(instrumentation.TraceConfig{
        HideInputs:  true,
        HideOutputs: false,
    }),
)
```

`WithTraceConfig` fully replaces the env-derived config — set it once at construction. Matches the Python `TraceConfig` and JS `generateTraceConfig` patterns.

## Limitations (v0)

- Only `/v1/messages` is instrumented; the models endpoints and the Beta surfaces fall through untouched.
- `tool_use` content blocks in messages are not yet captured as `message.tool_calls` attributes; only text blocks make it onto the span.
- Streaming spans currently capture only request attributes (output and token counts arrive in SSE deltas the middleware does not yet parse). Span duration *does* correctly reflect end-of-stream because the body wrapper ends the span on `Read`-to-EOF or `Close`.

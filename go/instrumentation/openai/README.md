# openinference-instrumentation-openai (Go)

OTel `http.RoundTripper` that traces calls made through [`sashabaranov/go-openai`](https://github.com/sashabaranov/go-openai) with OpenInference LLM spans.

## Install

```bash
go get github.com/Arize-ai/openinference/go/instrumentation/openai
```

## Use

```go
import (
    "net/http"

    "github.com/sashabaranov/go-openai"
    "go.opentelemetry.io/otel"

    openaiotel "github.com/Arize-ai/openinference/go/instrumentation/openai"
)

cfg := openai.DefaultConfig(apiKey)
cfg.HTTPClient = &http.Client{
    Transport: openaiotel.NewTransport(http.DefaultTransport, otel.Tracer("my-app")),
}
client := openai.NewClientWithConfig(cfg)

resp, err := client.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
    Model: openai.GPT4o,
    Messages: []openai.ChatCompletionMessage{
        {Role: "user", Content: "hello"},
    },
})
```

Every `/v1/chat/completions` call now emits an LLM-kind span with:

| Attribute | Source |
|-----------|--------|
| `openinference.span.kind` | `LLM` |
| `llm.system` / `llm.provider` | `openai` |
| `llm.model_name` | request `model`, then overwritten by response `model` (canonical name) |
| `llm.invocation_parameters` | JSON with `temperature`, `top_p`, `max_tokens`, `n` |
| `llm.input_messages.{i}.message.role` / `.content` / `.name` / `.tool_call_id` | each request message |
| `llm.input_messages.{i}.message.tool_calls.{j}.tool_call.*` | tool calls on the i-th input message |
| `llm.tools.{i}.tool.json_schema` | tool advertisements (one per tool) |
| `input.value` | last user message text |
| `llm.output_messages.{i}.message.role` / `.content` | each response choice |
| `llm.output_messages.{i}.message.tool_calls.{j}.tool_call.*` | tool calls in response |
| `output.value` | text of the first choice (omitted if first choice is pure tool-use) |
| `llm.finish_reason` | finish_reason of the first choice |
| `llm.token_count.prompt` / `.completion` / `.total` | usage fields |
| `llm.token_count.prompt_details.cache_read` / `.audio` | from `prompt_tokens_details` |
| `llm.token_count.completion_details.reasoning` / `.audio` | from `completion_tokens_details` (o1/gpt-4o) |

## Streaming

Streaming responses (`text/event-stream`) pass through unchanged so the caller's stream consumer keeps working. The transport wraps the response body in a small adapter so the span's `End()` fires when the caller closes (or fully reads) the body — the span's duration reflects the actual time-to-last-token, not just the HTTP handshake. Output attributes (`output.value`, `llm.token_count.*`) are not populated for streaming spans today; future versions may parse the SSE delta stream to fill them in.

## Suppression and context attributes

The sibling `openinference/go/instrumentation` package gives customers control over what shows up on LLM spans without setting attributes manually on each one:

```go
import "github.com/Arize-ai/openinference/go/instrumentation"

// Suppression: evaluator/grader code that itself calls an LLM but
// should not appear in the customer's product trace.
ctx := instrumentation.WithSuppression(ctx)
resp, _ := client.CreateChatCompletion(ctx, req)   // no span emitted

// Context attributes propagate from baggage to every LLM span
// descended from ctx, even when the call is several layers deep.
ctx = instrumentation.WithSession(ctx, "session-abc")
ctx = instrumentation.WithUser(ctx, "user-xyz")
ctx = instrumentation.WithMetadata(ctx, `{"team":"platform"}`)
ctx = instrumentation.WithTags(ctx, "prod", "canary")  // typed []string, matching the spec
resp, _ := client.CreateChatCompletion(ctx, req)   // span has session.id, user.id, …
```

These use OTel baggage, so they cross goroutine boundaries the way customers already expect for distributed-tracing context.

## Masking sensitive data

The transport honors the canonical OpenInference `OPENINFERENCE_HIDE_*` environment variables for PII / sensitive-data protection. Set any of these to `true` to redact the corresponding attribute family:

| Env var | What it does |
|---|---|
| `OPENINFERENCE_HIDE_INPUTS` | Replaces `input.value` with `__REDACTED__` AND drops `llm.input_messages.*` entirely (including nested `tool_calls`, `name`, `tool_call_id`) AND drops `llm.tools.*`. Strongest input-side flag. |
| `OPENINFERENCE_HIDE_OUTPUTS` | Replaces `output.value` with `__REDACTED__` AND drops `llm.output_messages.*` entirely (including nested `tool_calls`). `llm.finish_reason` still set. Strongest output-side flag. |
| `OPENINFERENCE_HIDE_INPUT_MESSAGES` | Drops `llm.input_messages.*` entirely; `input.value` and `llm.tools.*` still set. |
| `OPENINFERENCE_HIDE_OUTPUT_MESSAGES` | Drops `llm.output_messages.*` entirely; `output.value` and `llm.finish_reason` still set. |
| `OPENINFERENCE_HIDE_INPUT_TEXT` / `_OUTPUT_TEXT` | Keeps message structure (role, indices, tool-call shells) and redacts only the `.content` field with `__REDACTED__`. |
| `OPENINFERENCE_HIDE_LLM_INVOCATION_PARAMETERS` | Omits `llm.invocation_parameters`. |
| `OPENINFERENCE_HIDE_LLM_TOOLS` | Omits the `llm.tools.*` advertised-tools list. (Implied by `HIDE_INPUTS`.) |

Top-level values (`input.value` / `output.value`) are replaced with the `__REDACTED__` sentinel rather than omitted, so downstream consumers can distinguish "hidden" from "never recorded". Structural attribute families (`llm.input_messages.*`, `llm.output_messages.*`, `llm.tools.*`) are dropped wholesale — the wire-format keys do not appear on the span at all. Token counts, model name, `llm.finish_reason`, and timing are never affected.

To override the env-driven config programmatically:

```go
import "github.com/Arize-ai/openinference/go/instrumentation"

cfg := openai.DefaultConfig(apiKey)
cfg.HTTPClient = &http.Client{
    Transport: openaiotel.NewTransport(
        http.DefaultTransport,
        otel.Tracer("my-app"),
        openaiotel.WithTraceConfig(instrumentation.TraceConfig{
            HideInputs:  true,
            HideOutputs: false,
        }),
    ),
}
```

`WithTraceConfig` fully replaces the env-derived config — set it once at construction. Matches the Python `TraceConfig` and JS `generateTraceConfig` patterns.

## Limitations (v0)

- Only `/v1/chat/completions` is instrumented. Embeddings, responses, completions, and image endpoints fall through to the base transport unchanged.
- For requests with `n > 1`, `llm.finish_reason` is set from the first choice only.
- Streaming spans currently capture only request attributes (output and token counts arrive in SSE deltas the transport does not yet parse). Span duration *does* correctly reflect end-of-stream because the body wrapper ends the span on `Read`-to-EOF or `Close`.

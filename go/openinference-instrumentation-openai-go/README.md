# openinference-instrumentation-openai-go (Go)

OTel middleware that traces calls made through the official [`openai/openai-go`](https://github.com/openai/openai-go) SDK with OpenInference LLM spans.

## Install

```bash
go get github.com/Arize-ai/openinference/go/openinference-instrumentation-openai-go
```

## Use

```go
import (
    "github.com/openai/openai-go"
    "github.com/openai/openai-go/option"
    "github.com/openai/openai-go/shared"
    "go.opentelemetry.io/otel"

    openaiotel "github.com/Arize-ai/openinference/go/openinference-instrumentation-openai-go"
)

client := openai.NewClient(
    option.WithAPIKey(apiKey),
    option.WithMiddleware(openaiotel.Middleware(otel.Tracer("my-app"))),
)

resp, err := client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
    Model: shared.ChatModelGPT4o,
    Messages: []openai.ChatCompletionMessageParamUnion{
        openai.UserMessage("hello"),
    },
})
```

Every `/v1/chat/completions` call now emits an LLM-kind span with:

| Attribute | Source |
|-----------|--------|
| `openinference.span.kind` | `LLM` |
| `llm.system` | `openai` |
| `llm.provider` | `openai` for direct OpenAI; `azure` when the request host is `*.openai.azure.com`, `*.services.ai.azure.com`, or `*.cognitiveservices.azure.com` |
| `llm.model_name` | request `model`, then overwritten by response `model` (canonical name) |
| `llm.invocation_parameters` | JSON of every non-content request field (model, temperature, top_p, max_tokens, max_completion_tokens, reasoning_effort, response_format, tool_choice, stream_options, presence_penalty, frequency_penalty, n, seed, …) |
| `llm.input_messages.{i}.message.role` / `.content` / `.name` / `.tool_call_id` | each request message |
| `llm.input_messages.{i}.message.function_call_*` | legacy request `function_call` fields |
| `llm.input_messages.{i}.message.tool_calls.{j}.tool_call.*` | tool calls on the i-th input message |
| `llm.tools.{i}.tool.json_schema` | tool advertisements (one per tool) |
| `input.value` | last user message text |
| `llm.output_messages.{i}.message.role` / `.content` | each response choice |
| `llm.output_messages.{i}.message.function_call_*` | legacy response `function_call` fields |
| `llm.output_messages.{i}.message.tool_calls.{j}.tool_call.*` | tool calls in response |
| `output.value` | text of the first choice (omitted if first choice is pure tool-use) |
| `llm.finish_reason` | finish_reason of the first choice |
| `llm.token_count.prompt` / `.completion` / `.total` | usage fields |
| `llm.token_count.prompt_details.cache_read` / `.audio` | from `prompt_tokens_details` |
| `llm.token_count.completion_details.reasoning` / `.audio` | from `completion_tokens_details` (o1/gpt-4o) |

## Azure OpenAI

Azure-hosted clients (created via [`openai-go/azure`](https://pkg.go.dev/github.com/openai/openai-go/azure)) are instrumented the same way — just pass `openaiotel.Middleware(...)` alongside `azure.WithEndpoint(...)`. The middleware recognises the Azure host suffixes and sets `llm.provider=azure` on those spans so backend queries can distinguish them from direct OpenAI traffic; `llm.system` stays `openai`.

## Streaming

Streaming responses (`text/event-stream`) pass through unchanged so the caller's stream consumer keeps working. The middleware wraps the response body in a small adapter so the span's `End()` fires when the caller closes (or fully reads) the body — the span's duration reflects the actual time-to-last-token, not just the HTTP handshake. Output attributes (`output.value`, `llm.token_count.*`) are not populated for streaming spans today; future versions may parse the SSE delta stream to fill them in.

## Suppression and context attributes

The sibling `openinference/go/openinference-instrumentation` package gives customers control over what shows up on LLM spans without setting attributes manually on each one:

```go
import "github.com/Arize-ai/openinference/go/openinference-instrumentation"

// Suppression: evaluator/grader code that itself calls an LLM but
// should not appear in the customer's product trace.
ctx := instrumentation.WithSuppression(ctx)
resp, _ := client.Chat.Completions.New(ctx, req)   // no span emitted

// Context attributes propagate from ctx to every LLM span descended
// from it, even when the call is several layers deep.
ctx = instrumentation.WithSession(ctx, "session-abc")
ctx = instrumentation.WithUser(ctx, "user-xyz")
ctx = instrumentation.WithMetadata(ctx, `{"team":"platform"}`)
ctx = instrumentation.WithTags(ctx, "prod", "canary")  // typed []string, matching the spec
resp, _ := client.Chat.Completions.New(ctx, req)   // span has session.id, user.id, …
```

These ride the standard `context.Context` (via unexported keys, not OTel baggage) so they flow through your call graph in-process but never leak out as `baggage` HTTP headers on downstream requests.

## Masking sensitive data

The middleware honors the canonical OpenInference `OPENINFERENCE_HIDE_*` environment variables for PII / sensitive-data protection. Set any of these to `true` to redact the corresponding attribute family:

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
import "github.com/Arize-ai/openinference/go/openinference-instrumentation"

client := openai.NewClient(
    option.WithAPIKey(apiKey),
    option.WithMiddleware(openaiotel.Middleware(
        otel.Tracer("my-app"),
        openaiotel.WithTraceConfig(instrumentation.TraceConfig{
            HideInputs:  true,
            HideOutputs: false,
        }),
    )),
)
```

`WithTraceConfig` fully replaces the env-derived config — set it once at construction. Matches the Python `TraceConfig` and JS `generateTraceConfig` patterns.

## Limitations (v0)

- Only `/v1/chat/completions` is instrumented. Embeddings, responses, completions, and image endpoints fall through to the next middleware unchanged.
- For requests with `n > 1`, `llm.finish_reason` is set from the first choice only.
- Streaming spans currently capture only request attributes (output and token counts arrive in SSE deltas the middleware does not yet parse). Span duration *does* correctly reflect end-of-stream because the body wrapper ends the span on `Read`-to-EOF or `Close`.

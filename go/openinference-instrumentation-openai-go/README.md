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

Every `/v1/chat/completions` and `/v1/responses` call now emits an LLM-kind span.

### Chat Completions

Chat Completions calls produce an `openai.chat.completions.create` span with:

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

### Responses

`client.Responses.New` and `client.Responses.NewStreaming` calls (`POST /v1/responses`) produce an `openai.responses.create` span. The attributes match the Python and JS OpenAI instrumentors:

```go
resp, err := client.Responses.New(ctx, responses.ResponseNewParams{
    Model:        "gpt-6.1-sol",
    Instructions: openai.String("Answer in one sentence."),
    Input:        responses.ResponseNewParamsInputUnion{OfString: openai.String("hello")},
})
```

| Attribute | Source |
|-----------|--------|
| `openinference.span.kind` / `llm.system` / `llm.provider` | as for Chat Completions, including the Azure host mapping |
| `llm.model_name` | request `model`, then overwritten by response `model` |
| `llm.invocation_parameters` | JSON of the request minus `input`, `instructions`, and `tools` (so `model`, `max_output_tokens`, `reasoning`, `previous_response_id`, `tool_choice`, `stream`, …) |
| `llm.input_messages.0` | `instructions`, as a `system` message (when set) |
| `llm.input_messages.{i}` | the string `input` as one `user` message, or one message per `input` item: `message` items keep their role and content (string as `.message.content`, part lists as `.message.contents.{k}.message_content.*`, with text parts as type `text` and `input_image` parts as type `image` plus `.message_content.image.image.url`); an item without a `type` counts as a message only when it has both `role` and `content`; `function_call` and `computer_call` items become an `assistant` message with `.message.tool_calls.0.tool_call.*`; `function_call_output` and `computer_call_output` items become a `tool` message with `.message.tool_call_id` (and, for `function_call_output`, the output as `.message.content`); `reasoning` items become a `reasoning` content part |
| `llm.tools.{i}.tool.json_schema` | each request tool definition |
| `input.value` / `input.mime_type` | the request body as JSON / `application/json`, with input image URLs redacted when images are hidden or are base64 data URIs over 32,000 characters |
| `llm.output_messages.{i}` | one message per `output` item, mapped the same way as input items (`message`, `function_call`, `reasoning`, `custom_tool_call`, `computer_call`, `web_search_call`, `file_search_call`) |
| `output.value` / `output.mime_type` | the response body (or, when streaming, the `response.completed` event's response) as JSON / `application/json` |
| `llm.token_count.prompt` / `.completion` / `.total` | `usage.input_tokens` / `output_tokens` / `total_tokens` |
| `llm.token_count.prompt_details.cache_read` / `.cache_write` | `usage.input_tokens_details.cached_tokens` / `cache_write_tokens` |
| `llm.token_count.completion_details.reasoning` | `usage.output_tokens_details.reasoning_tokens` |

Only response creation is traced. `GET /v1/responses/{id}`, `/cancel`, and the other `responses` sub-resources pass through without a span. A successful call sets the span status to `OK`, as Python and JS do; Chat Completions spans leave it unset. Non-2xx responses set the span status to `Error` and record no output or token attributes.

[`examples/responses`](examples/responses) runs a two-call tool loop: the first call returns a `function_call`, and the second sends the `function_call_output` with `previous_response_id`.

## Azure OpenAI

Azure-hosted clients (created via [`openai-go/azure`](https://pkg.go.dev/github.com/openai/openai-go/azure)) are instrumented the same way — just pass `openaiotel.Middleware(...)` alongside `azure.WithEndpoint(...)`. The middleware recognises the Azure host suffixes and sets `llm.provider=azure` on those spans so backend queries can distinguish them from direct OpenAI traffic; `llm.system` stays `openai`.

## Streaming

Streaming responses (`text/event-stream`) pass through unchanged so the caller's stream consumer keeps working. The middleware wraps the response body in a small adapter so the span's `End()` fires when the caller closes (or fully reads) the body — the span's duration reflects the actual time-to-last-token, not just the HTTP handshake. This applies to both Chat Completions and Responses.

For streamed Responses calls, the middleware also parses the SSE events as the caller reads them, without reading ahead of the caller or changing the bytes it receives. When the span ends, it records the response from the `response.completed` event: output messages, `output.value`, the response model, token counts, and status `OK`, as the Python and JS instrumentors do. A stream that is closed before `response.completed` ends the span with request attributes only. Malformed events are skipped, and a stream that hits a read error keeps its `Error` status.

Streamed Chat Completions spans still carry request attributes only.

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

Responses spans follow the same rules. Their content parts (`.message.contents.{k}.message_content.text`) are redacted by the `_TEXT` flags like `.content` is. `OPENINFERENCE_HIDE_INPUT_IMAGES` (implied by `HIDE_INPUTS`) drops input image URLs but keeps the parts' `image` type, and redacts the URLs in `input.value`. Because their `input.value` / `output.value` are JSON, `HIDE_INPUTS` / `HIDE_OUTPUTS` also drop `input.mime_type` / `output.mime_type`, so the `__REDACTED__` sentinel is not labeled as JSON.

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

- Only `/v1/chat/completions` and `POST /v1/responses` are instrumented. Embeddings, completions, and image endpoints fall through to the next middleware unchanged.
- Responses input files and audio parts are not recorded as message contents; they remain in `input.value`. The base64 image limit is fixed at Python's default of 32,000 characters, because the Go `TraceConfig` has no `base64_image_max_length` setting yet.
- For requests with `n > 1`, `llm.finish_reason` is set from the first choice only.
- Streamed Chat Completions spans capture only request attributes (output and token counts arrive in SSE deltas the middleware does not yet parse). Span duration *does* correctly reflect end-of-stream because the body wrapper ends the span on `Read`-to-EOF or `Close`.

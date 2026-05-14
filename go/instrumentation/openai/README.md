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

## Limitations (v0)

- Only `/v1/chat/completions` is instrumented. Embeddings, responses, completions, and image endpoints fall through to the base transport unchanged.
- For requests with `n > 1`, `llm.finish_reason` is set from the first choice only.
- No shared `OpenInferenceConfig` yet for suppression / context attribute propagation / masking — coming in a follow-up. Workaround: wrap calls in a parent CHAIN span and set `session.id` / `user.id` there; child LLM spans inherit via parent-child relationship.
- Streaming spans currently capture only request attributes (output and token counts arrive in SSE deltas the transport does not yet parse).

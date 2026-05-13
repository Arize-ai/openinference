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

## Limitations (v0)

- Only `/v1/messages` is instrumented; the models endpoints and the Beta surfaces fall through untouched.
- `tool_use` content blocks in messages are not yet captured as `message.tool_calls` attributes; only text blocks make it onto the span.
- No shared `OpenInferenceConfig` yet for suppression / context attribute propagation / masking — coming in a follow-up. Workaround: wrap calls in a parent CHAIN span and set `session.id` / `user.id` there; child LLM spans inherit via parent-child relationship.
- Streaming spans currently capture only request attributes (output and token counts arrive in SSE deltas the middleware does not yet parse).

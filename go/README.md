# OpenInference Go

Go packages for instrumenting AI/LLM applications with OpenInference semantic conventions over OpenTelemetry. Sibling of [`python/`](../python), [`java/`](../java), and [`js/`](../js).

## Packages

| Path | Status | Description |
|------|--------|-------------|
| `semconv/` | ✅ available | Attribute keys and value constants for OpenInference span kinds, messages, tools, embeddings, retrieval, cost/token counts, and prompts. |
| `instrumentation/` | ✅ available | Customer-facing config: `WithSuppression` for evaluator-style off-trace calls; `WithSession` / `WithUser` / `WithMetadata` / `WithTags` propagating via OTel baggage onto every LLM span emitted by the sibling instrumentors. Mirrors Python's `openinference-instrumentation`. |
| `instrumentation/anthropic/` | ✅ available | `option.Middleware` for `anthropics/anthropic-sdk-go` that emits OpenInference LLM spans on `/v1/messages` calls. |
| `instrumentation/openai/` | ✅ available | `http.RoundTripper` for `sashabaranov/go-openai` that emits OpenInference LLM spans on `/v1/chat/completions` calls. |

## Install

```bash
go get github.com/Arize-ai/openinference/go/semconv@latest
```

Requires Go 1.25+ (driven by the latest `anthropic-sdk-go` and OTel module floors).

## Quick start

```go
package main

import (
    "context"

    "go.opentelemetry.io/otel"
    "go.opentelemetry.io/otel/attribute"

    "github.com/Arize-ai/openinference/go/semconv"
)

func runAgent(ctx context.Context, userInput string) string {
    tracer := otel.Tracer("my-app")
    ctx, span := tracer.Start(ctx, "run_agent")
    defer span.End()

    span.SetAttributes(
        attribute.String(semconv.OpenInferenceSpanKind, semconv.SpanKindChain),
        attribute.String(semconv.InputValue, userInput),
        attribute.String(semconv.SessionID, "session-123"),
        attribute.String(semconv.UserID, "user-456"),
    )

    // ... your agent logic ...

    span.SetAttributes(attribute.String(semconv.OutputValue, "hello"))
    return "hello"
}
```

For an LLM span with message-array attributes, use the indexer helpers:

```go
span.SetAttributes(
    attribute.String(semconv.OpenInferenceSpanKind, semconv.SpanKindLLM),
    attribute.String(semconv.LLMModelName, "gpt-4o"),
    attribute.String(semconv.LLMProvider, semconv.LLMProviderOpenAI),
    attribute.String(semconv.LLMSystem, semconv.LLMSystemOpenAI),

    attribute.String(semconv.LLMInputMessageRoleKey(0), "system"),
    attribute.String(semconv.LLMInputMessageContentKey(0), "You are a helpful assistant."),
    attribute.String(semconv.LLMInputMessageRoleKey(1), "user"),
    attribute.String(semconv.LLMInputMessageContentKey(1), userInput),

    attribute.Int(semconv.LLMTokenCountPrompt, 42),
    attribute.Int(semconv.LLMTokenCountCompletion, 17),
    attribute.Int(semconv.LLMTokenCountTotal, 59),
)
```

## Sending traces to Arize

Pair `semconv` with the `arize-otel-go` helper (separate repo, [Arize-ai/arize-otel-go](https://github.com/Arize-ai/arize-otel-go)) which wires up the OTLP/HTTP exporter to `otlp.arize.com`. Or wire up OTel manually — `semconv` is exporter-agnostic.

## Status

- v0.1.0 — initial port of `python/openinference-semantic-conventions`.

## Contributing

`python/openinference-semantic-conventions/` is the source of truth. New attribute keys MUST be added there first, then ported into `go/semconv/`. Two layers catch drift:

- **`scripts/check_go_semconv.py`** (runs in CI) parses the Python source and asserts every Python wire-format key value is present in the Go package. This is the canonical cross-language check.
- **`semconv_test.go`** asserts each Go constant against the literal wire-format value it should hold. Cheaper Go-side typo guard; complements but does not replace the Python parity check.

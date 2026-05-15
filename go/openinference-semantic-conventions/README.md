# openinference-semantic-conventions (Go)

Attribute keys and value constants for OpenInference span kinds, messages, tools, embeddings, retrieval, cost/token counts, and prompts — the Go port of [`openinference-semantic-conventions`](../../python/openinference-semantic-conventions) (Python) / `openinference-semantic-conventions` (Java / JS).

Exported as plain string constants so they work directly with `go.opentelemetry.io/otel/attribute` helpers.

## Install

```bash
go get github.com/Arize-ai/openinference/go/openinference-semantic-conventions@latest
```

Requires Go 1.25+.

## Use

```go
import (
    "go.opentelemetry.io/otel/attribute"

    semconv "github.com/Arize-ai/openinference/go/openinference-semantic-conventions"
)

span.SetAttributes(
    attribute.String(semconv.OpenInferenceSpanKind, semconv.SpanKindLLM),
    attribute.String(semconv.LLMModelName, "gpt-4o"),
    attribute.String(semconv.LLMProvider, semconv.LLMProviderOpenAI),
)
```

Indexed attributes (array-valued, like message lists) are produced by helper functions:

```go
span.SetAttributes(
    attribute.String(semconv.LLMInputMessageRoleKey(0), "system"),
    attribute.String(semconv.LLMInputMessageContentKey(0), "You are a helpful assistant."),
    attribute.String(semconv.LLMInputMessageRoleKey(1), "user"),
    attribute.String(semconv.LLMInputMessageContentKey(1), userInput),
    attribute.Int(semconv.LLMTokenCountPrompt, 42),
)
```

## Send the trace to Arize AX or Phoenix

This package is exporter-agnostic — pair it with whatever OTel exporter your runtime needs:

- **Arize AX**: configure an OTLP/HTTP exporter pointed at `otlp.arize.com` with `space_id` + `api_key` headers, or use the [`arize-otel-go`](https://github.com/Arize-ai/arize-otel-go) helper which wraps the setup in one call.
- **Phoenix (self-hosted)**: configure OTLP/HTTP at `http://localhost:6006/v1/traces` (or your Phoenix host).

For LLM clients with first-party auto-instrumentation, use [`openinference-instrumentation-openai`](../openinference-instrumentation-openai) (when available) or [`openinference-instrumentation-anthropic-sdk-go`](../openinference-instrumentation-anthropic-sdk-go) (when available) — they call into this package automatically.

## Drift policy

The Python package is the source of truth. New attribute keys MUST be added to `python/openinference-semantic-conventions/` first, then ported here. Two layers catch drift in CI:

- **`scripts/check_go_semconv.py`** parses the Python source and asserts every wire-format key value is present in the Go package. Canonical cross-language check.
- **`semconv_test.go`** asserts each Go constant against its wire-format value. Cheaper Go-side typo guard.

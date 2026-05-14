// Package semconv defines the OpenInference semantic conventions for tracing
// AI/LLM applications. It mirrors the openinference-semantic-conventions
// package published for Python, Java, and JS.
//
// Attribute keys are exported as plain string constants so they can be passed
// directly to go.opentelemetry.io/otel/attribute helpers, e.g.:
//
//	span.SetAttributes(
//	    attribute.String(semconv.OpenInferenceSpanKind, string(semconv.SpanKindLLM)),
//	    attribute.String(semconv.InputValue, userPrompt),
//	    attribute.String(semconv.LLMModelName, "gpt-4o"),
//	)
//
// Indexed keys (arrays of messages, documents, etc.) are produced by the
// helper functions in indexers.go.
package semconv

// Package instrumentation is the customer-facing config layer for the
// OpenInference Go instrumentors. It mirrors the Python
// `openinference-instrumentation` package along three axes:
//
//   - suppress-tracing escape hatch for evaluator/grader calls;
//   - context attribute propagation (session.id, user.id, metadata,
//     tag.tags) that auto-applies to every LLM span;
//   - sensitive-data masking via TraceConfig, driven by the canonical
//     OPENINFERENCE_HIDE_* environment variables.
//
// Context attributes need a propagation channel because OTel span
// attributes don't auto-inherit from parent to child — a customer
// setting session.id on a CHAIN span does NOT cause the child LLM
// spans to carry session.id. We use unexported `context.Context` keys
// (the same pattern as Java's ContextKey, JS's createContextKey, and
// Python's OTel context values) so the values flow through the
// customer's call graph in-process but never leave the process. In
// particular, these values are NOT placed in OTel baggage, so an
// application running a baggage propagator will not serialize them
// into outbound HTTP headers on downstream calls (e.g. LLM provider
// requests). This matters for `metadata`, which callers may legitimately
// populate with tenant IDs, workflow labels, or other internal data
// that should not cross trust boundaries.
//
// Typical use:
//
//	import "github.com/Arize-ai/openinference/go/openinference-instrumentation"
//
//	ctx = instrumentation.WithSession(ctx, "session-abc")
//	ctx = instrumentation.WithUser(ctx, "user-xyz")
//	resp, _ := client.CreateChatCompletion(ctx, req)  // span carries session.id, user.id
//
//	// Skip instrumentation entirely for a sub-call (e.g. evaluator code):
//	suppressedCtx := instrumentation.WithSuppression(ctx)
//	_, _ = evalClient.CreateChatCompletion(suppressedCtx, req)
//
// For PII / sensitive-data masking, set OPENINFERENCE_HIDE_INPUTS=true
// (or any of the other OPENINFERENCE_HIDE_* env vars) before running
// the app — both the openai and anthropic instrumentors pick it up at
// construction time. See traceconfig.go for the full set of flags and
// the WithTraceConfig functional option each instrumentor exposes for
// in-code overrides.
package instrumentation

import (
	"context"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"

	"github.com/Arize-ai/openinference/go/openinference-semantic-conventions"
)

// Unexported key types ensure these values can only be set via the
// With* helpers in this package — no risk of accidental key collisions
// with other libraries' context values, and (unlike baggage) they
// cannot escape the process via a propagator.
type (
	suppressKey struct{}
	sessionKey  struct{}
	userKey     struct{}
	metadataKey struct{}
	tagsKey     struct{}
)

// WithSuppression returns a context that suppresses OpenInference LLM
// instrumentation for any provider call descended from it. Use this in
// evaluator/grader code that itself calls an LLM but should not appear
// in the customer's product trace.
//
// Provider instrumentors check IsSuppressed at the top of their
// middleware/transport and pass the request through unchanged when set.
func WithSuppression(ctx context.Context) context.Context {
	return context.WithValue(ctx, suppressKey{}, true)
}

// IsSuppressed reports whether ctx was marked by WithSuppression.
func IsSuppressed(ctx context.Context) bool {
	v, _ := ctx.Value(suppressKey{}).(bool)
	return v
}

// WithSession returns a context carrying sessionID. Provider
// instrumentors apply it to every LLM span descended from this context
// as the OpenInference session.id attribute.
//
// Returns ctx unchanged if sessionID is empty.
func WithSession(ctx context.Context, sessionID string) context.Context {
	if sessionID == "" {
		return ctx
	}
	return context.WithValue(ctx, sessionKey{}, sessionID)
}

// WithUser returns a context carrying userID. Applied as the
// OpenInference user.id attribute.
//
// Returns ctx unchanged if userID is empty.
func WithUser(ctx context.Context, userID string) context.Context {
	if userID == "" {
		return ctx
	}
	return context.WithValue(ctx, userKey{}, userID)
}

// WithMetadata returns a context carrying a free-form metadata JSON
// string. Applied as the OpenInference metadata attribute. Caller is
// responsible for JSON-encoding the map.
//
// Returns ctx unchanged if metadataJSON is empty.
func WithMetadata(ctx context.Context, metadataJSON string) context.Context {
	if metadataJSON == "" {
		return ctx
	}
	return context.WithValue(ctx, metadataKey{}, metadataJSON)
}

// WithTags returns a context carrying a list of categorical tags.
// Applied as the OpenInference tag.tags attribute, which is typed as a
// string list (the spec, the Python SDK, and the Arize UI all treat it
// as []string).
//
// Returns ctx unchanged if no tags are provided.
func WithTags(ctx context.Context, tags ...string) context.Context {
	if len(tags) == 0 {
		return ctx
	}
	// Defensive copy so a caller mutating their slice afterwards
	// cannot retroactively change what the span will record.
	copied := make([]string, len(tags))
	copy(copied, tags)
	return context.WithValue(ctx, tagsKey{}, copied)
}

// ApplyContextAttributes copies any OpenInference context attributes
// from ctx onto span. Provider instrumentors call this once per LLM
// span (right after Start) so customer-set session.id / user.id /
// metadata / tags appear on every LLM span in the trace, not just on
// the manually-instrumented CHAIN span.
func ApplyContextAttributes(ctx context.Context, span trace.Span) {
	if v, ok := ctx.Value(sessionKey{}).(string); ok && v != "" {
		span.SetAttributes(attribute.String(semconv.SessionID, v))
	}
	if v, ok := ctx.Value(userKey{}).(string); ok && v != "" {
		span.SetAttributes(attribute.String(semconv.UserID, v))
	}
	if v, ok := ctx.Value(metadataKey{}).(string); ok && v != "" {
		span.SetAttributes(attribute.String(semconv.Metadata, v))
	}
	if v, ok := ctx.Value(tagsKey{}).([]string); ok && len(v) > 0 {
		span.SetAttributes(attribute.StringSlice(semconv.TagTags, v))
	}
}

// Package instrumentation is the customer-facing config layer for the
// OpenInference Go instrumentors. It mirrors the Python
// `openinference-instrumentation` package along three axes:
//
//   - suppress-tracing escape hatch for evaluator/grader calls;
//   - context attribute propagation (session.id, user.id, metadata,
//     tag.tags) that auto-applies to every LLM span via OTel baggage;
//   - sensitive-data masking via TraceConfig, driven by the canonical
//     OPENINFERENCE_HIDE_* environment variables.
//
// Context attributes need a propagation channel because OTel span
// attributes don't auto-inherit from parent to child — a customer
// setting session.id on a CHAIN span does NOT cause the child LLM
// spans to carry session.id. We use OTel baggage so the values cross
// goroutine and process boundaries the way customers already expect
// for distributed-tracing context.
//
// Typical use:
//
//	import "github.com/Arize-ai/openinference/go/instrumentation"
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
	"encoding/json"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/baggage"
	"go.opentelemetry.io/otel/trace"

	"github.com/Arize-ai/openinference/go/semconv"
)

// Baggage keys recognised by OpenInference instrumentors. Customers can
// either set them via the With* helpers below, or directly via the OTel
// baggage API. The literal values mirror the corresponding OpenInference
// attribute names so a customer's mental model is "the baggage key IS
// the attribute key" — except BaggageTags, whose baggage value is a
// JSON-encoded array because OTel baggage values must be strings.
const (
	BaggageSessionID = "session.id"
	BaggageUserID    = "user.id"
	BaggageMetadata  = "metadata"
	BaggageTags      = "tag.tags"
)

type suppressKey struct{}

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

// WithSession returns a context carrying sessionID via baggage. Provider
// instrumentors apply it to every LLM span descended from this context
// as the OpenInference session.id attribute.
//
// Returns ctx unchanged if sessionID is empty.
func WithSession(ctx context.Context, sessionID string) context.Context {
	return setBaggage(ctx, BaggageSessionID, sessionID)
}

// WithUser returns a context carrying userID via baggage. Applied as
// the OpenInference user.id attribute.
func WithUser(ctx context.Context, userID string) context.Context {
	return setBaggage(ctx, BaggageUserID, userID)
}

// WithMetadata returns a context carrying a free-form metadata JSON
// string via baggage. Applied as the OpenInference metadata attribute.
// Caller is responsible for JSON-encoding the map.
func WithMetadata(ctx context.Context, metadataJSON string) context.Context {
	return setBaggage(ctx, BaggageMetadata, metadataJSON)
}

// WithTags returns a context carrying a list of categorical tags via
// baggage. Applied as the OpenInference tag.tags attribute, which is
// typed as a string list (the spec, the Python SDK, and the Arize UI
// all treat it as []string).
//
// Tags are JSON-encoded inside the baggage value because OTel baggage
// values must be strings; ApplyContextAttributes decodes them back to
// a string slice for attribute.StringSlice on the span.
//
// Returns ctx unchanged if no tags are provided.
func WithTags(ctx context.Context, tags ...string) context.Context {
	if len(tags) == 0 {
		return ctx
	}
	encoded, err := json.Marshal(tags)
	if err != nil {
		return ctx
	}
	return setBaggage(ctx, BaggageTags, string(encoded))
}

// ApplyContextAttributes copies any recognised OpenInference baggage
// members from ctx onto span. Provider instrumentors call this once
// per LLM span (right after Start) so customer-set session.id / user.id
// / metadata / tags appear on every LLM span in the trace, not just on
// the manually-instrumented CHAIN span.
func ApplyContextAttributes(ctx context.Context, span trace.Span) {
	bag := baggage.FromContext(ctx)
	if v := memberValue(bag, BaggageSessionID); v != "" {
		span.SetAttributes(attribute.String(semconv.SessionID, v))
	}
	if v := memberValue(bag, BaggageUserID); v != "" {
		span.SetAttributes(attribute.String(semconv.UserID, v))
	}
	if v := memberValue(bag, BaggageMetadata); v != "" {
		span.SetAttributes(attribute.String(semconv.Metadata, v))
	}
	if v := memberValue(bag, BaggageTags); v != "" {
		// Decode the JSON-encoded slice back to []string and emit
		// as a typed list attribute, matching the OpenInference spec
		// (and the Python implementation).
		var tags []string
		if err := json.Unmarshal([]byte(v), &tags); err == nil && len(tags) > 0 {
			span.SetAttributes(attribute.StringSlice(semconv.TagTags, tags))
		}
	}
}

func memberValue(bag baggage.Baggage, key string) string {
	m := bag.Member(key)
	if m.Key() == "" {
		return ""
	}
	return m.Value()
}

// setBaggage is the internal builder used by the With* helpers.
// Returns ctx unchanged on a zero-length value or on a baggage-API
// rejection (e.g., a value containing reserved characters); we never
// want a tracing helper to error a customer's request path.
func setBaggage(ctx context.Context, key, value string) context.Context {
	if value == "" {
		return ctx
	}
	bag := baggage.FromContext(ctx)
	m, err := baggage.NewMemberRaw(key, value)
	if err != nil {
		return ctx
	}
	updated, err := bag.SetMember(m)
	if err != nil {
		return ctx
	}
	return baggage.ContextWithBaggage(ctx, updated)
}

package instrumentation_test

import (
	"context"
	"reflect"
	"testing"

	"go.opentelemetry.io/otel/attribute"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"

	"github.com/Arize-ai/openinference/go/instrumentation"
)

func TestSuppression(t *testing.T) {
	ctx := context.Background()
	if instrumentation.IsSuppressed(ctx) {
		t.Error("vanilla context should not be suppressed")
	}
	sup := instrumentation.WithSuppression(ctx)
	if !instrumentation.IsSuppressed(sup) {
		t.Error("WithSuppression should mark context as suppressed")
	}
	if instrumentation.IsSuppressed(ctx) {
		t.Error("WithSuppression must not mutate the parent context")
	}
}

func TestApplyContextAttributes_NoBaggageNoOp(t *testing.T) {
	recorder := tracetest.NewSpanRecorder()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
	ctx := context.Background()
	_, span := tp.Tracer("test").Start(ctx, "test")
	instrumentation.ApplyContextAttributes(ctx, span)
	span.End()

	spans := recorder.Ended()
	if len(spans) != 1 {
		t.Fatalf("expected 1 span, got %d", len(spans))
	}
	for _, kv := range spans[0].Attributes() {
		for _, k := range []string{"session.id", "user.id", "metadata", "tag.tags"} {
			if string(kv.Key) == k {
				t.Errorf("unexpected attribute %s = %v", k, kv.Value.AsInterface())
			}
		}
	}
}

func TestApplyContextAttributes_BaggagePropagation(t *testing.T) {
	recorder := tracetest.NewSpanRecorder()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))

	ctx := context.Background()
	ctx = instrumentation.WithSession(ctx, "session-abc")
	ctx = instrumentation.WithUser(ctx, "user-xyz")
	ctx = instrumentation.WithMetadata(ctx, `{"team":"platform"}`)
	ctx = instrumentation.WithTags(ctx, "prod", "canary")

	_, span := tp.Tracer("test").Start(ctx, "test")
	instrumentation.ApplyContextAttributes(ctx, span)
	span.End()

	spans := recorder.Ended()
	attrs := map[string]any{}
	for _, kv := range spans[0].Attributes() {
		attrs[string(kv.Key)] = kv.Value.AsInterface()
	}

	if got := attrs["session.id"]; got != "session-abc" {
		t.Errorf("session.id: got %v", got)
	}
	if got := attrs["user.id"]; got != "user-xyz" {
		t.Errorf("user.id: got %v", got)
	}
	if got := attrs["metadata"]; got != `{"team":"platform"}` {
		t.Errorf("metadata: got %v", got)
	}
	// tag.tags MUST be a string slice (matches the OpenInference spec
	// and Python). attribute.StringSlice serialises as []string via
	// AsInterface().
	gotTags, ok := attrs["tag.tags"].([]string)
	if !ok {
		t.Fatalf("tag.tags should be []string, got %T = %v", attrs["tag.tags"], attrs["tag.tags"])
	}
	if !reflect.DeepEqual(gotTags, []string{"prod", "canary"}) {
		t.Errorf("tag.tags: got %v want [prod canary]", gotTags)
	}
}

func TestWithTags_VariadicAndEmpty(t *testing.T) {
	ctx := context.Background()

	// No tags → ctx unchanged.
	got := instrumentation.WithTags(ctx)
	if got != ctx {
		t.Error("WithTags() with no args should return ctx unchanged")
	}

	// One tag.
	recorder := tracetest.NewSpanRecorder()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
	ctx2 := instrumentation.WithTags(context.Background(), "solo")
	_, span := tp.Tracer("test").Start(ctx2, "test")
	instrumentation.ApplyContextAttributes(ctx2, span)
	span.End()

	attrs := map[string]any{}
	for _, kv := range recorder.Ended()[0].Attributes() {
		attrs[string(kv.Key)] = kv.Value.AsInterface()
	}
	gotTags, ok := attrs["tag.tags"].([]string)
	if !ok || !reflect.DeepEqual(gotTags, []string{"solo"}) {
		t.Errorf("single-tag: got %v want [solo]", attrs["tag.tags"])
	}
}

func TestWithTags_PreservesTagsContainingCommas(t *testing.T) {
	// JSON encoding inside baggage protects against the comma-split
	// foot-gun a naive separator-based encoding would have.
	recorder := tracetest.NewSpanRecorder()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))

	ctx := instrumentation.WithTags(context.Background(), "team:platform,sre", "env:prod")
	_, span := tp.Tracer("test").Start(ctx, "test")
	instrumentation.ApplyContextAttributes(ctx, span)
	span.End()

	attrs := map[string]any{}
	for _, kv := range recorder.Ended()[0].Attributes() {
		attrs[string(kv.Key)] = kv.Value.AsInterface()
	}
	gotTags, ok := attrs["tag.tags"].([]string)
	if !ok {
		t.Fatalf("tag.tags should be []string, got %T", attrs["tag.tags"])
	}
	if !reflect.DeepEqual(gotTags, []string{"team:platform,sre", "env:prod"}) {
		t.Errorf("tags with embedded commas: got %v", gotTags)
	}
}

func TestApplyContextAttributes_PartialBaggage(t *testing.T) {
	recorder := tracetest.NewSpanRecorder()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))

	ctx := instrumentation.WithSession(context.Background(), "only-session")
	_, span := tp.Tracer("test").Start(ctx, "test")
	instrumentation.ApplyContextAttributes(ctx, span)
	span.End()

	spans := recorder.Ended()
	attrs := map[string]any{}
	for _, kv := range spans[0].Attributes() {
		attrs[string(kv.Key)] = kv.Value.AsInterface()
	}
	if attrs["session.id"] != "only-session" {
		t.Errorf("session.id: got %v", attrs["session.id"])
	}
	for _, k := range []string{"user.id", "metadata", "tag.tags"} {
		if _, present := attrs[k]; present {
			t.Errorf("unexpected attribute %s should not be set", k)
		}
	}
}

func TestStringHelpers_EmptyValueIsNoOp(t *testing.T) {
	ctx := context.Background()
	for _, fn := range []func(context.Context, string) context.Context{
		instrumentation.WithSession,
		instrumentation.WithUser,
		instrumentation.WithMetadata,
	} {
		got := fn(ctx, "")
		if got != ctx {
			t.Errorf("empty value should return ctx unchanged")
		}
	}
}

var _ = attribute.String

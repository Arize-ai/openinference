package anthropic_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"

	"github.com/Arize-ai/openinference/go/openinference-instrumentation"
	anthropicotel "github.com/Arize-ai/openinference/go/openinference-instrumentation-anthropic-sdk-go"
	"github.com/Arize-ai/openinference/go/openinference-semantic-conventions"
)

func TestMiddleware_HappyPath(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id": "msg_01ABC",
			"role": "assistant",
			"model": "claude-3-5-sonnet-20241022",
			"stop_reason": "end_turn",
			"content": [{"type":"text","text":"The capital of France is Paris."}],
			"usage": {
				"input_tokens": 12,
				"output_tokens": 8,
				"cache_creation_input_tokens": 0,
				"cache_read_input_tokens": 0
			}
		}`))
	}))
	defer server.Close()

	recorder := tracetest.NewSpanRecorder()
	tp := trace.NewTracerProvider(trace.WithSpanProcessor(recorder))
	tracer := tp.Tracer("test")

	client := anthropicsdk.NewClient(
		option.WithBaseURL(server.URL),
		option.WithAPIKey("test-key"),
		option.WithMiddleware(anthropicotel.Middleware(tracer)),
	)

	resp, err := client.Messages.New(context.Background(), anthropicsdk.MessageNewParams{
		Model:     "claude-3-5-sonnet-latest",
		MaxTokens: 100,
		Messages: []anthropicsdk.MessageParam{
			anthropicsdk.NewUserMessage(anthropicsdk.NewTextBlock("What is the capital of France?")),
		},
	})
	if err != nil {
		t.Fatalf("Messages.New: %v", err)
	}
	if resp == nil || len(resp.Content) == 0 {
		t.Fatal("expected non-empty response")
	}

	if err := tp.ForceFlush(context.Background()); err != nil {
		t.Fatalf("ForceFlush: %v", err)
	}

	spans := recorder.Ended()
	if len(spans) != 1 {
		t.Fatalf("expected 1 span, got %d", len(spans))
	}
	span := spans[0]
	attrs := attrMap(span.Attributes())

	want := map[string]any{
		semconv.OpenInferenceSpanKind:         semconv.SpanKindLLM,
		semconv.LLMSystem:                     semconv.LLMSystemAnthropic,
		semconv.LLMProvider:                   semconv.LLMProviderAnthropic,
		semconv.LLMModelName:                  "claude-3-5-sonnet-20241022", // response wins over request alias
		semconv.LLMFinishReason:               "end_turn",
		semconv.LLMInputMessageRoleKey(0):     "user",
		semconv.LLMInputMessageContentKey(0):  "What is the capital of France?",
		semconv.InputValue:                    "What is the capital of France?",
		semconv.OutputValue:                   "The capital of France is Paris.",
		semconv.LLMOutputMessageRoleKey(0):    "assistant",
		semconv.LLMOutputMessageContentKey(0): "The capital of France is Paris.",
		semconv.LLMTokenCountPrompt:           int64(12),
		semconv.LLMTokenCountCompletion:       int64(8),
		semconv.LLMTokenCountTotal:            int64(20),
	}
	for k, v := range want {
		if got := attrs[k]; got != v {
			t.Errorf("attr %s: got %v want %v", k, got, v)
		}
	}

	// Invocation params should include max_tokens.
	invocation, _ := attrs[semconv.LLMInvocationParameters].(string)
	if !strings.Contains(invocation, `"max_tokens":100`) {
		t.Errorf("invocation params missing max_tokens: %q", invocation)
	}
}

func TestMiddleware_SystemPromptCapturedAsMessage(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"id","role":"assistant","model":"m","stop_reason":"end_turn",
			"content":[{"type":"text","text":"hi"}],
			"usage":{"input_tokens":1,"output_tokens":1,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}
		}`))
	}))
	defer server.Close()

	recorder := tracetest.NewSpanRecorder()
	tp := trace.NewTracerProvider(trace.WithSpanProcessor(recorder))

	client := anthropicsdk.NewClient(
		option.WithBaseURL(server.URL),
		option.WithAPIKey("test-key"),
		option.WithMiddleware(anthropicotel.Middleware(tp.Tracer("test"))),
	)

	_, err := client.Messages.New(context.Background(), anthropicsdk.MessageNewParams{
		Model:     "claude-3-5-sonnet-latest",
		MaxTokens: 10,
		System: []anthropicsdk.TextBlockParam{
			{Text: "You are a helpful assistant."},
		},
		Messages: []anthropicsdk.MessageParam{
			anthropicsdk.NewUserMessage(anthropicsdk.NewTextBlock("hello")),
		},
	})
	if err != nil {
		t.Fatalf("Messages.New: %v", err)
	}
	_ = tp.ForceFlush(context.Background())

	spans := recorder.Ended()
	if len(spans) != 1 {
		t.Fatalf("expected 1 span, got %d", len(spans))
	}
	attrs := attrMap(spans[0].Attributes())

	if got := attrs[semconv.LLMInputMessageRoleKey(0)]; got != "system" {
		t.Errorf("expected message 0 to be system, got %v", got)
	}
	if got := attrs[semconv.LLMInputMessageContentKey(0)]; got != "You are a helpful assistant." {
		t.Errorf("system content: got %v", got)
	}
	if got := attrs[semconv.LLMInputMessageRoleKey(1)]; got != "user" {
		t.Errorf("expected message 1 to be user, got %v", got)
	}
}

func TestMiddleware_ErrorResponseDoesNotPolluteTokenCounts(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"type":"error","error":{"type":"rate_limit_error","message":"rate limited"}}`))
	}))
	defer server.Close()

	recorder := tracetest.NewSpanRecorder()
	tp := trace.NewTracerProvider(trace.WithSpanProcessor(recorder))

	client := anthropicsdk.NewClient(
		option.WithBaseURL(server.URL),
		option.WithAPIKey("test-key"),
		option.WithMaxRetries(0),
		option.WithMiddleware(anthropicotel.Middleware(tp.Tracer("test"))),
	)
	_, _ = client.Messages.New(context.Background(), anthropicsdk.MessageNewParams{
		Model:     "claude-3-5-sonnet-latest",
		MaxTokens: 10,
		Messages:  []anthropicsdk.MessageParam{anthropicsdk.NewUserMessage(anthropicsdk.NewTextBlock("hi"))},
	})
	_ = tp.ForceFlush(context.Background())

	spans := recorder.Ended()
	if len(spans) != 1 {
		t.Fatalf("expected 1 span, got %d", len(spans))
	}
	attrs := attrMap(spans[0].Attributes())

	// Token counts MUST NOT be set on error responses — emitting zeros
	// would be misleading (zero is a real value, indistinguishable from
	// "couldn't measure" otherwise).
	for _, k := range []string{
		semconv.LLMTokenCountPrompt,
		semconv.LLMTokenCountCompletion,
		semconv.LLMTokenCountTotal,
	} {
		if _, present := attrs[k]; present {
			t.Errorf("error response should not produce %s, got %v", k, attrs[k])
		}
	}
	// Output text MUST NOT be set from the error body.
	if _, present := attrs[semconv.OutputValue]; present {
		t.Errorf("error response should not set output.value")
	}
	// But the span SHOULD be marked as an error so downstream filters
	// can find failed LLM calls.
	if status := spans[0].Status(); status.Code.String() != "Error" {
		t.Errorf("expected span status Error, got %s", status.Code)
	}
}

func TestMiddleware_StreamingSpanEndsOnBodyClose(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("event: content_block_delta\ndata: {\"type\":\"content_block_delta\"}\n\n"))
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
	}))
	defer server.Close()

	recorder := tracetest.NewSpanRecorder()
	tp := trace.NewTracerProvider(trace.WithSpanProcessor(recorder))
	mw := anthropicotel.Middleware(tp.Tracer("test"))

	reqBody := strings.NewReader(`{"model":"claude-3-5-sonnet-latest","max_tokens":10,"messages":[{"role":"user","content":"hi"}],"stream":true}`)
	req, err := http.NewRequestWithContext(context.Background(), http.MethodPost, server.URL+"/v1/messages", reqBody)
	if err != nil {
		t.Fatalf("NewRequest: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := mw(req, http.DefaultTransport.RoundTrip)
	if err != nil {
		t.Fatalf("middleware: %v", err)
	}
	defer resp.Body.Close()

	if got := len(recorder.Ended()); got != 0 {
		t.Fatalf("span ended prematurely (%d) — streaming response should keep span open until body close", got)
	}

	if _, err := io.Copy(io.Discard, resp.Body); err != nil {
		t.Fatalf("drain body: %v", err)
	}
	if got := len(recorder.Ended()); got != 1 {
		t.Fatalf("span did not end on EOF: got %d ended spans", got)
	}

	span := recorder.Ended()[0]
	attrs := attrMap(span.Attributes())
	if attrs[semconv.OpenInferenceSpanKind] != semconv.SpanKindLLM {
		t.Errorf("span kind: got %v", attrs[semconv.OpenInferenceSpanKind])
	}
	if _, present := attrs[semconv.OutputValue]; present {
		t.Errorf("streaming span should not set output.value")
	}
}

func TestMiddleware_ResponseParseFailureRecordsError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{this is not valid json}`))
	}))
	defer server.Close()

	recorder := tracetest.NewSpanRecorder()
	tp := trace.NewTracerProvider(trace.WithSpanProcessor(recorder))
	mw := anthropicotel.Middleware(tp.Tracer("test"))

	reqBody := strings.NewReader(`{"model":"claude-3-5-sonnet-latest","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}`)
	req, err := http.NewRequestWithContext(context.Background(), http.MethodPost, server.URL+"/v1/messages", reqBody)
	if err != nil {
		t.Fatalf("NewRequest: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := mw(req, http.DefaultTransport.RoundTrip)
	if err != nil {
		t.Fatalf("middleware: %v", err)
	}
	defer resp.Body.Close()

	spans := recorder.Ended()
	if len(spans) != 1 {
		t.Fatalf("expected 1 ended span, got %d", len(spans))
	}
	var foundException bool
	for _, e := range spans[0].Events() {
		if e.Name == "exception" {
			foundException = true
			break
		}
	}
	if !foundException {
		t.Errorf("expected exception event for parse failure, got events: %+v", spans[0].Events())
	}
}

func TestMiddleware_SuppressedContextEmitsNoSpan(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"id","role":"assistant","model":"m","stop_reason":"end_turn",
			"content":[{"type":"text","text":"hi"}],
			"usage":{"input_tokens":1,"output_tokens":1,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}
		}`))
	}))
	defer server.Close()

	recorder := tracetest.NewSpanRecorder()
	tp := trace.NewTracerProvider(trace.WithSpanProcessor(recorder))
	client := anthropicsdk.NewClient(
		option.WithBaseURL(server.URL),
		option.WithAPIKey("test-key"),
		option.WithMiddleware(anthropicotel.Middleware(tp.Tracer("test"))),
	)

	ctx := instrumentation.WithSuppression(context.Background())
	_, err := client.Messages.New(ctx, anthropicsdk.MessageNewParams{
		Model: "claude-3-5-sonnet-latest", MaxTokens: 10,
		Messages: []anthropicsdk.MessageParam{anthropicsdk.NewUserMessage(anthropicsdk.NewTextBlock("hi"))},
	})
	if err != nil {
		t.Fatalf("Messages.New: %v", err)
	}
	_ = tp.ForceFlush(context.Background())

	if got := len(recorder.Ended()); got != 0 {
		t.Errorf("suppressed context should produce no span, got %d", got)
	}
}

func TestMiddleware_BaggagePropagatesToSpan(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"id","role":"assistant","model":"m","stop_reason":"end_turn",
			"content":[{"type":"text","text":"hi"}],
			"usage":{"input_tokens":1,"output_tokens":1,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}
		}`))
	}))
	defer server.Close()

	recorder := tracetest.NewSpanRecorder()
	tp := trace.NewTracerProvider(trace.WithSpanProcessor(recorder))
	client := anthropicsdk.NewClient(
		option.WithBaseURL(server.URL),
		option.WithAPIKey("test-key"),
		option.WithMiddleware(anthropicotel.Middleware(tp.Tracer("test"))),
	)

	ctx := context.Background()
	ctx = instrumentation.WithSession(ctx, "session-abc")
	ctx = instrumentation.WithUser(ctx, "user-xyz")
	ctx = instrumentation.WithMetadata(ctx, `{"team":"platform"}`)
	ctx = instrumentation.WithTags(ctx, "prod", "canary")

	_, err := client.Messages.New(ctx, anthropicsdk.MessageNewParams{
		Model: "claude-3-5-sonnet-latest", MaxTokens: 10,
		Messages: []anthropicsdk.MessageParam{anthropicsdk.NewUserMessage(anthropicsdk.NewTextBlock("hi"))},
	})
	if err != nil {
		t.Fatalf("Messages.New: %v", err)
	}
	_ = tp.ForceFlush(context.Background())

	spans := recorder.Ended()
	if len(spans) != 1 {
		t.Fatalf("expected 1 span, got %d", len(spans))
	}
	attrs := attrMap(spans[0].Attributes())

	if got := attrs["session.id"]; got != "session-abc" {
		t.Errorf("session.id: got %v", got)
	}
	if got := attrs["user.id"]; got != "user-xyz" {
		t.Errorf("user.id: got %v", got)
	}
	if got := attrs["metadata"]; got != `{"team":"platform"}` {
		t.Errorf("metadata: got %v", got)
	}
	// tag.tags MUST be a typed []string — that's the OpenInference spec
	// shape and what the Python SDK emits. A naive single-string
	// "prod,canary" attribute would NOT match.
	gotTags, ok := attrs["tag.tags"].([]string)
	if !ok {
		t.Fatalf("tag.tags should be []string, got %T = %v", attrs["tag.tags"], attrs["tag.tags"])
	}
	if !reflect.DeepEqual(gotTags, []string{"prod", "canary"}) {
		t.Errorf("tag.tags: got %v want [prod canary]", gotTags)
	}
}

func TestMiddleware_HideInputsRedactsContent(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"id","role":"assistant","model":"m","stop_reason":"end_turn",
			"content":[{"type":"text","text":"hello world"}],
			"usage":{"input_tokens":5,"output_tokens":2,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}
		}`))
	}))
	defer server.Close()

	t.Setenv(instrumentation.EnvHideInputs, "true")

	recorder := tracetest.NewSpanRecorder()
	tp := trace.NewTracerProvider(trace.WithSpanProcessor(recorder))
	client := anthropicsdk.NewClient(
		option.WithBaseURL(server.URL),
		option.WithAPIKey("test-key"),
		option.WithMiddleware(anthropicotel.Middleware(tp.Tracer("test"))),
	)

	_, err := client.Messages.New(context.Background(), anthropicsdk.MessageNewParams{
		Model:     "claude-3-5-sonnet-latest",
		MaxTokens: 10,
		Messages: []anthropicsdk.MessageParam{
			anthropicsdk.NewUserMessage(anthropicsdk.NewTextBlock("the secret password is 12345")),
		},
	})
	if err != nil {
		t.Fatalf("Messages.New: %v", err)
	}
	_ = tp.ForceFlush(context.Background())

	spans := recorder.Ended()
	if len(spans) != 1 {
		t.Fatalf("expected 1 span, got %d", len(spans))
	}
	attrs := attrMap(spans[0].Attributes())

	// HideInputs DROPS the entire llm.input_messages.* structure
	// (matching Python TraceConfig.mask) and replaces input.value
	// with REDACTED.
	if got := attrs[semconv.InputValue]; got != instrumentation.RedactedValue {
		t.Errorf("input.value: got %v want %q", got, instrumentation.RedactedValue)
	}
	if _, present := attrs[semconv.LLMInputMessageRoleKey(0)]; present {
		t.Errorf("HideInputs should drop llm.input_messages.* entirely, found role=%v", attrs[semconv.LLMInputMessageRoleKey(0)])
	}
	if _, present := attrs[semconv.LLMInputMessageContentKey(0)]; present {
		t.Errorf("HideInputs should drop llm.input_messages.* entirely, found content=%v", attrs[semconv.LLMInputMessageContentKey(0)])
	}
	// Output remains visible — only HideInputs was set.
	if got := attrs[semconv.OutputValue]; got != "hello world" {
		t.Errorf("output should not be redacted under HideInputs, got %v", got)
	}
}

func TestMiddleware_HideOutputsRedactsContent(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"id","role":"assistant","model":"m","stop_reason":"end_turn",
			"content":[{"type":"text","text":"the answer is 42"}],
			"usage":{"input_tokens":5,"output_tokens":4,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}
		}`))
	}))
	defer server.Close()

	t.Setenv(instrumentation.EnvHideOutputs, "true")

	recorder := tracetest.NewSpanRecorder()
	tp := trace.NewTracerProvider(trace.WithSpanProcessor(recorder))
	client := anthropicsdk.NewClient(
		option.WithBaseURL(server.URL),
		option.WithAPIKey("test-key"),
		option.WithMiddleware(anthropicotel.Middleware(tp.Tracer("test"))),
	)
	_, err := client.Messages.New(context.Background(), anthropicsdk.MessageNewParams{
		Model:     "claude-3-5-sonnet-latest",
		MaxTokens: 10,
		Messages:  []anthropicsdk.MessageParam{anthropicsdk.NewUserMessage(anthropicsdk.NewTextBlock("hi"))},
	})
	if err != nil {
		t.Fatalf("Messages.New: %v", err)
	}
	_ = tp.ForceFlush(context.Background())

	attrs := attrMap(recorder.Ended()[0].Attributes())
	// HideOutputs DROPS the entire llm.output_messages.* structure
	// and replaces output.value with REDACTED.
	if got := attrs[semconv.OutputValue]; got != instrumentation.RedactedValue {
		t.Errorf("output.value: got %v want REDACTED", got)
	}
	if _, present := attrs[semconv.LLMOutputMessageRoleKey(0)]; present {
		t.Errorf("HideOutputs should drop llm.output_messages.* entirely, found role=%v", attrs[semconv.LLMOutputMessageRoleKey(0)])
	}
	if _, present := attrs[semconv.LLMOutputMessageContentKey(0)]; present {
		t.Errorf("HideOutputs should drop llm.output_messages.* entirely, found content=%v", attrs[semconv.LLMOutputMessageContentKey(0)])
	}
	// Input remains visible.
	if got := attrs[semconv.InputValue]; got != "hi" {
		t.Errorf("input should not be redacted under HideOutputs, got %v", got)
	}
	// Token counts are metadata — must stay visible.
	if got := attrs[semconv.LLMTokenCountPrompt]; got != int64(5) {
		t.Errorf("token counts should not be redacted, got %v", got)
	}
}

func TestMiddleware_WithTraceConfigOverridesEnv(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"id","role":"assistant","model":"m","stop_reason":"end_turn",
			"content":[{"type":"text","text":"ok"}],
			"usage":{"input_tokens":1,"output_tokens":1,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}
		}`))
	}))
	defer server.Close()

	// Env says hide; explicit config says don't. Explicit wins.
	t.Setenv(instrumentation.EnvHideInputs, "true")
	cfg := instrumentation.TraceConfig{} // all false

	recorder := tracetest.NewSpanRecorder()
	tp := trace.NewTracerProvider(trace.WithSpanProcessor(recorder))
	client := anthropicsdk.NewClient(
		option.WithBaseURL(server.URL),
		option.WithAPIKey("test-key"),
		option.WithMiddleware(anthropicotel.Middleware(tp.Tracer("test"), anthropicotel.WithTraceConfig(cfg))),
	)
	_, err := client.Messages.New(context.Background(), anthropicsdk.MessageNewParams{
		Model:     "claude-3-5-sonnet-latest",
		MaxTokens: 10,
		Messages:  []anthropicsdk.MessageParam{anthropicsdk.NewUserMessage(anthropicsdk.NewTextBlock("visible"))},
	})
	if err != nil {
		t.Fatalf("Messages.New: %v", err)
	}
	_ = tp.ForceFlush(context.Background())

	attrs := attrMap(recorder.Ended()[0].Attributes())
	if got := attrs[semconv.InputValue]; got != "visible" {
		t.Errorf("explicit WithTraceConfig should override env: got %v", got)
	}
}

func TestMiddleware_NilBodyRequestFlowsThrough(t *testing.T) {
	// A POST request that arrives at /v1/messages with a nil body is
	// unusual but valid HTTP — instrumentation must NOT block it.
	hit := false
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hit = true
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"id":"id","role":"assistant","model":"m","stop_reason":"end_turn","content":[],"usage":{"input_tokens":0,"output_tokens":0,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}}`))
	}))
	defer server.Close()

	recorder := tracetest.NewSpanRecorder()
	tp := trace.NewTracerProvider(trace.WithSpanProcessor(recorder))
	mw := anthropicotel.Middleware(tp.Tracer("test"))

	req, err := http.NewRequestWithContext(context.Background(), http.MethodPost, server.URL+"/v1/messages", nil)
	if err != nil {
		t.Fatalf("NewRequest: %v", err)
	}
	resp, err := mw(req, http.DefaultTransport.RoundTrip)
	if err != nil {
		t.Fatalf("middleware with nil body should not error: %v", err)
	}
	defer resp.Body.Close()

	if !hit {
		t.Fatal("upstream never received the request — instrumentation aborted a valid nil-body request")
	}
	_ = tp.ForceFlush(context.Background())
	spans := recorder.Ended()
	if len(spans) != 1 {
		t.Fatalf("expected 1 span, got %d", len(spans))
	}
	attrs := attrMap(spans[0].Attributes())
	if got := attrs[semconv.OpenInferenceSpanKind]; got != semconv.SpanKindLLM {
		t.Errorf("span kind: got %v", got)
	}
	if _, present := attrs[semconv.InputValue]; present {
		t.Errorf("nil-body request should not produce input.value, got %v", attrs[semconv.InputValue])
	}
}

func TestMiddleware_CachedTokensFoldIntoPromptCount(t *testing.T) {
	// Anthropic's input_tokens excludes the cached portions. The
	// instrumentor must fold cache_read + cache_creation back into
	// llm.token_count.prompt (and therefore total) so the totals stay
	// consistent with the prompt-detail breakdown and match the
	// Python instrumentor.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"id","role":"assistant","model":"m","stop_reason":"end_turn",
			"content":[{"type":"text","text":"hi"}],
			"usage":{
				"input_tokens": 10,
				"output_tokens": 5,
				"cache_creation_input_tokens": 7,
				"cache_read_input_tokens": 3
			}
		}`))
	}))
	defer server.Close()

	recorder := tracetest.NewSpanRecorder()
	tp := trace.NewTracerProvider(trace.WithSpanProcessor(recorder))

	client := anthropicsdk.NewClient(
		option.WithBaseURL(server.URL),
		option.WithAPIKey("test-key"),
		option.WithMiddleware(anthropicotel.Middleware(tp.Tracer("test"))),
	)
	_, _ = client.Messages.New(context.Background(), anthropicsdk.MessageNewParams{
		Model:     "claude-3-5-sonnet-latest",
		MaxTokens: 10,
		Messages: []anthropicsdk.MessageParam{
			anthropicsdk.NewUserMessage(anthropicsdk.NewTextBlock("hi")),
		},
	})

	_ = tp.ForceFlush(context.Background())
	attrs := attrMap(recorder.Ended()[0].Attributes())

	// prompt = input_tokens + cache_read + cache_creation = 10 + 3 + 7 = 20
	if got, want := attrs[semconv.LLMTokenCountPrompt], int64(20); got != want {
		t.Errorf("llm.token_count.prompt: got %v want %v (input_tokens=10 + cache_read=3 + cache_creation=7)", got, want)
	}
	if got, want := attrs[semconv.LLMTokenCountCompletion], int64(5); got != want {
		t.Errorf("llm.token_count.completion: got %v want %v", got, want)
	}
	// total must equal prompt + completion using the cache-inclusive
	// prompt — not the bare input_tokens (which would give 15).
	if got, want := attrs[semconv.LLMTokenCountTotal], int64(25); got != want {
		t.Errorf("llm.token_count.total: got %v want %v (prompt=20 + completion=5)", got, want)
	}
	if got, want := attrs[semconv.LLMTokenCountPromptDetailsCacheRead], int64(3); got != want {
		t.Errorf("cache_read detail: got %v want %v", got, want)
	}
	if got, want := attrs[semconv.LLMTokenCountPromptDetailsCacheWrite], int64(7); got != want {
		t.Errorf("cache_write detail: got %v want %v", got, want)
	}
}

// toolFixtureParams returns a MessageNewParams that advertises a single
// "get_weather" tool. Shared by the tool-emission and hide-tools tests
// so they exercise the same wire shape.
func toolFixtureParams() anthropicsdk.MessageNewParams {
	return anthropicsdk.MessageNewParams{
		Model:     "claude-3-5-sonnet-latest",
		MaxTokens: 10,
		Messages: []anthropicsdk.MessageParam{
			anthropicsdk.NewUserMessage(anthropicsdk.NewTextBlock("What's the weather?")),
		},
		Tools: []anthropicsdk.ToolUnionParam{
			{OfTool: &anthropicsdk.ToolParam{
				Name: "get_weather",
				InputSchema: anthropicsdk.ToolInputSchemaParam{
					Properties: map[string]any{
						"location": map[string]any{"type": "string", "description": "city, state"},
					},
					Required: []string{"location"},
				},
			}},
		},
	}
}

func TestMiddleware_AdvertisedToolsEmittedAsAttributes(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"id","role":"assistant","model":"m","stop_reason":"end_turn",
			"content":[{"type":"text","text":"ok"}],
			"usage":{"input_tokens":1,"output_tokens":1,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}
		}`))
	}))
	defer server.Close()

	recorder := tracetest.NewSpanRecorder()
	tp := trace.NewTracerProvider(trace.WithSpanProcessor(recorder))

	client := anthropicsdk.NewClient(
		option.WithBaseURL(server.URL),
		option.WithAPIKey("test-key"),
		option.WithMiddleware(anthropicotel.Middleware(tp.Tracer("test"))),
	)
	_, _ = client.Messages.New(context.Background(), toolFixtureParams())

	_ = tp.ForceFlush(context.Background())
	attrs := attrMap(recorder.Ended()[0].Attributes())

	raw, ok := attrs[semconv.LLMToolKey(0)].(string)
	if !ok {
		t.Fatalf("llm.tools.0.tool.json_schema missing or wrong type: %T = %v", attrs[semconv.LLMToolKey(0)], attrs[semconv.LLMToolKey(0)])
	}
	// The attribute value is a JSON-encoded copy of the tool object on
	// the wire. Decode and assert on the meaningful fields rather than
	// the exact byte sequence (SDK field order is not guaranteed).
	var decoded map[string]any
	if err := json.Unmarshal([]byte(raw), &decoded); err != nil {
		t.Fatalf("llm.tools.0.tool.json_schema not valid JSON: %v\nraw=%s", err, raw)
	}
	if decoded["name"] != "get_weather" {
		t.Errorf("tool name: got %v want get_weather", decoded["name"])
	}
	if _, present := decoded["input_schema"]; !present {
		t.Errorf("tool schema missing input_schema field; got %v", decoded)
	}
}

func TestMiddleware_HideLLMToolsOmitsAdvertisedTools(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"id","role":"assistant","model":"m","stop_reason":"end_turn",
			"content":[{"type":"text","text":"ok"}],
			"usage":{"input_tokens":1,"output_tokens":1,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}
		}`))
	}))
	defer server.Close()

	recorder := tracetest.NewSpanRecorder()
	tp := trace.NewTracerProvider(trace.WithSpanProcessor(recorder))

	client := anthropicsdk.NewClient(
		option.WithBaseURL(server.URL),
		option.WithAPIKey("test-key"),
		option.WithMiddleware(anthropicotel.Middleware(
			tp.Tracer("test"),
			anthropicotel.WithTraceConfig(instrumentation.TraceConfig{HideLLMTools: true}),
		)),
	)
	_, _ = client.Messages.New(context.Background(), toolFixtureParams())

	_ = tp.ForceFlush(context.Background())
	attrs := attrMap(recorder.Ended()[0].Attributes())

	if v, present := attrs[semconv.LLMToolKey(0)]; present {
		t.Errorf("HideLLMTools should drop llm.tools.0.tool.json_schema; got %v", v)
	}
	// Input messages must still be present — HideLLMTools is targeted,
	// not the same broad opt-out as HideInputs.
	if _, present := attrs[semconv.LLMInputMessageRoleKey(0)]; !present {
		t.Errorf("HideLLMTools must not drop llm.input_messages.* — got missing role[0]")
	}
}

func TestMiddleware_NonMessageRequestUntouched(t *testing.T) {
	hit := false
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hit = true
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"data":[]}`))
	}))
	defer server.Close()

	recorder := tracetest.NewSpanRecorder()
	tp := trace.NewTracerProvider(trace.WithSpanProcessor(recorder))

	client := anthropicsdk.NewClient(
		option.WithBaseURL(server.URL),
		option.WithAPIKey("test-key"),
		option.WithMiddleware(anthropicotel.Middleware(tp.Tracer("test"))),
	)

	_, err := client.Models.List(context.Background(), anthropicsdk.ModelListParams{})
	// Models.List may parse the empty fake response into an error; we
	// don't care about success, just that no span got emitted.
	_ = err
	if !hit {
		t.Fatal("fake server never hit")
	}

	_ = tp.ForceFlush(context.Background())
	if got := len(recorder.Ended()); got != 0 {
		t.Errorf("non-message requests should not produce spans, got %d", got)
	}
}

func attrMap(attrs []attribute.KeyValue) map[string]any {
	out := make(map[string]any, len(attrs))
	for _, kv := range attrs {
		out[string(kv.Key)] = kv.Value.AsInterface()
	}
	// Decode raw JSON-string attributes that callers care about. The OTel
	// attribute API stores everything via AsInterface; for invocation
	// params we want the raw string back as-is.
	if raw, ok := out[semconv.LLMInvocationParameters].(string); ok {
		var v any
		if err := json.Unmarshal([]byte(raw), &v); err == nil {
			out[semconv.LLMInvocationParameters] = raw // keep string form for substring assertions
		}
	}
	return out
}

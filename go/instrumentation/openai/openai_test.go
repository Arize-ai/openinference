package openai_test

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	openaisdk "github.com/sashabaranov/go-openai"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"

	"github.com/Arize-ai/openinference/go/instrumentation"
	openaiotel "github.com/Arize-ai/openinference/go/instrumentation/openai"
	"github.com/Arize-ai/openinference/go/semconv"
)

func newClient(t *testing.T, baseURL string, tracer *trace.TracerProvider) *openaisdk.Client {
	t.Helper()
	cfg := openaisdk.DefaultConfig("test-key")
	cfg.BaseURL = baseURL
	cfg.HTTPClient = &http.Client{
		Transport: openaiotel.NewTransport(http.DefaultTransport, tracer.Tracer("test")),
	}
	return openaisdk.NewClientWithConfig(cfg)
}

func TestTransport_HappyPath(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"chatcmpl-abc",
			"model":"gpt-4o-2024-08-06",
			"choices":[{
				"index":0,
				"finish_reason":"stop",
				"message":{"role":"assistant","content":"4"}
			}],
			"usage":{
				"prompt_tokens":12,
				"completion_tokens":1,
				"total_tokens":13,
				"prompt_tokens_details":{"cached_tokens":0,"audio_tokens":0}
			}
		}`))
	}))
	defer server.Close()

	recorder := tracetest.NewSpanRecorder()
	tp := trace.NewTracerProvider(trace.WithSpanProcessor(recorder))
	client := newClient(t, server.URL, tp)

	resp, err := client.CreateChatCompletion(context.Background(), openaisdk.ChatCompletionRequest{
		Model:     openaisdk.GPT4o,
		MaxTokens: 100,
		Messages: []openaisdk.ChatCompletionMessage{
			{Role: openaisdk.ChatMessageRoleSystem, Content: "You are a helpful assistant."},
			{Role: openaisdk.ChatMessageRoleUser, Content: "What is 2+2?"},
		},
	})
	if err != nil {
		t.Fatalf("CreateChatCompletion: %v", err)
	}
	if resp.Choices[0].Message.Content != "4" {
		t.Fatalf("unexpected content: %q", resp.Choices[0].Message.Content)
	}

	if err := tp.ForceFlush(context.Background()); err != nil {
		t.Fatalf("ForceFlush: %v", err)
	}

	spans := recorder.Ended()
	if len(spans) != 1 {
		t.Fatalf("expected 1 span, got %d", len(spans))
	}
	attrs := attrMap(spans[0].Attributes())

	want := map[string]any{
		semconv.OpenInferenceSpanKind:         semconv.SpanKindLLM,
		semconv.LLMSystem:                     semconv.LLMSystemOpenAI,
		semconv.LLMProvider:                   semconv.LLMProviderOpenAI,
		semconv.LLMModelName:                  "gpt-4o-2024-08-06", // response wins
		semconv.LLMInputMessageRoleKey(0):     "system",
		semconv.LLMInputMessageContentKey(0):  "You are a helpful assistant.",
		semconv.LLMInputMessageRoleKey(1):     "user",
		semconv.LLMInputMessageContentKey(1):  "What is 2+2?",
		semconv.InputValue:                    "What is 2+2?",
		semconv.LLMFinishReason:               "stop",
		semconv.LLMOutputMessageRoleKey(0):    "assistant",
		semconv.LLMOutputMessageContentKey(0): "4",
		semconv.OutputValue:                   "4",
		semconv.LLMTokenCountPrompt:           int64(12),
		semconv.LLMTokenCountCompletion:       int64(1),
		semconv.LLMTokenCountTotal:            int64(13),
	}
	for k, v := range want {
		if got := attrs[k]; got != v {
			t.Errorf("attr %s: got %v want %v", k, got, v)
		}
	}

	invocation, _ := attrs[semconv.LLMInvocationParameters].(string)
	if !strings.Contains(invocation, `"max_tokens":100`) {
		t.Errorf("invocation params missing max_tokens: %q", invocation)
	}
}

func TestTransport_ToolCallsCaptured(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"id","model":"gpt-4o",
			"choices":[{
				"index":0,"finish_reason":"tool_calls",
				"message":{
					"role":"assistant","content":"",
					"tool_calls":[{
						"id":"tc_1","type":"function",
						"function":{"name":"get_weather","arguments":"{\"city\":\"sf\"}"}
					}]
				}
			}],
			"usage":{"prompt_tokens":5,"completion_tokens":7,"total_tokens":12}
		}`))
	}))
	defer server.Close()

	recorder := tracetest.NewSpanRecorder()
	tp := trace.NewTracerProvider(trace.WithSpanProcessor(recorder))
	client := newClient(t, server.URL, tp)

	resp, err := client.CreateChatCompletion(context.Background(), openaisdk.ChatCompletionRequest{
		Model: openaisdk.GPT4o,
		Messages: []openaisdk.ChatCompletionMessage{
			{Role: openaisdk.ChatMessageRoleUser, Content: "weather in sf?"},
		},
		Tools: []openaisdk.Tool{
			{
				Type: openaisdk.ToolTypeFunction,
				Function: &openaisdk.FunctionDefinition{
					Name:        "get_weather",
					Description: "Get the current weather",
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("CreateChatCompletion: %v", err)
	}
	if len(resp.Choices[0].Message.ToolCalls) != 1 {
		t.Fatal("expected tool call in response")
	}

	_ = tp.ForceFlush(context.Background())
	spans := recorder.Ended()
	if len(spans) != 1 {
		t.Fatalf("expected 1 span, got %d", len(spans))
	}
	attrs := attrMap(spans[0].Attributes())

	if got := attrs[semconv.LLMOutputMessageToolCallKey(0, 0, semconv.ToolCallFunctionName)]; got != "get_weather" {
		t.Errorf("tool call function name: got %v", got)
	}
	if got := attrs[semconv.LLMOutputMessageToolCallKey(0, 0, semconv.ToolCallFunctionArgumentsJSON)]; got != `{"city":"sf"}` {
		t.Errorf("tool call args: got %v", got)
	}
	if got := attrs[semconv.LLMOutputMessageToolCallKey(0, 0, semconv.ToolCallID)]; got != "tc_1" {
		t.Errorf("tool call id: got %v", got)
	}
	if got := attrs[semconv.LLMToolKey(0)]; got == nil {
		t.Error("expected tool advertisement to be captured")
	}
}

func TestTransport_ErrorResponseDoesNotPolluteTokenCounts(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":{"message":"rate limited","type":"rate_limit_error","code":"rate_limit_exceeded"}}`))
	}))
	defer server.Close()

	recorder := tracetest.NewSpanRecorder()
	tp := trace.NewTracerProvider(trace.WithSpanProcessor(recorder))
	client := newClient(t, server.URL, tp)

	_, _ = client.CreateChatCompletion(context.Background(), openaisdk.ChatCompletionRequest{
		Model: openaisdk.GPT4o,
		Messages: []openaisdk.ChatCompletionMessage{
			{Role: openaisdk.ChatMessageRoleUser, Content: "hi"},
		},
	})
	_ = tp.ForceFlush(context.Background())

	spans := recorder.Ended()
	if len(spans) != 1 {
		t.Fatalf("expected 1 span, got %d", len(spans))
	}
	attrs := attrMap(spans[0].Attributes())

	for _, k := range []string{
		semconv.LLMTokenCountPrompt,
		semconv.LLMTokenCountCompletion,
		semconv.LLMTokenCountTotal,
	} {
		if _, present := attrs[k]; present {
			t.Errorf("error response should not produce %s, got %v", k, attrs[k])
		}
	}
	if _, present := attrs[semconv.OutputValue]; present {
		t.Error("error response should not set output.value")
	}
	if status := spans[0].Status(); status.Code.String() != "Error" {
		t.Errorf("expected span status Error, got %s", status.Code)
	}
}

func TestTransport_StreamingSpanEndsOnBodyClose(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n"))
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
	}))
	defer server.Close()

	recorder := tracetest.NewSpanRecorder()
	tp := trace.NewTracerProvider(trace.WithSpanProcessor(recorder))
	rt := openaiotel.NewTransport(http.DefaultTransport, tp.Tracer("test"))

	reqBody := strings.NewReader(`{"model":"gpt-4o","stream":true,"messages":[{"role":"user","content":"hi"}]}`)
	req, err := http.NewRequestWithContext(context.Background(), http.MethodPost, server.URL+"/v1/chat/completions", reqBody)
	if err != nil {
		t.Fatalf("NewRequest: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := rt.RoundTrip(req)
	if err != nil {
		t.Fatalf("RoundTrip: %v", err)
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

	attrs := attrMap(recorder.Ended()[0].Attributes())
	if attrs[semconv.OpenInferenceSpanKind] != semconv.SpanKindLLM {
		t.Errorf("span kind: got %v", attrs[semconv.OpenInferenceSpanKind])
	}
	if _, present := attrs[semconv.OutputValue]; present {
		t.Errorf("streaming span should not set output.value")
	}
}

func TestTransport_ResponseParseFailureRecordsError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{this is not valid json}`))
	}))
	defer server.Close()

	recorder := tracetest.NewSpanRecorder()
	tp := trace.NewTracerProvider(trace.WithSpanProcessor(recorder))
	client := newClient(t, server.URL, tp)

	_, _ = client.CreateChatCompletion(context.Background(), openaisdk.ChatCompletionRequest{
		Model: openaisdk.GPT4o,
		Messages: []openaisdk.ChatCompletionMessage{
			{Role: openaisdk.ChatMessageRoleUser, Content: "hi"},
		},
	})
	_ = tp.ForceFlush(context.Background())

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

func TestTransport_SuppressedContextEmitsNoSpan(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"id","model":"gpt-4o","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"ok"}}],
			"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
		}`))
	}))
	defer server.Close()

	recorder := tracetest.NewSpanRecorder()
	tp := trace.NewTracerProvider(trace.WithSpanProcessor(recorder))
	client := newClient(t, server.URL, tp)

	ctx := instrumentation.WithSuppression(context.Background())
	_, err := client.CreateChatCompletion(ctx, openaisdk.ChatCompletionRequest{
		Model:    openaisdk.GPT4o,
		Messages: []openaisdk.ChatCompletionMessage{{Role: openaisdk.ChatMessageRoleUser, Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("CreateChatCompletion: %v", err)
	}
	_ = tp.ForceFlush(context.Background())

	if got := len(recorder.Ended()); got != 0 {
		t.Errorf("suppressed context should produce no span, got %d", got)
	}
}

func TestTransport_BaggagePropagatesToSpan(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"id","model":"gpt-4o","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"ok"}}],
			"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
		}`))
	}))
	defer server.Close()

	recorder := tracetest.NewSpanRecorder()
	tp := trace.NewTracerProvider(trace.WithSpanProcessor(recorder))
	client := newClient(t, server.URL, tp)

	ctx := context.Background()
	ctx = instrumentation.WithSession(ctx, "session-abc")
	ctx = instrumentation.WithUser(ctx, "user-xyz")
	ctx = instrumentation.WithMetadata(ctx, `{"team":"platform"}`)
	ctx = instrumentation.WithTags(ctx, "prod", "canary")

	_, err := client.CreateChatCompletion(ctx, openaisdk.ChatCompletionRequest{
		Model:    openaisdk.GPT4o,
		Messages: []openaisdk.ChatCompletionMessage{{Role: openaisdk.ChatMessageRoleUser, Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("CreateChatCompletion: %v", err)
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
	gotTags, ok := attrs["tag.tags"].([]string)
	if !ok {
		t.Fatalf("tag.tags should be []string, got %T = %v", attrs["tag.tags"], attrs["tag.tags"])
	}
	if !reflect.DeepEqual(gotTags, []string{"prod", "canary"}) {
		t.Errorf("tag.tags: got %v want [prod canary]", gotTags)
	}
}

func TestTransport_HideInputsRedactsContent(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"id","model":"gpt-4o","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"hello world"}}],
			"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}
		}`))
	}))
	defer server.Close()

	t.Setenv(instrumentation.EnvHideInputs, "true")

	recorder := tracetest.NewSpanRecorder()
	tp := trace.NewTracerProvider(trace.WithSpanProcessor(recorder))
	client := newClient(t, server.URL, tp)

	_, err := client.CreateChatCompletion(context.Background(), openaisdk.ChatCompletionRequest{
		Model: openaisdk.GPT4o,
		Messages: []openaisdk.ChatCompletionMessage{
			{Role: openaisdk.ChatMessageRoleUser, Content: "the secret password is 12345"},
		},
	})
	if err != nil {
		t.Fatalf("CreateChatCompletion: %v", err)
	}
	_ = tp.ForceFlush(context.Background())

	attrs := attrMap(recorder.Ended()[0].Attributes())
	// HideInputs DROPS the entire llm.input_messages.* structure
	// (matching Python TraceConfig.mask) and replaces input.value
	// with REDACTED.
	if got := attrs[semconv.InputValue]; got != instrumentation.RedactedValue {
		t.Errorf("input.value: got %v want REDACTED", got)
	}
	if _, present := attrs[semconv.LLMInputMessageRoleKey(0)]; present {
		t.Errorf("HideInputs should drop llm.input_messages.* entirely, found role=%v", attrs[semconv.LLMInputMessageRoleKey(0)])
	}
	if _, present := attrs[semconv.LLMInputMessageContentKey(0)]; present {
		t.Errorf("HideInputs should drop llm.input_messages.* entirely, found content=%v", attrs[semconv.LLMInputMessageContentKey(0)])
	}
	if got := attrs[semconv.OutputValue]; got != "hello world" {
		t.Errorf("output should not be redacted under HideInputs, got %v", got)
	}
}

func TestTransport_HideInputsAlsoDropsAdvertisedTools(t *testing.T) {
	// HideInputs implies HideLLMTools per Python's TraceConfig.mask:
	// the tool list is considered part of the input.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"id","model":"gpt-4o","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"ok"}}],
			"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
		}`))
	}))
	defer server.Close()

	t.Setenv(instrumentation.EnvHideInputs, "true")

	recorder := tracetest.NewSpanRecorder()
	tp := trace.NewTracerProvider(trace.WithSpanProcessor(recorder))
	client := newClient(t, server.URL, tp)

	_, err := client.CreateChatCompletion(context.Background(), openaisdk.ChatCompletionRequest{
		Model:    openaisdk.GPT4o,
		Messages: []openaisdk.ChatCompletionMessage{{Role: openaisdk.ChatMessageRoleUser, Content: "weather?"}},
		Tools: []openaisdk.Tool{{
			Type:     openaisdk.ToolTypeFunction,
			Function: &openaisdk.FunctionDefinition{Name: "get_weather"},
		}},
	})
	if err != nil {
		t.Fatalf("CreateChatCompletion: %v", err)
	}
	_ = tp.ForceFlush(context.Background())

	attrs := attrMap(recorder.Ended()[0].Attributes())
	if _, present := attrs[semconv.LLMToolKey(0)]; present {
		t.Errorf("HideInputs should drop llm.tools.* (matches Python semantics), found %v", attrs[semconv.LLMToolKey(0)])
	}
}

func TestTransport_HideOutputsRedactsContent(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"id","model":"gpt-4o","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"the answer is 42"}}],
			"usage":{"prompt_tokens":5,"completion_tokens":4,"total_tokens":9}
		}`))
	}))
	defer server.Close()

	t.Setenv(instrumentation.EnvHideOutputs, "true")

	recorder := tracetest.NewSpanRecorder()
	tp := trace.NewTracerProvider(trace.WithSpanProcessor(recorder))
	client := newClient(t, server.URL, tp)

	_, err := client.CreateChatCompletion(context.Background(), openaisdk.ChatCompletionRequest{
		Model:    openaisdk.GPT4o,
		Messages: []openaisdk.ChatCompletionMessage{{Role: openaisdk.ChatMessageRoleUser, Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("CreateChatCompletion: %v", err)
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
	if got := attrs[semconv.InputValue]; got != "hi" {
		t.Errorf("input should not be redacted under HideOutputs, got %v", got)
	}
	// finish_reason is metadata, not content — must remain visible.
	if got := attrs[semconv.LLMFinishReason]; got != "stop" {
		t.Errorf("finish_reason should not be hidden, got %v", got)
	}
	if got := attrs[semconv.LLMTokenCountPrompt]; got != int64(5) {
		t.Errorf("token counts should not be hidden, got %v", got)
	}
}

func TestTransport_WithTraceConfigOverridesEnv(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"id","model":"gpt-4o","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"ok"}}],
			"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
		}`))
	}))
	defer server.Close()

	t.Setenv(instrumentation.EnvHideInputs, "true")
	cfg := instrumentation.TraceConfig{} // all false — should override env

	recorder := tracetest.NewSpanRecorder()
	tp := trace.NewTracerProvider(trace.WithSpanProcessor(recorder))
	cfgClient := openaisdk.DefaultConfig("test-key")
	cfgClient.BaseURL = server.URL
	cfgClient.HTTPClient = &http.Client{
		Transport: openaiotel.NewTransport(http.DefaultTransport, tp.Tracer("test"), openaiotel.WithTraceConfig(cfg)),
	}
	client := openaisdk.NewClientWithConfig(cfgClient)

	_, err := client.CreateChatCompletion(context.Background(), openaisdk.ChatCompletionRequest{
		Model:    openaisdk.GPT4o,
		Messages: []openaisdk.ChatCompletionMessage{{Role: openaisdk.ChatMessageRoleUser, Content: "visible"}},
	})
	if err != nil {
		t.Fatalf("CreateChatCompletion: %v", err)
	}
	_ = tp.ForceFlush(context.Background())

	attrs := attrMap(recorder.Ended()[0].Attributes())
	if got := attrs[semconv.InputValue]; got != "visible" {
		t.Errorf("explicit WithTraceConfig should override env: got %v", got)
	}
}

func TestTransport_HideLLMToolsOmitsAdvertisedTools(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"id","model":"gpt-4o","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"ok"}}],
			"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
		}`))
	}))
	defer server.Close()

	t.Setenv(instrumentation.EnvHideLLMTools, "true")

	recorder := tracetest.NewSpanRecorder()
	tp := trace.NewTracerProvider(trace.WithSpanProcessor(recorder))
	client := newClient(t, server.URL, tp)

	_, err := client.CreateChatCompletion(context.Background(), openaisdk.ChatCompletionRequest{
		Model:    openaisdk.GPT4o,
		Messages: []openaisdk.ChatCompletionMessage{{Role: openaisdk.ChatMessageRoleUser, Content: "weather?"}},
		Tools: []openaisdk.Tool{{
			Type:     openaisdk.ToolTypeFunction,
			Function: &openaisdk.FunctionDefinition{Name: "get_weather"},
		}},
	})
	if err != nil {
		t.Fatalf("CreateChatCompletion: %v", err)
	}
	_ = tp.ForceFlush(context.Background())

	attrs := attrMap(recorder.Ended()[0].Attributes())
	if got, present := attrs[semconv.LLMToolKey(0)]; present {
		t.Errorf("llm.tools should be hidden under HideLLMTools, got %v", got)
	}
}

func TestTransport_NilBodyRequestFlowsThrough(t *testing.T) {
	// A POST request that arrives at /v1/chat/completions with a nil
	// body is unusual but valid HTTP — instrumentation must NOT block
	// it. The middleware should forward to the upstream and emit a
	// span carrying only the const LLM attributes.
	hit := false
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hit = true
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"id":"id","model":"m","choices":[],"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}`))
	}))
	defer server.Close()

	recorder := tracetest.NewSpanRecorder()
	tp := trace.NewTracerProvider(trace.WithSpanProcessor(recorder))
	rt := openaiotel.NewTransport(http.DefaultTransport, tp.Tracer("test"))

	req, err := http.NewRequestWithContext(context.Background(), http.MethodPost, server.URL+"/v1/chat/completions", nil)
	if err != nil {
		t.Fatalf("NewRequest: %v", err)
	}
	resp, err := rt.RoundTrip(req)
	if err != nil {
		t.Fatalf("RoundTrip with nil body should not error: %v", err)
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
	// Span should have the const LLM attributes but no input.value.
	attrs := attrMap(spans[0].Attributes())
	if got := attrs[semconv.OpenInferenceSpanKind]; got != semconv.SpanKindLLM {
		t.Errorf("span kind: got %v", got)
	}
	if _, present := attrs[semconv.InputValue]; present {
		t.Errorf("nil-body request should not produce input.value, got %v", attrs[semconv.InputValue])
	}
}

func TestTransport_NonChatRequestUntouched(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"data":[],"object":"list"}`))
	}))
	defer server.Close()

	recorder := tracetest.NewSpanRecorder()
	tp := trace.NewTracerProvider(trace.WithSpanProcessor(recorder))
	client := newClient(t, server.URL, tp)

	_, _ = client.ListModels(context.Background())

	_ = tp.ForceFlush(context.Background())
	if got := len(recorder.Ended()); got != 0 {
		t.Errorf("non-chat-completion requests should not produce spans, got %d", got)
	}
}

func attrMap(attrs []attribute.KeyValue) map[string]any {
	out := make(map[string]any, len(attrs))
	for _, kv := range attrs {
		out[string(kv.Key)] = kv.Value.AsInterface()
	}
	return out
}

package openai_test

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	openaisdk "github.com/sashabaranov/go-openai"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"

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

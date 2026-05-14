package openai_test

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	openaisdk "github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/shared"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"

	"github.com/Arize-ai/openinference/go/openinference-instrumentation"
	openaiotel "github.com/Arize-ai/openinference/go/openinference-instrumentation-openai-go"
	"github.com/Arize-ai/openinference/go/openinference-semantic-conventions"
)

// newClient returns an openai-go Client whose HTTP path is wrapped by
// the OpenInference middleware and pointed at baseURL (typically a
// httptest server).
func newClient(t *testing.T, baseURL string, tp *trace.TracerProvider, opts ...openaiotel.Option) openaisdk.Client {
	t.Helper()
	return openaisdk.NewClient(
		option.WithBaseURL(baseURL),
		option.WithAPIKey("test-key"),
		option.WithMiddleware(openaiotel.Middleware(tp.Tracer("test"), opts...)),
	)
}

func TestMiddleware_HappyPath(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"chatcmpl-abc",
			"object":"chat.completion",
			"created":0,
			"model":"gpt-4o-2024-08-06",
			"choices":[{
				"index":0,
				"finish_reason":"stop",
				"message":{"role":"assistant","content":"4"},
				"logprobs":null
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

	resp, err := client.Chat.Completions.New(context.Background(), openaisdk.ChatCompletionNewParams{
		Model:     shared.ChatModelGPT4o,
		MaxTokens: openaisdk.Int(100),
		Messages: []openaisdk.ChatCompletionMessageParamUnion{
			openaisdk.SystemMessage("You are a helpful assistant."),
			openaisdk.UserMessage("What is 2+2?"),
		},
	})
	if err != nil {
		t.Fatalf("Chat.Completions.New: %v", err)
	}
	if resp.Choices[0].Message.Content != "4" {
		t.Fatalf("unexpected content: %q", resp.Choices[0].Message.Content)
	}

	_ = tp.ForceFlush(context.Background())
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

func TestMiddleware_ToolCallsCaptured(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"id","object":"chat.completion","created":0,"model":"gpt-4o",
			"choices":[{
				"index":0,"finish_reason":"tool_calls","logprobs":null,
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

	resp, err := client.Chat.Completions.New(context.Background(), openaisdk.ChatCompletionNewParams{
		Model: shared.ChatModelGPT4o,
		Messages: []openaisdk.ChatCompletionMessageParamUnion{
			openaisdk.UserMessage("weather in sf?"),
		},
		Tools: []openaisdk.ChatCompletionToolParam{
			{
				Function: shared.FunctionDefinitionParam{
					Name:        "get_weather",
					Description: openaisdk.String("Get the current weather"),
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("Chat.Completions.New: %v", err)
	}
	if len(resp.Choices[0].Message.ToolCalls) != 1 {
		t.Fatal("expected tool call in response")
	}

	_ = tp.ForceFlush(context.Background())
	attrs := attrMap(recorder.Ended()[0].Attributes())

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
		t.Error("expected advertised tool to be captured as llm.tools.0.tool.json_schema")
	}
}

func TestMiddleware_ErrorResponseDoesNotPolluteTokenCounts(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":{"message":"rate limited","type":"rate_limit_error","code":"rate_limit_exceeded"}}`))
	}))
	defer server.Close()

	recorder := tracetest.NewSpanRecorder()
	tp := trace.NewTracerProvider(trace.WithSpanProcessor(recorder))
	// Disable retries so we see exactly one span (the SDK retries 429
	// by default and would emit multiple spans).
	client := openaisdk.NewClient(
		option.WithBaseURL(server.URL),
		option.WithAPIKey("test-key"),
		option.WithMaxRetries(0),
		option.WithMiddleware(openaiotel.Middleware(tp.Tracer("test"))),
	)

	_, _ = client.Chat.Completions.New(context.Background(), openaisdk.ChatCompletionNewParams{
		Model:    shared.ChatModelGPT4o,
		Messages: []openaisdk.ChatCompletionMessageParamUnion{openaisdk.UserMessage("hi")},
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

func TestMiddleware_StreamingSpanEndsOnBodyClose(t *testing.T) {
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
	mw := openaiotel.Middleware(tp.Tracer("test"))

	// Drive the middleware directly with a synthesised request — the
	// SDK's stream consumer would otherwise drain the body before we
	// can assert on span lifetime.
	body := strings.NewReader(`{"model":"gpt-4o","stream":true,"messages":[{"role":"user","content":"hi"}]}`)
	req, err := http.NewRequestWithContext(context.Background(), http.MethodPost, server.URL+"/v1/chat/completions", body)
	if err != nil {
		t.Fatalf("NewRequest: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := mw(req, http.DefaultClient.Do)
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

	attrs := attrMap(recorder.Ended()[0].Attributes())
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
	client := newClient(t, server.URL, tp)

	_, _ = client.Chat.Completions.New(context.Background(), openaisdk.ChatCompletionNewParams{
		Model:    shared.ChatModelGPT4o,
		Messages: []openaisdk.ChatCompletionMessageParamUnion{openaisdk.UserMessage("hi")},
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

func TestMiddleware_SuppressedContextEmitsNoSpan(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(okResponse))
	}))
	defer server.Close()

	recorder := tracetest.NewSpanRecorder()
	tp := trace.NewTracerProvider(trace.WithSpanProcessor(recorder))
	client := newClient(t, server.URL, tp)

	ctx := instrumentation.WithSuppression(context.Background())
	_, err := client.Chat.Completions.New(ctx, openaisdk.ChatCompletionNewParams{
		Model:    shared.ChatModelGPT4o,
		Messages: []openaisdk.ChatCompletionMessageParamUnion{openaisdk.UserMessage("hi")},
	})
	if err != nil {
		t.Fatalf("Chat.Completions.New: %v", err)
	}
	_ = tp.ForceFlush(context.Background())

	if got := len(recorder.Ended()); got != 0 {
		t.Errorf("suppressed context should produce no span, got %d", got)
	}
}

func TestMiddleware_ContextAttributesPropagateToSpan(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(okResponse))
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

	_, err := client.Chat.Completions.New(ctx, openaisdk.ChatCompletionNewParams{
		Model:    shared.ChatModelGPT4o,
		Messages: []openaisdk.ChatCompletionMessageParamUnion{openaisdk.UserMessage("hi")},
	})
	if err != nil {
		t.Fatalf("Chat.Completions.New: %v", err)
	}
	_ = tp.ForceFlush(context.Background())

	attrs := attrMap(recorder.Ended()[0].Attributes())
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

func TestMiddleware_HideInputsRedactsContent(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"id","object":"chat.completion","created":0,"model":"gpt-4o",
			"choices":[{"index":0,"finish_reason":"stop","logprobs":null,"message":{"role":"assistant","content":"hello world"}}],
			"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}
		}`))
	}))
	defer server.Close()

	t.Setenv(instrumentation.EnvHideInputs, "true")

	recorder := tracetest.NewSpanRecorder()
	tp := trace.NewTracerProvider(trace.WithSpanProcessor(recorder))
	client := newClient(t, server.URL, tp)

	_, err := client.Chat.Completions.New(context.Background(), openaisdk.ChatCompletionNewParams{
		Model:    shared.ChatModelGPT4o,
		Messages: []openaisdk.ChatCompletionMessageParamUnion{openaisdk.UserMessage("the secret password is 12345")},
	})
	if err != nil {
		t.Fatalf("Chat.Completions.New: %v", err)
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

func TestMiddleware_HideInputsAlsoDropsAdvertisedTools(t *testing.T) {
	// HideInputs implies HideLLMTools per Python's TraceConfig.mask:
	// the tool list is considered part of the input.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(okResponse))
	}))
	defer server.Close()

	t.Setenv(instrumentation.EnvHideInputs, "true")

	recorder := tracetest.NewSpanRecorder()
	tp := trace.NewTracerProvider(trace.WithSpanProcessor(recorder))
	client := newClient(t, server.URL, tp)

	_, err := client.Chat.Completions.New(context.Background(), openaisdk.ChatCompletionNewParams{
		Model:    shared.ChatModelGPT4o,
		Messages: []openaisdk.ChatCompletionMessageParamUnion{openaisdk.UserMessage("weather?")},
		Tools: []openaisdk.ChatCompletionToolParam{
			{Function: shared.FunctionDefinitionParam{Name: "get_weather"}},
		},
	})
	if err != nil {
		t.Fatalf("Chat.Completions.New: %v", err)
	}
	_ = tp.ForceFlush(context.Background())

	attrs := attrMap(recorder.Ended()[0].Attributes())
	if _, present := attrs[semconv.LLMToolKey(0)]; present {
		t.Errorf("HideInputs should drop llm.tools.* (matches Python semantics), found %v", attrs[semconv.LLMToolKey(0)])
	}
}

func TestMiddleware_HideOutputsRedactsContent(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"id","object":"chat.completion","created":0,"model":"gpt-4o",
			"choices":[{"index":0,"finish_reason":"stop","logprobs":null,"message":{"role":"assistant","content":"the answer is 42"}}],
			"usage":{"prompt_tokens":5,"completion_tokens":4,"total_tokens":9}
		}`))
	}))
	defer server.Close()

	t.Setenv(instrumentation.EnvHideOutputs, "true")

	recorder := tracetest.NewSpanRecorder()
	tp := trace.NewTracerProvider(trace.WithSpanProcessor(recorder))
	client := newClient(t, server.URL, tp)

	_, err := client.Chat.Completions.New(context.Background(), openaisdk.ChatCompletionNewParams{
		Model:    shared.ChatModelGPT4o,
		Messages: []openaisdk.ChatCompletionMessageParamUnion{openaisdk.UserMessage("hi")},
	})
	if err != nil {
		t.Fatalf("Chat.Completions.New: %v", err)
	}
	_ = tp.ForceFlush(context.Background())

	attrs := attrMap(recorder.Ended()[0].Attributes())
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
	if got := attrs[semconv.LLMFinishReason]; got != "stop" {
		t.Errorf("finish_reason should not be hidden, got %v", got)
	}
	if got := attrs[semconv.LLMTokenCountPrompt]; got != int64(5) {
		t.Errorf("token counts should not be hidden, got %v", got)
	}
}

func TestMiddleware_WithTraceConfigOverridesEnv(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(okResponse))
	}))
	defer server.Close()

	t.Setenv(instrumentation.EnvHideInputs, "true")
	cfg := instrumentation.TraceConfig{} // all false — should override env

	recorder := tracetest.NewSpanRecorder()
	tp := trace.NewTracerProvider(trace.WithSpanProcessor(recorder))
	client := newClient(t, server.URL, tp, openaiotel.WithTraceConfig(cfg))

	_, err := client.Chat.Completions.New(context.Background(), openaisdk.ChatCompletionNewParams{
		Model:    shared.ChatModelGPT4o,
		Messages: []openaisdk.ChatCompletionMessageParamUnion{openaisdk.UserMessage("visible")},
	})
	if err != nil {
		t.Fatalf("Chat.Completions.New: %v", err)
	}
	_ = tp.ForceFlush(context.Background())

	attrs := attrMap(recorder.Ended()[0].Attributes())
	if got := attrs[semconv.InputValue]; got != "visible" {
		t.Errorf("explicit WithTraceConfig should override env: got %v", got)
	}
}

func TestMiddleware_HideLLMToolsOmitsAdvertisedTools(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(okResponse))
	}))
	defer server.Close()

	t.Setenv(instrumentation.EnvHideLLMTools, "true")

	recorder := tracetest.NewSpanRecorder()
	tp := trace.NewTracerProvider(trace.WithSpanProcessor(recorder))
	client := newClient(t, server.URL, tp)

	_, err := client.Chat.Completions.New(context.Background(), openaisdk.ChatCompletionNewParams{
		Model:    shared.ChatModelGPT4o,
		Messages: []openaisdk.ChatCompletionMessageParamUnion{openaisdk.UserMessage("weather?")},
		Tools: []openaisdk.ChatCompletionToolParam{
			{Function: shared.FunctionDefinitionParam{Name: "get_weather"}},
		},
	})
	if err != nil {
		t.Fatalf("Chat.Completions.New: %v", err)
	}
	_ = tp.ForceFlush(context.Background())

	attrs := attrMap(recorder.Ended()[0].Attributes())
	if got, present := attrs[semconv.LLMToolKey(0)]; present {
		t.Errorf("llm.tools should be hidden under HideLLMTools, got %v", got)
	}
	// Input messages must still be present — HideLLMTools is targeted,
	// not the same broad opt-out as HideInputs.
	if _, present := attrs[semconv.LLMInputMessageRoleKey(0)]; !present {
		t.Error("HideLLMTools must not drop llm.input_messages.* — got missing role[0]")
	}
}

func TestMiddleware_NilBodyRequestFlowsThrough(t *testing.T) {
	// A POST request that arrives at /v1/chat/completions with a nil
	// body is unusual but valid HTTP — instrumentation must NOT block
	// it. Span carries only the const LLM attributes.
	hit := false
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hit = true
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"id":"id","object":"chat.completion","created":0,"model":"m","choices":[],"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}`))
	}))
	defer server.Close()

	recorder := tracetest.NewSpanRecorder()
	tp := trace.NewTracerProvider(trace.WithSpanProcessor(recorder))
	mw := openaiotel.Middleware(tp.Tracer("test"))

	req, err := http.NewRequestWithContext(context.Background(), http.MethodPost, server.URL+"/v1/chat/completions", nil)
	if err != nil {
		t.Fatalf("NewRequest: %v", err)
	}
	resp, err := mw(req, http.DefaultClient.Do)
	if err != nil {
		t.Fatalf("middleware should not error on nil-body request: %v", err)
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

func TestMiddleware_NonChatRequestUntouched(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"data":[],"object":"list"}`))
	}))
	defer server.Close()

	recorder := tracetest.NewSpanRecorder()
	tp := trace.NewTracerProvider(trace.WithSpanProcessor(recorder))
	client := newClient(t, server.URL, tp)

	// Models.List hits /v1/models — should pass through with no span.
	_, _ = client.Models.List(context.Background())

	_ = tp.ForceFlush(context.Background())
	if got := len(recorder.Ended()); got != 0 {
		t.Errorf("non-chat-completion requests should not produce spans, got %d", got)
	}
}

func TestMiddleware_AzureHostMapsProviderToAzure(t *testing.T) {
	// Drive the middleware with a synthesised request whose URL.Host
	// matches each of the Azure-hosted patterns. The actual HTTP call
	// is short-circuited by a canned `next` so we don't need to spin
	// up TLS terminations or mock DNS.
	canned := func(*http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: 200,
			Header:     http.Header{"Content-Type": []string{"application/json"}},
			Body:       io.NopCloser(strings.NewReader(okResponse)),
		}, nil
	}

	hosts := []struct {
		name string
		host string
	}{
		{"openai.azure.com subdomain", "my-resource.openai.azure.com"},
		{"services.ai.azure.com subdomain", "my-resource.services.ai.azure.com"},
		{"cognitiveservices.azure.com subdomain", "my-resource.cognitiveservices.azure.com"},
		{"port-suffixed Azure host", "my-resource.openai.azure.com:443"},
	}
	for _, tc := range hosts {
		t.Run(tc.name, func(t *testing.T) {
			recorder := tracetest.NewSpanRecorder()
			tp := trace.NewTracerProvider(trace.WithSpanProcessor(recorder))
			mw := openaiotel.Middleware(tp.Tracer("test"))

			req, err := http.NewRequest(http.MethodPost, "https://"+tc.host+"/openai/deployments/gpt-4o/chat/completions", bytes.NewReader([]byte(`{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}`)))
			if err != nil {
				t.Fatalf("NewRequest: %v", err)
			}
			resp, err := mw(req, canned)
			if err != nil {
				t.Fatalf("middleware: %v", err)
			}
			_, _ = io.Copy(io.Discard, resp.Body)
			resp.Body.Close()

			_ = tp.ForceFlush(context.Background())
			attrs := attrMap(recorder.Ended()[0].Attributes())
			if got := attrs[semconv.LLMProvider]; got != semconv.LLMProviderAzure {
				t.Errorf("Azure host %q should set llm.provider=%q, got %v", tc.host, semconv.LLMProviderAzure, got)
			}
			// llm.system stays openai regardless of host — Azure is a
			// deployment vehicle for OpenAI models.
			if got := attrs[semconv.LLMSystem]; got != semconv.LLMSystemOpenAI {
				t.Errorf("llm.system should remain %q on Azure, got %v", semconv.LLMSystemOpenAI, got)
			}
		})
	}

	// Sanity: a non-Azure host still maps to openai.
	t.Run("api.openai.com stays openai", func(t *testing.T) {
		recorder := tracetest.NewSpanRecorder()
		tp := trace.NewTracerProvider(trace.WithSpanProcessor(recorder))
		mw := openaiotel.Middleware(tp.Tracer("test"))

		req, _ := http.NewRequest(http.MethodPost, "https://api.openai.com/v1/chat/completions", bytes.NewReader([]byte(`{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}`)))
		resp, _ := mw(req, canned)
		_, _ = io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
		_ = tp.ForceFlush(context.Background())
		attrs := attrMap(recorder.Ended()[0].Attributes())
		if got := attrs[semconv.LLMProvider]; got != semconv.LLMProviderOpenAI {
			t.Errorf("api.openai.com should set llm.provider=openai, got %v", got)
		}
	})
}

func TestMiddleware_InvocationParamsCaptureForwardCompatibleFields(t *testing.T) {
	// Newer/common chat-completion params (max_completion_tokens,
	// reasoning_effort, response_format, tool_choice, stream_options,
	// presence/frequency penalty, n, seed, logit_bias, ...) must round-
	// trip through llm.invocation_parameters without us having to
	// enumerate them. The strip-then-marshal approach drops only the
	// fields we surface separately (messages, functions, tools).
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(okResponse))
	}))
	defer server.Close()

	recorder := tracetest.NewSpanRecorder()
	tp := trace.NewTracerProvider(trace.WithSpanProcessor(recorder))
	client := newClient(t, server.URL, tp)

	_, err := client.Chat.Completions.New(context.Background(), openaisdk.ChatCompletionNewParams{
		Model:               shared.ChatModelGPT4o,
		MaxCompletionTokens: openaisdk.Int(64),
		PresencePenalty:     openaisdk.Float(0.2),
		FrequencyPenalty:    openaisdk.Float(0.1),
		Seed:                openaisdk.Int(7),
		ReasoningEffort:     shared.ReasoningEffortMedium,
		Messages:            []openaisdk.ChatCompletionMessageParamUnion{openaisdk.UserMessage("hi")},
	})
	if err != nil {
		t.Fatalf("Chat.Completions.New: %v", err)
	}
	_ = tp.ForceFlush(context.Background())

	attrs := attrMap(recorder.Ended()[0].Attributes())
	rawInvocation, _ := attrs[semconv.LLMInvocationParameters].(string)
	if rawInvocation == "" {
		t.Fatalf("llm.invocation_parameters missing or wrong type: %T = %v", attrs[semconv.LLMInvocationParameters], attrs[semconv.LLMInvocationParameters])
	}

	var got map[string]any
	if err := json.Unmarshal([]byte(rawInvocation), &got); err != nil {
		t.Fatalf("invocation_parameters is not valid JSON: %v\nraw=%s", err, rawInvocation)
	}

	// Forward-compatible fields the SDK serialised — all must appear
	// without us enumerating them.
	for _, k := range []string{
		"model",
		"max_completion_tokens",
		"presence_penalty",
		"frequency_penalty",
		"seed",
		"reasoning_effort",
	} {
		if _, present := got[k]; !present {
			t.Errorf("expected %q in invocation_parameters, got %v", k, got)
		}
	}

	// Fields that we surface separately must NOT appear in
	// invocation_parameters.
	for _, k := range []string{"messages", "functions", "tools"} {
		if _, present := got[k]; present {
			t.Errorf("%q should not be in invocation_parameters; it's surfaced elsewhere", k)
		}
	}
}

// okResponse is a minimal valid chat.completion response body used by
// tests that only care about request-side attribute behaviour.
const okResponse = `{
	"id":"id","object":"chat.completion","created":0,"model":"gpt-4o",
	"choices":[{"index":0,"finish_reason":"stop","logprobs":null,"message":{"role":"assistant","content":"ok"}}],
	"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}
}`

func attrMap(attrs []attribute.KeyValue) map[string]any {
	out := make(map[string]any, len(attrs))
	for _, kv := range attrs {
		out[string(kv.Key)] = kv.Value.AsInterface()
	}
	return out
}

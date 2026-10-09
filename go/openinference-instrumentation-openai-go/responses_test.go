package openai_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strconv"
	"strings"
	"testing"

	openaisdk "github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/responses"
	"github.com/openai/openai-go/shared"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"

	"github.com/Arize-ai/openinference/go/openinference-instrumentation"
	openaiotel "github.com/Arize-ai/openinference/go/openinference-instrumentation-openai-go"
	"github.com/Arize-ai/openinference/go/openinference-semantic-conventions"
)

const responsesSpanName = "openai.responses.create"

// textResponse is a Responses API body with a single assistant message.
const textResponse = `{
	"id":"resp_123","object":"response","created_at":0,"status":"completed",
	"model":"gpt-6.1-sol-2026-09-01",
	"output":[{
		"type":"message","id":"msg_1","status":"completed","role":"assistant",
		"content":[{"type":"output_text","text":"Observability is seeing inside a system.","annotations":[]}]
	}],
	"usage":{
		"input_tokens":20,"output_tokens":12,"total_tokens":32,
		"input_tokens_details":{"cached_tokens":4},
		"output_tokens_details":{"reasoning_tokens":7}
	}
}`

// serveJSON starts a server that answers every request with body. When
// gotReq is non-nil it receives the last request body, so tests can
// compare input.value with what went over the wire.
func serveJSON(t *testing.T, status int, body string, gotReq *[]byte) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if gotReq != nil {
			*gotReq, _ = io.ReadAll(r.Body)
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(server.Close)
	return server
}

// contentKey returns the key of a message_content child, e.g.
// llm.output_messages.0.message.contents.1.message_content.text.
func contentKey(messages string, i, k int, child string) string {
	return messages + "." + strconv.Itoa(i) + "." + semconv.MessageContents + "." + strconv.Itoa(k) + "." + child
}

// assertExactAttrs fails the test unless attrs holds exactly the keys
// in want, with equal values. It is the Go equivalent of the pop-style
// assertions in the Python tests: unexpected extras fail as well as
// missing or wrong values.
func assertExactAttrs(t *testing.T, attrs, want map[string]any) {
	t.Helper()
	for k, v := range want {
		got, present := attrs[k]
		if !present {
			t.Errorf("missing attr %s (want %v)", k, v)
		} else if !reflect.DeepEqual(got, v) {
			t.Errorf("attr %s: got %v want %v", k, got, v)
		}
	}
	for k, v := range attrs {
		if _, expected := want[k]; !expected {
			t.Errorf("unexpected attr %s = %v", k, v)
		}
	}
}

// compactJSON returns the compacted form of a JSON fragment from a
// request body, for comparing llm.tools.* values.
func compactJSON(t *testing.T, raw json.RawMessage) string {
	t.Helper()
	var buf bytes.Buffer
	if err := json.Compact(&buf, raw); err != nil {
		t.Fatalf("compact: %v", err)
	}
	return buf.String()
}

func TestResponses_TextResponse(t *testing.T) {
	var upstreamReq []byte
	server := serveJSON(t, http.StatusOK, textResponse, &upstreamReq)

	recorder := tracetest.NewSpanRecorder()
	tp := trace.NewTracerProvider(trace.WithSpanProcessor(recorder))
	client := newClient(t, server.URL, tp)

	resp, err := client.Responses.New(context.Background(), responses.ResponseNewParams{
		Model:           "gpt-6.1-sol",
		Instructions:    openaisdk.String("Answer in one sentence."),
		Input:           responses.ResponseNewParamsInputUnion{OfString: openaisdk.String("What is observability?")},
		MaxOutputTokens: openaisdk.Int(200),
	})
	if err != nil {
		t.Fatalf("Responses.New: %v", err)
	}
	if got := resp.OutputText(); got != "Observability is seeing inside a system." {
		t.Fatalf("unexpected output text: %q", got)
	}

	_ = tp.ForceFlush(context.Background())
	spans := recorder.Ended()
	if len(spans) != 1 {
		t.Fatalf("expected 1 span, got %d", len(spans))
	}
	if got := spans[0].Name(); got != responsesSpanName {
		t.Errorf("span name: got %q want %q", got, responsesSpanName)
	}
	if got := spans[0].Status().Code; got != codes.Unset {
		t.Errorf("span status: got %s want Unset", got)
	}

	assertExactAttrs(t, attrMap(spans[0].Attributes()), map[string]any{
		semconv.OpenInferenceSpanKind:        semconv.SpanKindLLM,
		semconv.LLMSystem:                    semconv.LLMSystemOpenAI,
		semconv.LLMProvider:                  semconv.LLMProviderOpenAI,
		semconv.LLMModelName:                 "gpt-6.1-sol-2026-09-01", // response wins
		semconv.LLMInvocationParameters:      `{"max_output_tokens":200,"model":"gpt-6.1-sol"}`,
		semconv.LLMInputMessageRoleKey(0):    "system",
		semconv.LLMInputMessageContentKey(0): "Answer in one sentence.",
		semconv.LLMInputMessageRoleKey(1):    "user",
		semconv.LLMInputMessageContentKey(1): "What is observability?",
		semconv.InputValue:                   string(upstreamReq),
		semconv.InputMimeType:                semconv.MimeTypeJSON,
		semconv.LLMOutputMessageRoleKey(0):   "assistant",
		contentKey(semconv.LLMOutputMessages, 0, 0, semconv.MessageContentType): "text",
		contentKey(semconv.LLMOutputMessages, 0, 0, semconv.MessageContentText): "Observability is seeing inside a system.",
		semconv.OutputValue:                             textResponse,
		semconv.OutputMimeType:                          semconv.MimeTypeJSON,
		semconv.LLMTokenCountPrompt:                     int64(20),
		semconv.LLMTokenCountCompletion:                 int64(12),
		semconv.LLMTokenCountTotal:                      int64(32),
		semconv.LLMTokenCountPromptDetailsCacheRead:     int64(4),
		semconv.LLMTokenCountCompletionDetailsReasoning: int64(7),
	})
}

func TestResponses_FunctionCallResponse(t *testing.T) {
	const body = `{
		"id":"resp_1","object":"response","created_at":0,"status":"completed","model":"gpt-6.1-sol",
		"output":[
			{"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"Need the weather."}],"encrypted_content":"enc=="},
			{"type":"function_call","id":"fc_1","call_id":"call_abc","name":"get_weather","arguments":"{\"city\":\"Paris\"}","status":"completed"}
		],
		"usage":{"input_tokens":50,"output_tokens":18,"total_tokens":68,
			"input_tokens_details":{"cached_tokens":0,"cache_write_tokens":0},
			"output_tokens_details":{"reasoning_tokens":0}}
	}`
	var upstreamReq []byte
	server := serveJSON(t, http.StatusOK, body, &upstreamReq)

	recorder := tracetest.NewSpanRecorder()
	tp := trace.NewTracerProvider(trace.WithSpanProcessor(recorder))
	client := newClient(t, server.URL, tp)

	resp, err := client.Responses.New(context.Background(), responses.ResponseNewParams{
		Model: "gpt-6.1-sol",
		Input: responses.ResponseNewParamsInputUnion{OfString: openaisdk.String("Weather in Paris?")},
		Tools: []responses.ToolUnionParam{
			responses.ToolParamOfFunction("get_weather", map[string]any{
				"type":       "object",
				"properties": map[string]any{"city": map[string]any{"type": "string"}},
				"required":   []string{"city"},
			}, true),
		},
	})
	if err != nil {
		t.Fatalf("Responses.New: %v", err)
	}
	if len(resp.Output) != 2 {
		t.Fatalf("expected 2 output items, got %d", len(resp.Output))
	}

	var sent struct {
		Tools []json.RawMessage `json:"tools"`
	}
	if err := json.Unmarshal(upstreamReq, &sent); err != nil || len(sent.Tools) != 1 {
		t.Fatalf("upstream request tools: %v (%s)", err, upstreamReq)
	}

	_ = tp.ForceFlush(context.Background())
	assertExactAttrs(t, attrMap(recorder.Ended()[0].Attributes()), map[string]any{
		semconv.OpenInferenceSpanKind:   semconv.SpanKindLLM,
		semconv.LLMSystem:               semconv.LLMSystemOpenAI,
		semconv.LLMProvider:             semconv.LLMProviderOpenAI,
		semconv.LLMModelName:            "gpt-6.1-sol",
		semconv.LLMInvocationParameters: `{"model":"gpt-6.1-sol"}`,
		semconv.LLMToolKey(0):           compactJSON(t, sent.Tools[0]),
		// No instructions, so the string input is message 0.
		semconv.LLMInputMessageRoleKey(0):    "user",
		semconv.LLMInputMessageContentKey(0): "Weather in Paris?",
		semconv.InputValue:                   string(upstreamReq),
		semconv.InputMimeType:                semconv.MimeTypeJSON,
		semconv.LLMOutputMessageRoleKey(0):   "assistant",
		contentKey(semconv.LLMOutputMessages, 0, 0, semconv.MessageContentType):             "reasoning",
		contentKey(semconv.LLMOutputMessages, 0, 0, semconv.MessageContentText):             "Need the weather.",
		contentKey(semconv.LLMOutputMessages, 0, 0, semconv.MessageContentEncryptedContent): "enc==",
		semconv.LLMOutputMessageRoleKey(1):                                                  "assistant",
		semconv.LLMOutputMessageToolCallKey(1, 0, semconv.ToolCallID):                       "call_abc",
		semconv.LLMOutputMessageToolCallKey(1, 0, semconv.ToolCallFunctionName):             "get_weather",
		semconv.LLMOutputMessageToolCallKey(1, 0, semconv.ToolCallFunctionArgumentsJSON):    `{"city":"Paris"}`,
		semconv.OutputValue:                             body,
		semconv.OutputMimeType:                          semconv.MimeTypeJSON,
		semconv.LLMTokenCountPrompt:                     int64(50),
		semconv.LLMTokenCountCompletion:                 int64(18),
		semconv.LLMTokenCountTotal:                      int64(68),
		semconv.LLMTokenCountPromptDetailsCacheRead:     int64(0), // reported zeros are kept
		semconv.LLMTokenCountPromptDetailsCacheWrite:    int64(0),
		semconv.LLMTokenCountCompletionDetailsReasoning: int64(0),
	})
}

func TestResponses_ToolOutputsInInput(t *testing.T) {
	var upstreamReq []byte
	server := serveJSON(t, http.StatusOK, textResponse, &upstreamReq)

	recorder := tracetest.NewSpanRecorder()
	tp := trace.NewTracerProvider(trace.WithSpanProcessor(recorder))
	client := newClient(t, server.URL, tp)

	_, err := client.Responses.New(context.Background(), responses.ResponseNewParams{
		Model:              "gpt-6.1-sol",
		PreviousResponseID: openaisdk.String("resp_1"),
		Input: responses.ResponseNewParamsInputUnion{OfInputItemList: responses.ResponseInputParam{
			responses.ResponseInputItemParamOfInputMessage(responses.ResponseInputMessageContentListParam{
				{OfInputText: &responses.ResponseInputTextParam{Text: "Weather in Paris?"}},
				{OfInputImage: &responses.ResponseInputImageParam{ImageURL: openaisdk.String("https://example.com/paris.png")}},
				{OfInputText: &responses.ResponseInputTextParam{Text: "Use Celsius."}},
			}, "user"),
			responses.ResponseInputItemParamOfFunctionCall(`{"city":"Paris"}`, "call_abc", "get_weather"),
			responses.ResponseInputItemParamOfFunctionCallOutput("call_abc", `{"temp_c":18}`),
		}},
	})
	if err != nil {
		t.Fatalf("Responses.New: %v", err)
	}

	_ = tp.ForceFlush(context.Background())
	assertExactAttrs(t, attrMap(recorder.Ended()[0].Attributes()), map[string]any{
		semconv.OpenInferenceSpanKind:     semconv.SpanKindLLM,
		semconv.LLMSystem:                 semconv.LLMSystemOpenAI,
		semconv.LLMProvider:               semconv.LLMProviderOpenAI,
		semconv.LLMModelName:              "gpt-6.1-sol-2026-09-01",
		semconv.LLMInvocationParameters:   `{"model":"gpt-6.1-sol","previous_response_id":"resp_1"}`,
		semconv.InputValue:                string(upstreamReq),
		semconv.InputMimeType:             semconv.MimeTypeJSON,
		semconv.LLMInputMessageRoleKey(0): "user",
		contentKey(semconv.LLMInputMessages, 0, 0, semconv.MessageContentType): "text",
		contentKey(semconv.LLMInputMessages, 0, 0, semconv.MessageContentText): "Weather in Paris?",
		// The image part (index 1) is not recorded but keeps its index.
		contentKey(semconv.LLMInputMessages, 0, 2, semconv.MessageContentType):          "text",
		contentKey(semconv.LLMInputMessages, 0, 2, semconv.MessageContentText):          "Use Celsius.",
		semconv.LLMInputMessageRoleKey(1):                                               "assistant",
		semconv.LLMInputMessageToolCallKey(1, 0, semconv.ToolCallID):                    "call_abc",
		semconv.LLMInputMessageToolCallKey(1, 0, semconv.ToolCallFunctionName):          "get_weather",
		semconv.LLMInputMessageToolCallKey(1, 0, semconv.ToolCallFunctionArgumentsJSON): `{"city":"Paris"}`,
		semconv.LLMInputMessageRoleKey(2):                                               "tool",
		semconv.LLMInputMessageToolCallIDKey(2):                                         "call_abc",
		semconv.LLMInputMessageContentKey(2):                                            `{"temp_c":18}`,
		semconv.LLMOutputMessageRoleKey(0):                                              "assistant",
		contentKey(semconv.LLMOutputMessages, 0, 0, semconv.MessageContentType):         "text",
		contentKey(semconv.LLMOutputMessages, 0, 0, semconv.MessageContentText):         "Observability is seeing inside a system.",
		semconv.OutputValue:                             textResponse,
		semconv.OutputMimeType:                          semconv.MimeTypeJSON,
		semconv.LLMTokenCountPrompt:                     int64(20),
		semconv.LLMTokenCountCompletion:                 int64(12),
		semconv.LLMTokenCountTotal:                      int64(32),
		semconv.LLMTokenCountPromptDetailsCacheRead:     int64(4),
		semconv.LLMTokenCountCompletionDetailsReasoning: int64(7),
	})
}

func TestResponses_OtherItemTypes(t *testing.T) {
	// Drive the middleware with raw bodies to cover item shapes the
	// SDK's param helpers make awkward to build: an EasyInputMessage
	// without "type", custom tool calls, a function_call_output whose
	// output is a content list, hosted tool calls, refusals, and an
	// unknown item type that must keep its index.
	reqBody := `{"model":"gpt-6.1-sol","input":[
		{"role":"developer","content":"Be terse."},
		{"type":"custom_tool_call","call_id":"call_c","name":"run_sql","input":"SELECT 1"},
		{"type":"custom_tool_call_output","call_id":"call_c","output":"1"},
		{"type":"item_reference","id":"msg_0"},
		{"type":"function_call_output","call_id":"call_f","output":[{"type":"input_text","text":"done"}]}
	]}`
	respBody := `{"id":"resp_1","object":"response","model":"gpt-6.1-sol","output":[
		{"type":"web_search_call","id":"ws_1","status":"completed"},
		{"type":"file_search_call","id":"fs_1","status":"completed","queries":["q"]},
		{"type":"message","role":"assistant","content":[{"type":"refusal","refusal":"I can't help with that."}]}
	]}`

	recorder := tracetest.NewSpanRecorder()
	tp := trace.NewTracerProvider(trace.WithSpanProcessor(recorder))
	mw := openaiotel.Middleware(tp.Tracer("test"))

	req, err := http.NewRequest(http.MethodPost, "https://api.openai.com/v1/responses", strings.NewReader(reqBody))
	if err != nil {
		t.Fatalf("NewRequest: %v", err)
	}
	resp, err := mw(req, func(*http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": []string{"application/json"}},
			Body:       io.NopCloser(strings.NewReader(respBody)),
		}, nil
	})
	if err != nil {
		t.Fatalf("middleware: %v", err)
	}
	_, _ = io.Copy(io.Discard, resp.Body)
	resp.Body.Close()
	_ = tp.ForceFlush(context.Background())

	assertExactAttrs(t, attrMap(recorder.Ended()[0].Attributes()), map[string]any{
		semconv.OpenInferenceSpanKind:                                semconv.SpanKindLLM,
		semconv.LLMSystem:                                            semconv.LLMSystemOpenAI,
		semconv.LLMProvider:                                          semconv.LLMProviderOpenAI,
		semconv.LLMModelName:                                         "gpt-6.1-sol",
		semconv.LLMInvocationParameters:                              `{"model":"gpt-6.1-sol"}`,
		semconv.InputValue:                                           reqBody,
		semconv.InputMimeType:                                        semconv.MimeTypeJSON,
		semconv.LLMInputMessageRoleKey(0):                            "developer",
		semconv.LLMInputMessageContentKey(0):                         "Be terse.",
		semconv.LLMInputMessageRoleKey(1):                            "assistant",
		semconv.LLMInputMessageToolCallKey(1, 0, semconv.ToolCallID): "call_c",
		semconv.LLMInputMessageToolCallKey(1, 0, semconv.ToolCallFunctionName):          "run_sql",
		semconv.LLMInputMessageToolCallKey(1, 0, semconv.ToolCallFunctionArgumentsJSON): `{"input":"SELECT 1"}`,
		semconv.LLMInputMessageRoleKey(2):                                               "tool",
		semconv.LLMInputMessageToolCallIDKey(2):                                         "call_c",
		semconv.LLMInputMessageContentKey(2):                                            "1",
		// item_reference (index 3) is skipped; the next item stays at 4.
		semconv.LLMInputMessageRoleKey(4):                                       "tool",
		semconv.LLMInputMessageToolCallIDKey(4):                                 "call_f",
		semconv.LLMInputMessageContentKey(4):                                    `[{"type":"input_text","text":"done"}]`,
		semconv.LLMOutputMessageRoleKey(0):                                      "assistant",
		semconv.LLMOutputMessageToolCallKey(0, 0, semconv.ToolCallID):           "ws_1",
		semconv.LLMOutputMessageToolCallKey(0, 0, semconv.ToolCallFunctionName): "web_search_call",
		semconv.LLMOutputMessageRoleKey(1):                                      "assistant",
		semconv.LLMOutputMessageToolCallKey(1, 0, semconv.ToolCallID):           "fs_1",
		semconv.LLMOutputMessageToolCallKey(1, 0, semconv.ToolCallFunctionName): "file_search_call",
		semconv.LLMOutputMessageRoleKey(2):                                      "assistant",
		contentKey(semconv.LLMOutputMessages, 2, 0, semconv.MessageContentType): "text",
		contentKey(semconv.LLMOutputMessages, 2, 0, semconv.MessageContentText): "I can't help with that.",
		semconv.OutputValue:    respBody,
		semconv.OutputMimeType: semconv.MimeTypeJSON,
	})
}

func TestResponses_AzureHostMapsProviderToAzure(t *testing.T) {
	canned := func(*http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": []string{"application/json"}},
			Body:       io.NopCloser(strings.NewReader(textResponse)),
		}, nil
	}
	for _, url := range []string{
		"https://my-resource.openai.azure.com/openai/v1/responses",
		// openai-go/azure sends Responses calls to /openai/responses
		// (no deployment rewrite) with an api-version query.
		"https://my-resource.openai.azure.com/openai/responses?api-version=2025-04-01-preview",
		"https://my-resource.services.ai.azure.com/openai/v1/responses",
		"https://my-resource.cognitiveservices.azure.com:443/openai/v1/responses",
	} {
		t.Run(url, func(t *testing.T) {
			recorder := tracetest.NewSpanRecorder()
			tp := trace.NewTracerProvider(trace.WithSpanProcessor(recorder))
			mw := openaiotel.Middleware(tp.Tracer("test"))

			req, err := http.NewRequest(http.MethodPost, url, strings.NewReader(`{"model":"gpt-6.1-sol","input":"hi"}`))
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
			spans := recorder.Ended()
			if len(spans) != 1 {
				t.Fatalf("expected 1 span, got %d", len(spans))
			}
			if got := spans[0].Name(); got != responsesSpanName {
				t.Errorf("span name: got %q", got)
			}
			attrs := attrMap(spans[0].Attributes())
			if got := attrs[semconv.LLMProvider]; got != semconv.LLMProviderAzure {
				t.Errorf("llm.provider: got %v want %q", got, semconv.LLMProviderAzure)
			}
			if got := attrs[semconv.LLMSystem]; got != semconv.LLMSystemOpenAI {
				t.Errorf("llm.system: got %v", got)
			}
		})
	}
}

func TestResponses_ErrorResponse(t *testing.T) {
	var upstreamReq []byte
	server := serveJSON(t, http.StatusBadRequest,
		`{"error":{"message":"Unsupported parameter","type":"invalid_request_error","param":"temperature","code":"unsupported_parameter"}}`, &upstreamReq)

	recorder := tracetest.NewSpanRecorder()
	tp := trace.NewTracerProvider(trace.WithSpanProcessor(recorder))
	client := openaisdk.NewClient(
		option.WithBaseURL(server.URL),
		option.WithAPIKey("test-key"),
		option.WithMaxRetries(0),
		option.WithMiddleware(openaiotel.Middleware(tp.Tracer("test"))),
	)

	_, err := client.Responses.New(context.Background(), responses.ResponseNewParams{
		Model: "gpt-6.1-sol",
		Input: responses.ResponseNewParamsInputUnion{OfString: openaisdk.String("hi")},
	})
	if err == nil {
		t.Fatal("expected an error from a 400 response")
	}
	_ = tp.ForceFlush(context.Background())

	spans := recorder.Ended()
	if len(spans) != 1 {
		t.Fatalf("expected 1 span, got %d", len(spans))
	}
	if got := spans[0].Status(); got.Code != codes.Error || got.Description != "Bad Request" {
		t.Errorf("span status: got %+v want Error/Bad Request", got)
	}
	// Only request-side attributes: no output, output messages, or
	// token counts parsed out of the error body.
	assertExactAttrs(t, attrMap(spans[0].Attributes()), map[string]any{
		semconv.OpenInferenceSpanKind:        semconv.SpanKindLLM,
		semconv.LLMSystem:                    semconv.LLMSystemOpenAI,
		semconv.LLMProvider:                  semconv.LLMProviderOpenAI,
		semconv.LLMModelName:                 "gpt-6.1-sol",
		semconv.LLMInvocationParameters:      `{"model":"gpt-6.1-sol"}`,
		semconv.InputValue:                   string(upstreamReq),
		semconv.InputMimeType:                semconv.MimeTypeJSON,
		semconv.LLMInputMessageRoleKey(0):    "user",
		semconv.LLMInputMessageContentKey(0): "hi",
	})
}

func TestResponses_TransportErrorRecordsException(t *testing.T) {
	recorder := tracetest.NewSpanRecorder()
	tp := trace.NewTracerProvider(trace.WithSpanProcessor(recorder))
	mw := openaiotel.Middleware(tp.Tracer("test"))

	req, err := http.NewRequest(http.MethodPost, "https://api.openai.com/v1/responses", strings.NewReader(`{"model":"gpt-6.1-sol","input":"hi"}`))
	if err != nil {
		t.Fatalf("NewRequest: %v", err)
	}
	_, err = mw(req, func(*http.Request) (*http.Response, error) {
		return nil, errors.New("connection reset")
	})
	if err == nil {
		t.Fatal("expected the transport error to propagate")
	}
	_ = tp.ForceFlush(context.Background())

	spans := recorder.Ended()
	if len(spans) != 1 {
		t.Fatalf("expected 1 span, got %d", len(spans))
	}
	if got := spans[0].Status(); got.Code != codes.Error || got.Description != "connection reset" {
		t.Errorf("span status: got %+v", got)
	}
	var foundException bool
	for _, e := range spans[0].Events() {
		if e.Name == "exception" {
			foundException = true
		}
	}
	if !foundException {
		t.Errorf("expected exception event, got %+v", spans[0].Events())
	}
}

func TestResponses_StreamingPassesThrough(t *testing.T) {
	const stream = "event: response.created\n" +
		"data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\",\"status\":\"in_progress\"}}\n\n" +
		"event: response.output_text.delta\n" +
		"data: {\"type\":\"response.output_text.delta\",\"delta\":\"hi\"}\n\n" +
		"event: response.completed\n" +
		"data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"status\":\"completed\"}}\n\n"
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(stream))
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
	}))
	defer server.Close()

	recorder := tracetest.NewSpanRecorder()
	tp := trace.NewTracerProvider(trace.WithSpanProcessor(recorder))
	mw := openaiotel.Middleware(tp.Tracer("test"))

	const reqBody = `{"model":"gpt-6.1-sol","stream":true,"input":"hi"}`
	req, err := http.NewRequestWithContext(context.Background(), http.MethodPost, server.URL+"/v1/responses", strings.NewReader(reqBody))
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
	got, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("drain body: %v", err)
	}
	if string(got) != stream {
		t.Errorf("stream bytes changed in transit:\ngot  %q\nwant %q", got, stream)
	}
	spans := recorder.Ended()
	if len(spans) != 1 {
		t.Fatalf("span did not end on EOF: got %d ended spans", len(spans))
	}
	if name := spans[0].Name(); name != responsesSpanName {
		t.Errorf("span name: got %q", name)
	}
	// Request attributes only, as for streaming Chat Completions.
	assertExactAttrs(t, attrMap(spans[0].Attributes()), map[string]any{
		semconv.OpenInferenceSpanKind:        semconv.SpanKindLLM,
		semconv.LLMSystem:                    semconv.LLMSystemOpenAI,
		semconv.LLMProvider:                  semconv.LLMProviderOpenAI,
		semconv.LLMModelName:                 "gpt-6.1-sol",
		semconv.LLMInvocationParameters:      `{"model":"gpt-6.1-sol","stream":true}`,
		semconv.InputValue:                   reqBody,
		semconv.InputMimeType:                semconv.MimeTypeJSON,
		semconv.LLMInputMessageRoleKey(0):    "user",
		semconv.LLMInputMessageContentKey(0): "hi",
	})
}

func TestResponses_OtherResponsesEndpointsUntouched(t *testing.T) {
	server := serveJSON(t, http.StatusOK, textResponse, nil)

	recorder := tracetest.NewSpanRecorder()
	tp := trace.NewTracerProvider(trace.WithSpanProcessor(recorder))
	client := newClient(t, server.URL, tp)

	// GET /responses/{id} and POST /responses/{id}/cancel are not model
	// calls and must not produce LLM spans.
	_, _ = client.Responses.Get(context.Background(), "resp_123", responses.ResponseGetParams{})
	_, _ = client.Responses.Cancel(context.Background(), "resp_123")

	_ = tp.ForceFlush(context.Background())
	if got := len(recorder.Ended()); got != 0 {
		t.Errorf("non-create responses endpoints should not produce spans, got %d", got)
	}
}

func TestResponses_SuppressedContextEmitsNoSpan(t *testing.T) {
	server := serveJSON(t, http.StatusOK, textResponse, nil)

	recorder := tracetest.NewSpanRecorder()
	tp := trace.NewTracerProvider(trace.WithSpanProcessor(recorder))
	client := newClient(t, server.URL, tp)

	ctx := instrumentation.WithSuppression(context.Background())
	if _, err := client.Responses.New(ctx, responses.ResponseNewParams{
		Model: "gpt-6.1-sol",
		Input: responses.ResponseNewParamsInputUnion{OfString: openaisdk.String("hi")},
	}); err != nil {
		t.Fatalf("Responses.New: %v", err)
	}
	_ = tp.ForceFlush(context.Background())
	if got := len(recorder.Ended()); got != 0 {
		t.Errorf("suppressed context should produce no span, got %d", got)
	}
}

func TestResponses_ContextAttributesPropagateToSpan(t *testing.T) {
	server := serveJSON(t, http.StatusOK, textResponse, nil)

	recorder := tracetest.NewSpanRecorder()
	tp := trace.NewTracerProvider(trace.WithSpanProcessor(recorder))
	client := newClient(t, server.URL, tp)

	ctx := context.Background()
	ctx = instrumentation.WithSession(ctx, "session-abc")
	ctx = instrumentation.WithUser(ctx, "user-xyz")
	ctx = instrumentation.WithMetadata(ctx, `{"team":"platform"}`)
	ctx = instrumentation.WithTags(ctx, "prod", "canary")

	if _, err := client.Responses.New(ctx, responses.ResponseNewParams{
		Model: "gpt-6.1-sol",
		Input: responses.ResponseNewParamsInputUnion{OfString: openaisdk.String("hi")},
	}); err != nil {
		t.Fatalf("Responses.New: %v", err)
	}
	_ = tp.ForceFlush(context.Background())

	attrs := attrMap(recorder.Ended()[0].Attributes())
	want := map[string]any{
		"session.id": "session-abc",
		"user.id":    "user-xyz",
		"metadata":   `{"team":"platform"}`,
		"tag.tags":   []string{"prod", "canary"},
	}
	for k, v := range want {
		if got := attrs[k]; !reflect.DeepEqual(got, v) {
			t.Errorf("%s: got %v want %v", k, got, v)
		}
	}
}

func TestResponses_HideTextRedactsMessageContent(t *testing.T) {
	server := serveJSON(t, http.StatusOK, `{"id":"resp_1","object":"response","model":"gpt-6.1-sol","output":[
		{"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"secret reasoning"}]},
		{"type":"message","role":"assistant","content":[{"type":"output_text","text":"secret answer"}]}
	]}`, nil)

	recorder := tracetest.NewSpanRecorder()
	tp := trace.NewTracerProvider(trace.WithSpanProcessor(recorder))
	client := newClient(t, server.URL, tp, openaiotel.WithTraceConfig(instrumentation.TraceConfig{
		HideInputText:  true,
		HideOutputText: true,
	}))

	_, err := client.Responses.New(context.Background(), responses.ResponseNewParams{
		Model:        "gpt-6.1-sol",
		Instructions: openaisdk.String("secret instructions"),
		Input: responses.ResponseNewParamsInputUnion{OfInputItemList: responses.ResponseInputParam{
			responses.ResponseInputItemParamOfMessage("the secret password is 12345", responses.EasyInputMessageRoleUser),
			responses.ResponseInputItemParamOfFunctionCallOutput("call_abc", "secret tool output"),
		}},
	})
	if err != nil {
		t.Fatalf("Responses.New: %v", err)
	}
	_ = tp.ForceFlush(context.Background())

	attrs := attrMap(recorder.Ended()[0].Attributes())
	for _, k := range []string{
		semconv.LLMInputMessageContentKey(0),
		semconv.LLMInputMessageContentKey(1),
		semconv.LLMInputMessageContentKey(2),
		contentKey(semconv.LLMOutputMessages, 0, 0, semconv.MessageContentText),
		contentKey(semconv.LLMOutputMessages, 1, 0, semconv.MessageContentText),
	} {
		if got := attrs[k]; got != instrumentation.RedactedValue {
			t.Errorf("%s: got %v want %s", k, got, instrumentation.RedactedValue)
		}
	}
	// Structure stays: roles, content types, and the tool call id are
	// not text.
	want := map[string]any{
		semconv.LLMInputMessageRoleKey(2):                                       "tool",
		semconv.LLMInputMessageToolCallIDKey(2):                                 "call_abc",
		contentKey(semconv.LLMOutputMessages, 1, 0, semconv.MessageContentType): "text",
	}
	for k, v := range want {
		if got := attrs[k]; got != v {
			t.Errorf("%s: got %v want %v", k, got, v)
		}
	}
}

func TestResponses_HideInputsAndOutputs(t *testing.T) {
	server := serveJSON(t, http.StatusOK, textResponse, nil)

	t.Setenv(instrumentation.EnvHideInputs, "true")
	t.Setenv(instrumentation.EnvHideOutputs, "true")

	recorder := tracetest.NewSpanRecorder()
	tp := trace.NewTracerProvider(trace.WithSpanProcessor(recorder))
	client := newClient(t, server.URL, tp)

	_, err := client.Responses.New(context.Background(), responses.ResponseNewParams{
		Model:        "gpt-6.1-sol",
		Instructions: openaisdk.String("secret instructions"),
		Input:        responses.ResponseNewParamsInputUnion{OfString: openaisdk.String("secret input")},
		Tools:        []responses.ToolUnionParam{responses.ToolParamOfFunction("get_weather", nil, false)},
	})
	if err != nil {
		t.Fatalf("Responses.New: %v", err)
	}
	_ = tp.ForceFlush(context.Background())

	// Values are redacted; the mime types, messages, and tools are
	// dropped. Token counts and the model are never hidden.
	assertExactAttrs(t, attrMap(recorder.Ended()[0].Attributes()), map[string]any{
		semconv.OpenInferenceSpanKind:                   semconv.SpanKindLLM,
		semconv.LLMSystem:                               semconv.LLMSystemOpenAI,
		semconv.LLMProvider:                             semconv.LLMProviderOpenAI,
		semconv.LLMModelName:                            "gpt-6.1-sol-2026-09-01",
		semconv.LLMInvocationParameters:                 `{"model":"gpt-6.1-sol"}`,
		semconv.InputValue:                              instrumentation.RedactedValue,
		semconv.OutputValue:                             instrumentation.RedactedValue,
		semconv.LLMTokenCountPrompt:                     int64(20),
		semconv.LLMTokenCountCompletion:                 int64(12),
		semconv.LLMTokenCountTotal:                      int64(32),
		semconv.LLMTokenCountPromptDetailsCacheRead:     int64(4),
		semconv.LLMTokenCountCompletionDetailsReasoning: int64(7),
	})
}

func TestResponses_TargetedHideFlags(t *testing.T) {
	cases := []struct {
		name    string
		cfg     instrumentation.TraceConfig
		absent  []string
		present []string
	}{
		{
			name:    "HideInputMessages",
			cfg:     instrumentation.TraceConfig{HideInputMessages: true},
			absent:  []string{semconv.LLMInputMessageRoleKey(0), semconv.LLMInputMessageContentKey(0)},
			present: []string{semconv.InputValue, semconv.LLMToolKey(0), semconv.LLMOutputMessageRoleKey(0)},
		},
		{
			name:    "HideOutputMessages",
			cfg:     instrumentation.TraceConfig{HideOutputMessages: true},
			absent:  []string{semconv.LLMOutputMessageRoleKey(0), contentKey(semconv.LLMOutputMessages, 0, 0, semconv.MessageContentText)},
			present: []string{semconv.OutputValue, semconv.OutputMimeType, semconv.LLMInputMessageRoleKey(0)},
		},
		{
			name:    "HideLLMInvocationParameters",
			cfg:     instrumentation.TraceConfig{HideLLMInvocationParameters: true},
			absent:  []string{semconv.LLMInvocationParameters},
			present: []string{semconv.LLMModelName, semconv.InputValue},
		},
		{
			name:    "HideLLMTools",
			cfg:     instrumentation.TraceConfig{HideLLMTools: true},
			absent:  []string{semconv.LLMToolKey(0)},
			present: []string{semconv.LLMInputMessageRoleKey(0), semconv.InputValue},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			server := serveJSON(t, http.StatusOK, textResponse, nil)
			recorder := tracetest.NewSpanRecorder()
			tp := trace.NewTracerProvider(trace.WithSpanProcessor(recorder))
			client := newClient(t, server.URL, tp, openaiotel.WithTraceConfig(tc.cfg))

			_, err := client.Responses.New(context.Background(), responses.ResponseNewParams{
				Model: "gpt-6.1-sol",
				Input: responses.ResponseNewParamsInputUnion{OfString: openaisdk.String("weather?")},
				Tools: []responses.ToolUnionParam{responses.ToolParamOfFunction("get_weather", nil, false)},
			})
			if err != nil {
				t.Fatalf("Responses.New: %v", err)
			}
			_ = tp.ForceFlush(context.Background())

			attrs := attrMap(recorder.Ended()[0].Attributes())
			for _, k := range tc.absent {
				if v, present := attrs[k]; present {
					t.Errorf("%s should be hidden, got %v", k, v)
				}
			}
			for _, k := range tc.present {
				if _, present := attrs[k]; !present {
					t.Errorf("%s should still be set", k)
				}
			}
		})
	}
}

func TestChatCompletions_SpanNameUnchanged(t *testing.T) {
	server := serveJSON(t, http.StatusOK, okResponse, nil)

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
	if got := recorder.Ended()[0].Name(); got != "openai.chat.completions.create" {
		t.Errorf("chat span name: got %q", got)
	}
}

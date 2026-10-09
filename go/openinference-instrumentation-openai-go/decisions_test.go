package openai_test

import (
	"context"
	"errors"
	"io"
	"net/http"
	"reflect"
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

const decisionsSpanName = "openai.decisions.create"

// decisionRequest asks one question of each type. Bodies are sent as
// raw JSON because openai-go v1 has no Decisions client; v3.73.0 added
// client.Decisions.New, which sends the same request.
const decisionRequest = `{"model":"gpt-6-luna","input":"I was charged twice and the box arrived crushed.","questions":[` +
	`{"type":"predicate","name":"is_complaint","instructions":"Is this a complaint?"},` +
	`{"type":"choice","name":"department","instructions":"Which department should handle this?","choices":[{"value":"billing","description":"Payments"},{"value":"shipping","description":"Delivery"}]},` +
	`{"type":"score","name":"severity","instructions":"How severe is the issue?","levels":[{"label":"low"},{"label":"medium"},{"label":"high"}]},` +
	`{"type":"predicate","name":"is_legal_threat","instructions":"Does the customer threaten legal action?"}]}`

// decisionResponse answers each question in decisionRequest, ending with
// a refusal. The shape is the live gpt-6-luna response's.
const decisionResponse = `{"model":"gpt-6-luna","answers":[` +
	`{"type":"predicate","name":"is_complaint","probability":1.0},` +
	`{"type":"choice","name":"department","choice":"billing","probabilities":[{"value":"billing","probability":0.95},{"value":"shipping","probability":0.05}],"confidence":0.9},` +
	`{"type":"score","name":"severity","score":1.25,"probabilities":[{"value":0,"label":"low","probability":0.0},{"value":1,"label":"medium","probability":0.75},{"value":2,"label":"high","probability":0.25}],"confidence":0.63},` +
	`{"type":"refusal","name":"is_legal_threat"}],` +
	`"usage":{"input_tokens":403,"input_tokens_details":{"cached_tokens":0,"cache_write_tokens":0},"output_tokens":0,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":403}}`

// decisionAttrs is the full attribute set of a successful call that
// sends decisionRequest and receives decisionResponse from OpenAI.
var decisionAttrs = map[string]any{
	semconv.OpenInferenceSpanKind:     semconv.SpanKindDecision,
	semconv.DecisionSystem:            semconv.DecisionSystemOpenAI,
	semconv.DecisionProvider:          semconv.DecisionProviderOpenAI,
	semconv.DecisionRequestModelName:  "gpt-6-luna",
	semconv.DecisionResponseModelName: "gpt-6-luna",
	semconv.DecisionModelName:         "gpt-6-luna",
	semconv.DecisionTokenCountInput:   int64(403),
	semconv.DecisionTokenCountOutput:  int64(0), // a reported zero is kept
	semconv.InputValue:                decisionRequest,
	semconv.InputMimeType:             semconv.MimeTypeJSON,
	semconv.OutputValue:               decisionResponse,
	semconv.OutputMimeType:            semconv.MimeTypeJSON,
}

// withAttrs returns a copy of base with overrides applied; a nil
// override value deletes the key.
func withAttrs(base map[string]any, overrides map[string]any) map[string]any {
	out := make(map[string]any, len(base))
	for k, v := range base {
		out[k] = v
	}
	for k, v := range overrides {
		if v == nil {
			delete(out, k)
		} else {
			out[k] = v
		}
	}
	return out
}

// runDecision sends reqBody to url through the middleware, answering
// with next, and returns the single ended span.
func runDecision(t *testing.T, url, reqBody string, next func(*http.Request) (*http.Response, error), opts ...openaiotel.Option) trace.ReadOnlySpan {
	t.Helper()
	recorder := tracetest.NewSpanRecorder()
	tp := trace.NewTracerProvider(trace.WithSpanProcessor(recorder))
	mw := openaiotel.Middleware(tp.Tracer("test"), opts...)

	req, err := http.NewRequest(http.MethodPost, url, strings.NewReader(reqBody))
	if err != nil {
		t.Fatalf("NewRequest: %v", err)
	}
	resp, _ := mw(req, next)
	if resp != nil {
		_, _ = io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
	}
	_ = tp.ForceFlush(context.Background())

	spans := recorder.Ended()
	if len(spans) != 1 {
		t.Fatalf("expected 1 span, got %d", len(spans))
	}
	return spans[0]
}

// respondWith returns a next func that answers with status and body.
func respondWith(status int, body string) func(*http.Request) (*http.Response, error) {
	return func(*http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: status,
			Header:     http.Header{"Content-Type": []string{"application/json"}},
			Body:       io.NopCloser(strings.NewReader(body)),
		}, nil
	}
}

func TestDecisions_AllQuestionTypesAndRefusal(t *testing.T) {
	var upstreamReq []byte
	server := serveJSON(t, http.StatusOK, decisionResponse, &upstreamReq)

	recorder := tracetest.NewSpanRecorder()
	tp := trace.NewTracerProvider(trace.WithSpanProcessor(recorder))
	client := newClient(t, server.URL, tp)

	// openai-go v1 has no Decisions client, but the generic Post goes
	// through the same middleware chain.
	var res struct {
		Answers []struct {
			Type string `json:"type"`
			Name string `json:"name"`
		} `json:"answers"`
	}
	if err := client.Post(context.Background(), "decisions", []byte(decisionRequest), &res); err != nil {
		t.Fatalf("Post decisions: %v", err)
	}
	if len(res.Answers) != 4 || res.Answers[3].Type != "refusal" {
		t.Fatalf("unexpected answers: %+v", res.Answers)
	}
	if string(upstreamReq) != decisionRequest {
		t.Fatalf("request body changed in transit:\ngot  %s\nwant %s", upstreamReq, decisionRequest)
	}

	_ = tp.ForceFlush(context.Background())
	spans := recorder.Ended()
	if len(spans) != 1 {
		t.Fatalf("expected 1 span, got %d", len(spans))
	}
	if got := spans[0].Name(); got != decisionsSpanName {
		t.Errorf("span name: got %q want %q", got, decisionsSpanName)
	}
	if got := spans[0].Status().Code; got != codes.Ok {
		t.Errorf("span status: got %s want Ok", got)
	}
	// Exact set: no llm.* model, provider, system or token attributes.
	assertExactAttrs(t, attrMap(spans[0].Attributes()), decisionAttrs)
}

func TestDecisions_ResponseModelOverridesRequestModel(t *testing.T) {
	// The response names the model that answered; decision.model_name
	// takes it. With no usage block, no token counts are recorded.
	const respBody = `{"model":"gpt-6-luna-2026-10-01","answers":[{"type":"predicate","name":"q","probability":0.4}]}`
	const reqBody = `{"model":"gpt-6-luna","input":"x","questions":[{"type":"predicate","name":"q","instructions":"?"}]}`
	span := runDecision(t, "https://api.openai.com/v1/decisions", reqBody, respondWith(http.StatusOK, respBody))

	assertExactAttrs(t, attrMap(span.Attributes()), map[string]any{
		semconv.OpenInferenceSpanKind:     semconv.SpanKindDecision,
		semconv.DecisionSystem:            semconv.DecisionSystemOpenAI,
		semconv.DecisionProvider:          semconv.DecisionProviderOpenAI,
		semconv.DecisionRequestModelName:  "gpt-6-luna",
		semconv.DecisionResponseModelName: "gpt-6-luna-2026-10-01",
		semconv.DecisionModelName:         "gpt-6-luna-2026-10-01",
		semconv.InputValue:                reqBody,
		semconv.InputMimeType:             semconv.MimeTypeJSON,
		semconv.OutputValue:               respBody,
		semconv.OutputMimeType:            semconv.MimeTypeJSON,
	})
}

func TestDecisions_ProviderForHost(t *testing.T) {
	cases := []struct {
		url      string
		provider string
	}{
		{"https://api.openai.com/v1/decisions", semconv.DecisionProviderOpenAI},
		{"https://my-resource.openai.azure.com/openai/v1/decisions", semconv.LLMProviderAzure},
		{"https://my-resource.services.ai.azure.com/openai/v1/decisions", semconv.LLMProviderAzure},
		{"https://my-resource.cognitiveservices.azure.com:443/openai/v1/decisions", semconv.LLMProviderAzure},
		// Any other host (a gateway, a local proxy) is treated as OpenAI,
		// as it is for Chat Completions and Responses.
		{"https://llm-gateway.example.com/v1/decisions", semconv.DecisionProviderOpenAI},
		{"http://localhost:8080/v1/decisions", semconv.DecisionProviderOpenAI},
	}
	for _, tc := range cases {
		t.Run(tc.url, func(t *testing.T) {
			span := runDecision(t, tc.url, decisionRequest, respondWith(http.StatusOK, decisionResponse))
			if got := span.Name(); got != decisionsSpanName {
				t.Errorf("span name: got %q", got)
			}
			assertExactAttrs(t, attrMap(span.Attributes()), withAttrs(decisionAttrs, map[string]any{
				semconv.DecisionProvider: tc.provider,
			}))
		})
	}
}

func TestDecisions_ErrorResponse(t *testing.T) {
	var upstreamReq []byte
	server := serveJSON(t, http.StatusBadRequest,
		`{"error":{"message":"Missing required parameter: 'questions[0].levels[0].label'.","type":"invalid_request_error","param":"questions[0].levels[0].label","code":"missing_required_parameter"}}`, &upstreamReq)

	recorder := tracetest.NewSpanRecorder()
	tp := trace.NewTracerProvider(trace.WithSpanProcessor(recorder))
	client := newClient(t, server.URL, tp)

	if err := client.Post(context.Background(), "decisions", []byte(decisionRequest), nil, option.WithMaxRetries(0)); err == nil {
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
	// Request-side attributes only: the requested model stays as
	// decision.model_name, and nothing is parsed out of the error body.
	assertExactAttrs(t, attrMap(spans[0].Attributes()), map[string]any{
		semconv.OpenInferenceSpanKind:    semconv.SpanKindDecision,
		semconv.DecisionSystem:           semconv.DecisionSystemOpenAI,
		semconv.DecisionProvider:         semconv.DecisionProviderOpenAI,
		semconv.DecisionRequestModelName: "gpt-6-luna",
		semconv.DecisionModelName:        "gpt-6-luna",
		semconv.InputValue:               string(upstreamReq),
		semconv.InputMimeType:            semconv.MimeTypeJSON,
	})
}

func TestDecisions_MalformedResponseBody(t *testing.T) {
	span := runDecision(t, "https://api.openai.com/v1/decisions", decisionRequest,
		respondWith(http.StatusOK, `{"model":"gpt-6-luna","answers":[`))

	// The SDK fails to decode the body and returns the error to the
	// caller; as for Responses, the span records the parse error as an
	// event and is not marked OK.
	if got := span.Status().Code; got != codes.Unset {
		t.Errorf("span status: got %s want Unset", got)
	}

	var foundException bool
	for _, e := range span.Events() {
		if e.Name == "exception" {
			foundException = true
		}
	}
	if !foundException {
		t.Errorf("expected an exception event for the unparseable body, got %+v", span.Events())
	}
	assertExactAttrs(t, attrMap(span.Attributes()), map[string]any{
		semconv.OpenInferenceSpanKind:    semconv.SpanKindDecision,
		semconv.DecisionSystem:           semconv.DecisionSystemOpenAI,
		semconv.DecisionProvider:         semconv.DecisionProviderOpenAI,
		semconv.DecisionRequestModelName: "gpt-6-luna",
		semconv.DecisionModelName:        "gpt-6-luna",
		semconv.InputValue:               decisionRequest,
		semconv.InputMimeType:            semconv.MimeTypeJSON,
	})
}

func TestDecisions_TransportErrorRecordsException(t *testing.T) {
	span := runDecision(t, "https://api.openai.com/v1/decisions", decisionRequest,
		func(*http.Request) (*http.Response, error) { return nil, errors.New("connection reset") })

	if got := span.Status(); got.Code != codes.Error || got.Description != "connection reset" {
		t.Errorf("span status: got %+v", got)
	}
	var foundException bool
	for _, e := range span.Events() {
		if e.Name == "exception" {
			foundException = true
		}
	}
	if !foundException {
		t.Errorf("expected exception event, got %+v", span.Events())
	}
	assertExactAttrs(t, attrMap(span.Attributes()), map[string]any{
		semconv.OpenInferenceSpanKind:    semconv.SpanKindDecision,
		semconv.DecisionSystem:           semconv.DecisionSystemOpenAI,
		semconv.DecisionProvider:         semconv.DecisionProviderOpenAI,
		semconv.DecisionRequestModelName: "gpt-6-luna",
		semconv.DecisionModelName:        "gpt-6-luna",
		semconv.InputValue:               decisionRequest,
		semconv.InputMimeType:            semconv.MimeTypeJSON,
	})
}

func TestDecisions_NonPostPassesThrough(t *testing.T) {
	recorder := tracetest.NewSpanRecorder()
	tp := trace.NewTracerProvider(trace.WithSpanProcessor(recorder))
	mw := openaiotel.Middleware(tp.Tracer("test"))

	req, err := http.NewRequest(http.MethodGet, "https://api.openai.com/v1/decisions", nil)
	if err != nil {
		t.Fatalf("NewRequest: %v", err)
	}
	if _, err := mw(req, respondWith(http.StatusOK, `{}`)); err != nil {
		t.Fatalf("middleware: %v", err)
	}
	_ = tp.ForceFlush(context.Background())
	if got := len(recorder.Ended()); got != 0 {
		t.Errorf("GET /v1/decisions should not produce a span, got %d", got)
	}
}

func TestDecisions_HideInputsAndOutputsFromEnv(t *testing.T) {
	t.Setenv(instrumentation.EnvHideInputs, "true")
	t.Setenv(instrumentation.EnvHideOutputs, "true")

	span := runDecision(t, "https://api.openai.com/v1/decisions", decisionRequest, respondWith(http.StatusOK, decisionResponse))

	// Values are redacted and the mime types dropped; the model and
	// token counts are never hidden.
	assertExactAttrs(t, attrMap(span.Attributes()), withAttrs(decisionAttrs, map[string]any{
		semconv.InputValue:     instrumentation.RedactedValue,
		semconv.InputMimeType:  nil,
		semconv.OutputValue:    instrumentation.RedactedValue,
		semconv.OutputMimeType: nil,
	}))
}

func TestDecisions_HideFlags(t *testing.T) {
	cases := []struct {
		name string
		cfg  instrumentation.TraceConfig
		want map[string]any
	}{
		{
			name: "HideInputs",
			cfg:  instrumentation.TraceConfig{HideInputs: true},
			want: withAttrs(decisionAttrs, map[string]any{
				semconv.InputValue:    instrumentation.RedactedValue,
				semconv.InputMimeType: nil,
			}),
		},
		{
			name: "HideOutputs",
			cfg:  instrumentation.TraceConfig{HideOutputs: true},
			want: withAttrs(decisionAttrs, map[string]any{
				semconv.OutputValue:    instrumentation.RedactedValue,
				semconv.OutputMimeType: nil,
			}),
		},
		// The message, text, tool, prompt and invocation-parameter flags
		// gate llm.* attributes a decision span does not have, and
		// HideInputImages has no images to redact here.
		{name: "HideInputMessages", cfg: instrumentation.TraceConfig{HideInputMessages: true}, want: decisionAttrs},
		{name: "HideOutputMessages", cfg: instrumentation.TraceConfig{HideOutputMessages: true}, want: decisionAttrs},
		{name: "HideInputText", cfg: instrumentation.TraceConfig{HideInputText: true}, want: decisionAttrs},
		{name: "HideOutputText", cfg: instrumentation.TraceConfig{HideOutputText: true}, want: decisionAttrs},
		{name: "HideLLMInvocationParameters", cfg: instrumentation.TraceConfig{HideLLMInvocationParameters: true}, want: decisionAttrs},
		{name: "HideLLMTools", cfg: instrumentation.TraceConfig{HideLLMTools: true}, want: decisionAttrs},
		{name: "HidePrompts", cfg: instrumentation.TraceConfig{HidePrompts: true}, want: decisionAttrs},
		{name: "HideInputImages", cfg: instrumentation.TraceConfig{HideInputImages: true}, want: decisionAttrs},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			span := runDecision(t, "https://api.openai.com/v1/decisions", decisionRequest,
				respondWith(http.StatusOK, decisionResponse), openaiotel.WithTraceConfig(tc.cfg))
			assertExactAttrs(t, attrMap(span.Attributes()), tc.want)
		})
	}
}

func TestDecisions_HideInputImagesRedactsImageURLs(t *testing.T) {
	const reqBody = `{"model":"gpt-6-luna","input":[{"role":"user","content":[` +
		`{"type":"input_text","text":"Inspect the product in this photo."},` +
		`{"type":"input_image","image_url":"data:image/png;base64,iVBORw0KGgo="}]}],` +
		`"questions":[{"type":"predicate","name":"visible_damage","instructions":"Is the product damaged?","choices":[{"value":12345678901234567890}]}]}`
	// Keys are re-encoded in sorted order; text, other fields and the
	// exact numeric literal are kept.
	const redacted = `{"input":[{"content":[` +
		`{"text":"Inspect the product in this photo.","type":"input_text"},` +
		`{"image_url":"__REDACTED__","type":"input_image"}],"role":"user"}],` +
		`"model":"gpt-6-luna","questions":[{"choices":[{"value":12345678901234567890}],"instructions":"Is the product damaged?","name":"visible_damage","type":"predicate"}]}`

	for _, tc := range []struct {
		name  string
		cfg   instrumentation.TraceConfig
		input string
	}{
		{"HideInputImages", instrumentation.TraceConfig{HideInputImages: true}, redacted},
		{"NoFlags", instrumentation.TraceConfig{}, reqBody},
		{"HideInputs", instrumentation.TraceConfig{HideInputs: true, HideInputImages: true}, instrumentation.RedactedValue},
	} {
		t.Run(tc.name, func(t *testing.T) {
			span := runDecision(t, "https://api.openai.com/v1/decisions", reqBody,
				respondWith(http.StatusOK, decisionResponse), openaiotel.WithTraceConfig(tc.cfg))
			if got := attrMap(span.Attributes())[semconv.InputValue]; got != tc.input {
				t.Errorf("input.value:\ngot  %v\nwant %v", got, tc.input)
			}
		})
	}
}

func TestDecisions_OversizedBase64ImageRedactedWithoutFlags(t *testing.T) {
	// Base64 data URIs over Python's default 32,000-character limit are
	// redacted from input.value even with no hide flags, as for Responses.
	large := "data:image/png;base64," + strings.Repeat("A", 32_001)
	reqBody := `{"model":"gpt-6-luna","input":[{"role":"user","content":[{"type":"input_image","image_url":"` + large + `"}]}],` +
		`"questions":[{"type":"predicate","name":"q","instructions":"?"}]}`
	const want = `{"input":[{"content":[{"image_url":"__REDACTED__","type":"input_image"}],"role":"user"}],` +
		`"model":"gpt-6-luna","questions":[{"instructions":"?","name":"q","type":"predicate"}]}`

	span := runDecision(t, "https://api.openai.com/v1/decisions", reqBody,
		respondWith(http.StatusOK, decisionResponse), openaiotel.WithTraceConfig(instrumentation.TraceConfig{}))
	if got := attrMap(span.Attributes())[semconv.InputValue]; got != want {
		t.Errorf("input.value:\ngot  %.200v\nwant %v", got, want)
	}
}

func TestDecisions_SuppressedContextEmitsNoSpan(t *testing.T) {
	server := serveJSON(t, http.StatusOK, decisionResponse, nil)

	recorder := tracetest.NewSpanRecorder()
	tp := trace.NewTracerProvider(trace.WithSpanProcessor(recorder))
	client := newClient(t, server.URL, tp)

	ctx := instrumentation.WithSuppression(context.Background())
	if err := client.Post(ctx, "decisions", []byte(decisionRequest), nil); err != nil {
		t.Fatalf("Post decisions: %v", err)
	}
	_ = tp.ForceFlush(context.Background())
	if got := len(recorder.Ended()); got != 0 {
		t.Errorf("suppressed context should produce no span, got %d", got)
	}
}

func TestDecisions_ContextAttributesPropagateToSpan(t *testing.T) {
	server := serveJSON(t, http.StatusOK, decisionResponse, nil)

	recorder := tracetest.NewSpanRecorder()
	tp := trace.NewTracerProvider(trace.WithSpanProcessor(recorder))
	client := newClient(t, server.URL, tp)

	ctx := context.Background()
	ctx = instrumentation.WithSession(ctx, "session-abc")
	ctx = instrumentation.WithUser(ctx, "user-xyz")
	ctx = instrumentation.WithMetadata(ctx, `{"team":"platform"}`)
	ctx = instrumentation.WithTags(ctx, "prod", "canary")

	if err := client.Post(ctx, "decisions", []byte(decisionRequest), nil); err != nil {
		t.Fatalf("Post decisions: %v", err)
	}
	_ = tp.ForceFlush(context.Background())

	assertExactAttrs(t, attrMap(recorder.Ended()[0].Attributes()), withAttrs(decisionAttrs, map[string]any{
		"session.id": "session-abc",
		"user.id":    "user-xyz",
		"metadata":   `{"team":"platform"}`,
		"tag.tags":   []string{"prod", "canary"},
	}))
}

func TestDecisions_ChatAndResponsesStayLLMSpans(t *testing.T) {
	for _, tc := range []struct {
		name     string
		spanName string
		body     string
		call     func(client openaisdk.Client) error
	}{
		{
			name:     "chat",
			spanName: "openai.chat.completions.create",
			body:     okResponse,
			call: func(client openaisdk.Client) error {
				_, err := client.Chat.Completions.New(context.Background(), openaisdk.ChatCompletionNewParams{
					Model:    shared.ChatModelGPT4o,
					Messages: []openaisdk.ChatCompletionMessageParamUnion{openaisdk.UserMessage("hi")},
				})
				return err
			},
		},
		{
			name:     "responses",
			spanName: responsesSpanName,
			body:     textResponse,
			call: func(client openaisdk.Client) error {
				_, err := client.Responses.New(context.Background(), responses.ResponseNewParams{
					Model: "gpt-6.1-sol",
					Input: responses.ResponseNewParamsInputUnion{OfString: openaisdk.String("hi")},
				})
				return err
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			server := serveJSON(t, http.StatusOK, tc.body, nil)
			recorder := tracetest.NewSpanRecorder()
			tp := trace.NewTracerProvider(trace.WithSpanProcessor(recorder))
			if err := tc.call(newClient(t, server.URL, tp)); err != nil {
				t.Fatalf("call: %v", err)
			}
			_ = tp.ForceFlush(context.Background())

			span := recorder.Ended()[0]
			if got := span.Name(); got != tc.spanName {
				t.Errorf("span name: got %q want %q", got, tc.spanName)
			}
			attrs := attrMap(span.Attributes())
			want := map[string]any{
				semconv.OpenInferenceSpanKind: semconv.SpanKindLLM,
				semconv.LLMSystem:             semconv.LLMSystemOpenAI,
				semconv.LLMProvider:           semconv.LLMProviderOpenAI,
			}
			for k, v := range want {
				if got := attrs[k]; !reflect.DeepEqual(got, v) {
					t.Errorf("%s: got %v want %v", k, got, v)
				}
			}
			for k := range attrs {
				if strings.HasPrefix(k, "decision.") {
					t.Errorf("LLM span has decision attribute %s", k)
				}
			}
		})
	}
}

// Package openai instruments calls made through sashabaranov/go-openai
// (the most common Go OpenAI client) with OpenInference LLM spans.
//
// Wire it up by replacing the client's HTTPClient with one whose Transport
// is wrapped by NewTransport:
//
//	import (
//	    "net/http"
//
//	    "github.com/sashabaranov/go-openai"
//	    "go.opentelemetry.io/otel"
//
//	    openaiotel "github.com/Arize-ai/openinference/go/instrumentation/openai"
//	)
//
//	cfg := openai.DefaultConfig(apiKey)
//	cfg.HTTPClient = &http.Client{
//	    Transport: openaiotel.NewTransport(http.DefaultTransport, otel.Tracer("my-app")),
//	}
//	client := openai.NewClientWithConfig(cfg)
//
// Every /v1/chat/completions call now emits an LLM-kind span. Streaming
// responses (text/event-stream) pass through unchanged; their spans carry
// request attributes only.
package openai

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"

	"github.com/Arize-ai/openinference/go/instrumentation"
	"github.com/Arize-ai/openinference/go/instrumentation/internal/httputil"
	"github.com/Arize-ai/openinference/go/semconv"
)

// Option configures the Transport at construction time.
type Option func(*transport)

// WithTraceConfig overrides the OpenInference TraceConfig the
// transport uses to gate sensitive-data attributes (input.value,
// output.value, message contents, invocation parameters, tool
// definitions). The default is instrumentation.TraceConfigFromEnv(),
// so customers who set OPENINFERENCE_HIDE_* env vars get masking
// without touching code.
func WithTraceConfig(cfg instrumentation.TraceConfig) Option {
	return func(t *transport) { t.config = cfg }
}

// NewTransport wraps base with an http.RoundTripper that emits an
// OpenInference LLM span for every /v1/chat/completions request. If
// tracer is nil, NewTransport returns base unchanged.
func NewTransport(base http.RoundTripper, tracer trace.Tracer, opts ...Option) http.RoundTripper {
	if tracer == nil {
		return base
	}
	if base == nil {
		base = http.DefaultTransport
	}
	t := &transport{
		base:   base,
		tracer: tracer,
		config: instrumentation.TraceConfigFromEnv(),
	}
	for _, o := range opts {
		o(t)
	}
	return t
}

type transport struct {
	base   http.RoundTripper
	tracer trace.Tracer
	config instrumentation.TraceConfig
}

func (t *transport) RoundTrip(req *http.Request) (*http.Response, error) {
	if !isChatCompletion(req) {
		return t.base.RoundTrip(req)
	}
	// Suppression: customer marked this context as off-limits for
	// tracing (evaluator code, etc.). Pass the request through with
	// no span at all.
	if instrumentation.IsSuppressed(req.Context()) {
		return t.base.RoundTrip(req)
	}

	ctx, span := t.tracer.Start(req.Context(), "openai.chat.completions.create")
	req = req.WithContext(ctx)

	span.SetAttributes(
		attribute.String(semconv.OpenInferenceSpanKind, semconv.SpanKindLLM),
		attribute.String(semconv.LLMSystem, semconv.LLMSystemOpenAI),
		attribute.String(semconv.LLMProvider, semconv.LLMProviderOpenAI),
	)
	// Propagate session.id / user.id / metadata / tag.tags from
	// baggage so the LLM span carries the customer's context
	// without them having to set the attributes manually.
	instrumentation.ApplyContextAttributes(ctx, span)

	reqBody, err := httputil.ReadAndRestore(&req.Body)
	if err != nil {
		wrapped := fmt.Errorf("openai otel transport: read request body: %w", err)
		span.RecordError(wrapped)
		span.SetStatus(codes.Error, "request body read failed")
		span.End()
		return nil, wrapped
	}
	if reqBody != nil {
		t.setRequestAttrs(span, reqBody)
	}

	resp, err := t.base.RoundTrip(req)
	if err != nil {
		span.RecordError(err)
		span.SetStatus(codes.Error, err.Error())
		span.End()
		return resp, err
	}
	if resp == nil {
		span.End()
		return resp, nil
	}

	// SSE streaming responses: defer span.End() to whenever the caller
	// is done reading the body, so the span's duration reflects the
	// actual time-to-last-token. Reading or buffering the SSE bytes
	// here would break the caller's stream consumer.
	if httputil.IsStreaming(resp) {
		resp.Body = &httputil.SpanEndingBody{ReadCloser: resp.Body, Span: span}
		return resp, nil
	}

	respBody, readErr := httputil.ReadAndRestore(&resp.Body)
	if readErr != nil {
		span.RecordError(fmt.Errorf("read response body: %w", readErr))
	} else if respBody != nil {
		t.setResponseAttrs(span, respBody, resp.StatusCode)
	}
	span.End()
	return resp, nil
}

func isChatCompletion(req *http.Request) bool {
	return req.Method == http.MethodPost && strings.HasSuffix(req.URL.Path, "/chat/completions")
}

type requestPayload struct {
	Model       string           `json:"model"`
	Temperature *float64         `json:"temperature,omitempty"`
	TopP        *float64         `json:"top_p,omitempty"`
	MaxTokens   *int64           `json:"max_tokens,omitempty"`
	N           *int64           `json:"n,omitempty"`
	Stream      bool             `json:"stream,omitempty"`
	Messages    []requestMessage `json:"messages"`
	Tools       []map[string]any `json:"tools,omitempty"`
	ToolChoice  json.RawMessage  `json:"tool_choice,omitempty"`
}

type requestMessage struct {
	Role       string            `json:"role"`
	Name       string            `json:"name,omitempty"`
	Content    json.RawMessage   `json:"content,omitempty"`
	ToolCallID string            `json:"tool_call_id,omitempty"`
	ToolCalls  []requestToolCall `json:"tool_calls,omitempty"`
}

type requestToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

func (t *transport) setRequestAttrs(span trace.Span, body []byte) {
	var p requestPayload
	if err := json.Unmarshal(body, &p); err != nil {
		return
	}

	if p.Model != "" {
		span.SetAttributes(attribute.String(semconv.LLMModelName, p.Model))
	}
	if !t.config.HideLLMInvocationParameters {
		if invocation := invocationParamsJSON(p); invocation != "" {
			span.SetAttributes(attribute.String(semconv.LLMInvocationParameters, invocation))
		}
	}

	// Input messages: drop the entire llm.input_messages.* structure
	// (including nested tool_calls / name / tool_call_id) when either
	// HideInputs or HideInputMessages is set, matching Python's
	// TraceConfig.mask which returns None for the whole
	// LLM_INPUT_MESSAGES key family. HideInputText (alone) keeps the
	// structure but redacts text content.
	if !t.config.HideInputs && !t.config.HideInputMessages {
		for i, msg := range p.Messages {
			span.SetAttributes(attribute.String(semconv.LLMInputMessageRoleKey(i), msg.Role))
			if text := extractText(msg.Content); text != "" {
				if t.config.HideInputText {
					span.SetAttributes(attribute.String(semconv.LLMInputMessageContentKey(i), instrumentation.RedactedValue))
				} else {
					span.SetAttributes(attribute.String(semconv.LLMInputMessageContentKey(i), text))
				}
			}
			if msg.Name != "" {
				span.SetAttributes(attribute.String(semconv.LLMInputMessageNameKey(i), msg.Name))
			}
			if msg.ToolCallID != "" {
				span.SetAttributes(attribute.String(semconv.LLMInputMessageToolCallIDKey(i), msg.ToolCallID))
			}
			for j, tc := range msg.ToolCalls {
				span.SetAttributes(
					attribute.String(semconv.LLMInputMessageToolCallKey(i, j, semconv.ToolCallID), tc.ID),
					attribute.String(semconv.LLMInputMessageToolCallKey(i, j, semconv.ToolCallFunctionName), tc.Function.Name),
					attribute.String(semconv.LLMInputMessageToolCallKey(i, j, semconv.ToolCallFunctionArgumentsJSON), tc.Function.Arguments),
				)
			}
		}
	}

	if lastUser := lastUserText(p.Messages); lastUser != "" {
		span.SetAttributes(attribute.String(semconv.InputValue, t.config.MaskInputValue(lastUser)))
	}

	// Tools: dropped under either HideInputs (matches Python — the
	// tool list is considered part of the input) or the more granular
	// HideLLMTools.
	if !t.config.HideInputs && !t.config.HideLLMTools {
		for i, tool := range p.Tools {
			raw, err := json.Marshal(tool)
			if err == nil {
				span.SetAttributes(attribute.String(semconv.LLMToolKey(i), string(raw)))
			}
		}
	}
}

func invocationParamsJSON(p requestPayload) string {
	params := map[string]any{}
	if p.Temperature != nil {
		params["temperature"] = *p.Temperature
	}
	if p.TopP != nil {
		params["top_p"] = *p.TopP
	}
	if p.MaxTokens != nil {
		params["max_tokens"] = *p.MaxTokens
	}
	if p.N != nil {
		params["n"] = *p.N
	}
	if len(params) == 0 {
		return ""
	}
	out, err := json.Marshal(params)
	if err != nil {
		return ""
	}
	return string(out)
}

func lastUserText(msgs []requestMessage) string {
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role == "user" {
			return extractText(msgs[i].Content)
		}
	}
	return ""
}

// extractText handles the two shapes of an OpenAI message content field:
// a bare string, or an array of {type,text,...} parts (vision / tool
// inputs). Non-text parts are skipped — model image/audio inputs need
// dedicated handling not yet covered here.
func extractText(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s
	}
	var arr []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(raw, &arr); err == nil {
		var parts []string
		for _, b := range arr {
			if b.Type == "text" && b.Text != "" {
				parts = append(parts, b.Text)
			}
		}
		return strings.Join(parts, "\n")
	}
	return ""
}

type responsePayload struct {
	ID      string           `json:"id"`
	Model   string           `json:"model"`
	Choices []responseChoice `json:"choices"`
	Usage   *responseUsage   `json:"usage,omitempty"`
}

type responseChoice struct {
	Index        int             `json:"index"`
	FinishReason string          `json:"finish_reason"`
	Message      responseMessage `json:"message"`
}

type responseMessage struct {
	Role      string             `json:"role"`
	Content   string             `json:"content"`
	ToolCalls []responseToolCall `json:"tool_calls,omitempty"`
}

type responseToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type responseUsage struct {
	PromptTokens     int64 `json:"prompt_tokens"`
	CompletionTokens int64 `json:"completion_tokens"`
	TotalTokens      int64 `json:"total_tokens"`
	// OpenAI nested details for prompt cache + reasoning. Only set when
	// the model is the type that returns them (e.g. o1, gpt-4o, etc.).
	PromptTokensDetails *struct {
		CachedTokens int64 `json:"cached_tokens"`
		AudioTokens  int64 `json:"audio_tokens"`
	} `json:"prompt_tokens_details,omitempty"`
	CompletionTokensDetails *struct {
		ReasoningTokens int64 `json:"reasoning_tokens"`
		AudioTokens     int64 `json:"audio_tokens"`
	} `json:"completion_tokens_details,omitempty"`
}

func (t *transport) setResponseAttrs(span trace.Span, body []byte, statusCode int) {
	if statusCode < 200 || statusCode >= 300 {
		span.SetStatus(codes.Error, http.StatusText(statusCode))
		// Do not parse error bodies as success payloads; OpenAI returns
		// {"error":{...}} which unmarshals into responsePayload with zero
		// token counts and would pollute the span with misleading "0
		// tokens" attributes.
		return
	}

	var r responsePayload
	if err := json.Unmarshal(body, &r); err != nil {
		span.RecordError(fmt.Errorf("parse response body: %w", err))
		return
	}

	if r.Model != "" {
		span.SetAttributes(attribute.String(semconv.LLMModelName, r.Model))
	}

	// Output messages: drop the entire llm.output_messages.* structure
	// (including nested tool_calls) when either HideOutputs or
	// HideOutputMessages is set, matching Python. HideOutputText alone
	// keeps the structure but redacts text content.
	if !t.config.HideOutputs && !t.config.HideOutputMessages {
		for i, c := range r.Choices {
			span.SetAttributes(attribute.String(semconv.LLMOutputMessageRoleKey(i), c.Message.Role))
			if c.Message.Content != "" {
				if t.config.HideOutputText {
					span.SetAttributes(attribute.String(semconv.LLMOutputMessageContentKey(i), instrumentation.RedactedValue))
				} else {
					span.SetAttributes(attribute.String(semconv.LLMOutputMessageContentKey(i), c.Message.Content))
				}
			}
			for j, tc := range c.Message.ToolCalls {
				span.SetAttributes(
					attribute.String(semconv.LLMOutputMessageToolCallKey(i, j, semconv.ToolCallID), tc.ID),
					attribute.String(semconv.LLMOutputMessageToolCallKey(i, j, semconv.ToolCallFunctionName), tc.Function.Name),
					attribute.String(semconv.LLMOutputMessageToolCallKey(i, j, semconv.ToolCallFunctionArgumentsJSON), tc.Function.Arguments),
				)
			}
			if i == 0 && c.FinishReason != "" {
				span.SetAttributes(attribute.String(semconv.LLMFinishReason, c.FinishReason))
			}
		}
	} else {
		// Structural messages hidden, but llm.finish_reason is metadata
		// not content — keep it for the first choice.
		if len(r.Choices) > 0 && r.Choices[0].FinishReason != "" {
			span.SetAttributes(attribute.String(semconv.LLMFinishReason, r.Choices[0].FinishReason))
		}
	}

	if len(r.Choices) > 0 && r.Choices[0].Message.Content != "" {
		span.SetAttributes(attribute.String(semconv.OutputValue, t.config.MaskOutputValue(r.Choices[0].Message.Content)))
	}

	if r.Usage != nil {
		span.SetAttributes(
			attribute.Int64(semconv.LLMTokenCountPrompt, r.Usage.PromptTokens),
			attribute.Int64(semconv.LLMTokenCountCompletion, r.Usage.CompletionTokens),
			attribute.Int64(semconv.LLMTokenCountTotal, r.Usage.TotalTokens),
		)
		if r.Usage.PromptTokensDetails != nil {
			if r.Usage.PromptTokensDetails.CachedTokens > 0 {
				span.SetAttributes(attribute.Int64(semconv.LLMTokenCountPromptDetailsCacheRead, r.Usage.PromptTokensDetails.CachedTokens))
			}
			if r.Usage.PromptTokensDetails.AudioTokens > 0 {
				span.SetAttributes(attribute.Int64(semconv.LLMTokenCountPromptDetailsAudio, r.Usage.PromptTokensDetails.AudioTokens))
			}
		}
		if r.Usage.CompletionTokensDetails != nil {
			if r.Usage.CompletionTokensDetails.ReasoningTokens > 0 {
				span.SetAttributes(attribute.Int64(semconv.LLMTokenCountCompletionDetailsReasoning, r.Usage.CompletionTokensDetails.ReasoningTokens))
			}
			if r.Usage.CompletionTokensDetails.AudioTokens > 0 {
				span.SetAttributes(attribute.Int64(semconv.LLMTokenCountCompletionDetailsAudio, r.Usage.CompletionTokensDetails.AudioTokens))
			}
		}
	}
}

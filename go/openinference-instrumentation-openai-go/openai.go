// Package openai instruments calls made through the official
// openai/openai-go SDK with OpenInference LLM spans.
//
// Wire it up by passing Middleware as a RequestOption when constructing
// the client; every /v1/chat/completions call then emits an LLM-kind
// span:
//
//	import (
//	    "github.com/openai/openai-go"
//	    "github.com/openai/openai-go/option"
//	    "go.opentelemetry.io/otel"
//
//	    openaiotel "github.com/Arize-ai/openinference/go/openinference-instrumentation-openai-go"
//	)
//
//	client := openai.NewClient(
//	    option.WithMiddleware(openaiotel.Middleware(otel.Tracer("my-app"))),
//	)
//
// The middleware also works against Azure OpenAI: when the request host
// matches one of the Azure-hosted patterns (*.openai.azure.com,
// *.services.ai.azure.com, *.cognitiveservices.azure.com) the span's
// llm.provider attribute is set to "azure" instead of "openai".
//
// Streaming responses (text/event-stream) pass through unchanged; their
// spans carry request attributes only and end when the caller closes or
// fully reads the response body.
package openai

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/openai/openai-go/option"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"

	"github.com/Arize-ai/openinference/go/openinference-instrumentation"
	"github.com/Arize-ai/openinference/go/openinference-instrumentation-openai-go/internal/httputil"
	"github.com/Arize-ai/openinference/go/openinference-semantic-conventions"
)

// Option configures the middleware at construction time.
type Option func(*config)

// WithTraceConfig overrides the OpenInference TraceConfig the middleware
// uses to gate sensitive-data attributes (input.value, output.value,
// message contents, invocation parameters, tool definitions). The
// default is instrumentation.TraceConfigFromEnv(), so customers who set
// OPENINFERENCE_HIDE_* env vars get masking without touching code.
func WithTraceConfig(cfg instrumentation.TraceConfig) Option {
	return func(c *config) { c.trace = cfg }
}

type config struct {
	trace instrumentation.TraceConfig
}

// Middleware returns an openai-go option.Middleware that emits an
// OpenInference LLM span for every /v1/chat/completions request issued
// through the client. If tracer is nil, Middleware is a no-op
// pass-through so callers can wire it in unconditionally.
func Middleware(tracer trace.Tracer, opts ...Option) option.Middleware {
	cfg := config{trace: instrumentation.TraceConfigFromEnv()}
	for _, o := range opts {
		o(&cfg)
	}
	if tracer == nil {
		return func(req *http.Request, next option.MiddlewareNext) (*http.Response, error) {
			return next(req)
		}
	}
	m := &middleware{tracer: tracer, config: cfg.trace}
	return m.handle
}

type middleware struct {
	tracer trace.Tracer
	config instrumentation.TraceConfig
}

func (m *middleware) handle(req *http.Request, next option.MiddlewareNext) (*http.Response, error) {
	if !isChatCompletion(req) {
		return next(req)
	}
	// Suppression: customer marked this context as off-limits for
	// tracing (evaluator code, etc.). Pass the request through with
	// no span at all.
	if instrumentation.IsSuppressed(req.Context()) {
		return next(req)
	}

	ctx, span := m.tracer.Start(req.Context(), "openai.chat.completions.create")
	req = req.WithContext(ctx)

	span.SetAttributes(
		attribute.String(semconv.OpenInferenceSpanKind, semconv.SpanKindLLM),
		attribute.String(semconv.LLMSystem, semconv.LLMSystemOpenAI),
		attribute.String(semconv.LLMProvider, providerForHost(req.URL.Host)),
	)
	// Propagate session.id / user.id / metadata / tag.tags from the
	// customer's context so the LLM span carries that data without
	// them having to set the attributes manually.
	instrumentation.ApplyContextAttributes(ctx, span)

	reqBody, err := httputil.ReadAndRestore(&req.Body)
	if err != nil {
		wrapped := fmt.Errorf("openai otel middleware: read request body: %w", err)
		span.RecordError(wrapped)
		span.SetStatus(codes.Error, "request body read failed")
		span.End()
		return nil, wrapped
	}
	if reqBody != nil {
		m.setRequestAttrs(span, reqBody)
	}

	resp, err := next(req)
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
		m.setResponseAttrs(span, respBody, resp.StatusCode)
	}
	span.End()
	return resp, nil
}

func isChatCompletion(req *http.Request) bool {
	return req.Method == http.MethodPost && strings.HasSuffix(req.URL.Path, "/chat/completions")
}

// providerForHost returns the OpenInference llm.provider value for the
// given request Host. Azure OpenAI hosts are mapped to "azure"; every
// other host (including api.openai.com) is treated as "openai".
//
// Matched on a suffix basis so resource-specific subdomains
// (e.g. my-resource.openai.azure.com) and regional endpoints are
// recognised without enumerating them.
func providerForHost(host string) string {
	h := strings.ToLower(host)
	// Strip an explicit port so ":443" / ":8080" don't break suffix
	// matching.
	if idx := strings.IndexByte(h, ':'); idx >= 0 {
		h = h[:idx]
	}
	if strings.HasSuffix(h, ".openai.azure.com") ||
		strings.HasSuffix(h, ".services.ai.azure.com") ||
		strings.HasSuffix(h, ".cognitiveservices.azure.com") {
		return semconv.LLMProviderAzure
	}
	return semconv.LLMProviderOpenAI
}

type requestPayload struct {
	Model    string           `json:"model"`
	Stream   bool             `json:"stream,omitempty"`
	Messages []requestMessage `json:"messages"`
	Tools    []map[string]any `json:"tools,omitempty"`
}

type requestMessage struct {
	Role         string               `json:"role"`
	Name         string               `json:"name,omitempty"`
	Content      json.RawMessage      `json:"content,omitempty"`
	ToolCallID   string               `json:"tool_call_id,omitempty"`
	FunctionCall *requestFunctionCall `json:"function_call,omitempty"`
	ToolCalls    []requestToolCall    `json:"tool_calls,omitempty"`
}

type requestFunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type requestToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

func (m *middleware) setRequestAttrs(span trace.Span, body []byte) {
	var p requestPayload
	if err := json.Unmarshal(body, &p); err != nil {
		return
	}

	if p.Model != "" {
		span.SetAttributes(attribute.String(semconv.LLMModelName, p.Model))
	}
	if !m.config.HideLLMInvocationParameters {
		if invocation := invocationParamsJSON(body); invocation != "" {
			span.SetAttributes(attribute.String(semconv.LLMInvocationParameters, invocation))
		}
	}

	// Input messages: drop the entire llm.input_messages.* structure
	// (including nested tool_calls / name / tool_call_id) when either
	// HideInputs or HideInputMessages is set, matching Python's
	// TraceConfig.mask which returns None for the whole
	// LLM_INPUT_MESSAGES key family. HideInputText (alone) keeps the
	// structure but redacts text content.
	if !m.config.HideInputs && !m.config.HideInputMessages {
		for i, msg := range p.Messages {
			span.SetAttributes(attribute.String(semconv.LLMInputMessageRoleKey(i), msg.Role))
			if text := extractText(msg.Content); text != "" {
				if m.config.HideInputText {
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
			if msg.FunctionCall != nil {
				if msg.FunctionCall.Name != "" {
					span.SetAttributes(attribute.String(inputMessageKey(i, semconv.MessageFunctionCallName), msg.FunctionCall.Name))
				}
				if msg.FunctionCall.Arguments != "" {
					span.SetAttributes(attribute.String(inputMessageKey(i, semconv.MessageFunctionCallArgumentsJSON), msg.FunctionCall.Arguments))
				}
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
		span.SetAttributes(attribute.String(semconv.InputValue, m.config.MaskInputValue(lastUser)))
	}

	// Tools: dropped under either HideInputs (matches Python — the
	// tool list is considered part of the input) or the more granular
	// HideLLMTools.
	if !m.config.HideInputs && !m.config.HideLLMTools {
		for i, tool := range p.Tools {
			raw, err := json.Marshal(tool)
			if err == nil {
				span.SetAttributes(attribute.String(semconv.LLMToolKey(i), string(raw)))
			}
		}
	}
}

// invocationParamsJSON returns the JSON representation of every chat-
// completion request field except the ones we surface elsewhere
// (messages, functions, tools). This captures forward-compatible
// params like max_completion_tokens, reasoning_effort, response_format,
// tool_choice, stream_options, presence_penalty, frequency_penalty, n,
// seed, logit_bias, etc. without having to enumerate them — matches
// the Python instrumentor.
//
// Returns "" if the body is not a JSON object or if no params remain
// after the strip.
func invocationParamsJSON(body []byte) string {
	var raw map[string]any
	if err := json.Unmarshal(body, &raw); err != nil {
		return ""
	}
	delete(raw, "messages")
	delete(raw, "functions")
	delete(raw, "tools")
	if len(raw) == 0 {
		return ""
	}
	out, err := json.Marshal(raw)
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
	Role         string                `json:"role"`
	Content      string                `json:"content"`
	FunctionCall *responseFunctionCall `json:"function_call,omitempty"`
	ToolCalls    []responseToolCall    `json:"tool_calls,omitempty"`
}

type responseFunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
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

func (m *middleware) setResponseAttrs(span trace.Span, body []byte, statusCode int) {
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
	if !m.config.HideOutputs && !m.config.HideOutputMessages {
		for i, c := range r.Choices {
			span.SetAttributes(attribute.String(semconv.LLMOutputMessageRoleKey(i), c.Message.Role))
			if c.Message.Content != "" {
				if m.config.HideOutputText {
					span.SetAttributes(attribute.String(semconv.LLMOutputMessageContentKey(i), instrumentation.RedactedValue))
				} else {
					span.SetAttributes(attribute.String(semconv.LLMOutputMessageContentKey(i), c.Message.Content))
				}
			}
			if c.Message.FunctionCall != nil {
				if c.Message.FunctionCall.Name != "" {
					span.SetAttributes(attribute.String(outputMessageKey(i, semconv.MessageFunctionCallName), c.Message.FunctionCall.Name))
				}
				if c.Message.FunctionCall.Arguments != "" {
					span.SetAttributes(attribute.String(outputMessageKey(i, semconv.MessageFunctionCallArgumentsJSON), c.Message.FunctionCall.Arguments))
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
		span.SetAttributes(attribute.String(semconv.OutputValue, m.config.MaskOutputValue(r.Choices[0].Message.Content)))
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

func inputMessageKey(i int, child string) string {
	return fmt.Sprintf("%s.%d.%s", semconv.LLMInputMessages, i, child)
}

func outputMessageKey(i int, child string) string {
	return fmt.Sprintf("%s.%d.%s", semconv.LLMOutputMessages, i, child)
}

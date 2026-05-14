// Package anthropic instruments calls made through anthropics/anthropic-sdk-go
// with OpenInference LLM spans.
//
// Wire it up once at client construction:
//
//	import (
//	    "github.com/anthropics/anthropic-sdk-go"
//	    "github.com/anthropics/anthropic-sdk-go/option"
//	    "go.opentelemetry.io/otel"
//	    anthropicotel "github.com/Arize-ai/openinference/go/instrumentation/anthropic"
//	)
//
//	client := anthropic.NewClient(
//	    option.WithMiddleware(anthropicotel.Middleware(otel.Tracer("my-app"))),
//	)
//
//	resp, err := client.Messages.New(ctx, anthropic.MessageNewParams{ ... })
//
// Every /v1/messages call now emits an LLM-kind span with model, input
// messages, output text, and token counts. Streaming responses are
// pass-through (no body buffering) so SSE clients still work; their spans
// have request attributes only.
package anthropic

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/anthropics/anthropic-sdk-go/option"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"

	"github.com/Arize-ai/openinference/go/instrumentation"
	"github.com/Arize-ai/openinference/go/instrumentation/internal/httputil"
	"github.com/Arize-ai/openinference/go/semconv"
)

// Option configures the Middleware at construction time.
type Option func(*middleware)

// WithTraceConfig overrides the OpenInference TraceConfig the
// middleware uses to gate sensitive-data attributes (input.value,
// output.value, message contents, invocation parameters, tool
// definitions). The default is instrumentation.TraceConfigFromEnv(),
// so customers who set OPENINFERENCE_HIDE_* env vars get masking
// without touching code.
func WithTraceConfig(cfg instrumentation.TraceConfig) Option {
	return func(m *middleware) { m.config = cfg }
}

type middleware struct {
	tracer trace.Tracer
	config instrumentation.TraceConfig
}

// Middleware returns an Anthropic SDK middleware that wraps every
// /v1/messages request in an OpenInference LLM span.
func Middleware(tracer trace.Tracer, opts ...Option) option.Middleware {
	m := &middleware{
		tracer: tracer,
		config: instrumentation.TraceConfigFromEnv(),
	}
	for _, o := range opts {
		o(m)
	}
	return m.handle
}

func (m *middleware) handle(req *http.Request, next option.MiddlewareNext) (*http.Response, error) {
	if !isMessageCreate(req) {
		return next(req)
	}
	// Suppression: customer marked this context as off-limits for
	// tracing (evaluator code, etc.). Pass the request through with
	// no span at all.
	if instrumentation.IsSuppressed(req.Context()) {
		return next(req)
	}

	ctx, span := m.tracer.Start(req.Context(), "anthropic.messages.create")
	req = req.WithContext(ctx)

	span.SetAttributes(
		attribute.String(semconv.OpenInferenceSpanKind, semconv.SpanKindLLM),
		attribute.String(semconv.LLMSystem, semconv.LLMSystemAnthropic),
		attribute.String(semconv.LLMProvider, semconv.LLMProviderAnthropic),
	)
	// Propagate session.id / user.id / metadata / tag.tags from
	// baggage so the LLM span carries the customer's context
	// without them having to set the attributes manually.
	instrumentation.ApplyContextAttributes(ctx, span)

	reqBody, err := httputil.ReadAndRestore(&req.Body)
	if err != nil {
		// Body is partially consumed; forwarding would send a
		// malformed request. Fail fast.
		wrapped := fmt.Errorf("anthropic otel middleware: read request body: %w", err)
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
		// We have a response from the API but couldn't capture the
		// body for span attributes. Surface the failure so the span
		// isn't silently incomplete.
		span.RecordError(fmt.Errorf("read response body: %w", readErr))
	} else if respBody != nil {
		m.setResponseAttrs(span, respBody, resp.StatusCode)
	}
	span.End()
	return resp, nil
}

func isMessageCreate(req *http.Request) bool {
	return req.Method == http.MethodPost && strings.HasSuffix(req.URL.Path, "/v1/messages")
}

// requestPayload is a subset of MessageNewParams shaped for span attributes.
// We use a minimal struct rather than the SDK's typed params so the
// middleware never panics on unknown future fields.
type requestPayload struct {
	Model       string           `json:"model"`
	MaxTokens   int64            `json:"max_tokens"`
	Temperature *float64         `json:"temperature,omitempty"`
	TopP        *float64         `json:"top_p,omitempty"`
	TopK        *int64           `json:"top_k,omitempty"`
	System      json.RawMessage  `json:"system,omitempty"`
	Messages    []requestMessage `json:"messages"`
	Tools       []map[string]any `json:"tools,omitempty"`
}

type requestMessage struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
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
		if invocation := invocationParamsJSON(p); invocation != "" {
			span.SetAttributes(attribute.String(semconv.LLMInvocationParameters, invocation))
		}
	}

	// Input messages: drop the entire llm.input_messages.* structure
	// when either HideInputs or HideInputMessages is set, matching
	// Python's TraceConfig.mask which returns None for the whole
	// LLM_INPUT_MESSAGES key family. HideInputText (alone) keeps the
	// structure but redacts text content.
	if !m.config.HideInputs && !m.config.HideInputMessages {
		idx := 0
		if len(p.System) > 0 {
			text := extractText(p.System)
			if text != "" {
				span.SetAttributes(attribute.String(semconv.LLMInputMessageRoleKey(idx), "system"))
				if m.config.HideInputText {
					span.SetAttributes(attribute.String(semconv.LLMInputMessageContentKey(idx), instrumentation.RedactedValue))
				} else {
					span.SetAttributes(attribute.String(semconv.LLMInputMessageContentKey(idx), text))
				}
				idx++
			}
		}
		for _, msg := range p.Messages {
			text := extractText(msg.Content)
			span.SetAttributes(attribute.String(semconv.LLMInputMessageRoleKey(idx), msg.Role))
			if text != "" {
				if m.config.HideInputText {
					span.SetAttributes(attribute.String(semconv.LLMInputMessageContentKey(idx), instrumentation.RedactedValue))
				} else {
					span.SetAttributes(attribute.String(semconv.LLMInputMessageContentKey(idx), text))
				}
			}
			idx++
		}
	}

	if len(p.Messages) > 0 {
		lastUser := lastUserMessage(p.Messages)
		if lastUser != "" {
			span.SetAttributes(attribute.String(semconv.InputValue, m.config.MaskInputValue(lastUser)))
		}
	}
}

func invocationParamsJSON(p requestPayload) string {
	params := map[string]any{}
	if p.MaxTokens != 0 {
		params["max_tokens"] = p.MaxTokens
	}
	if p.Temperature != nil {
		params["temperature"] = *p.Temperature
	}
	if p.TopP != nil {
		params["top_p"] = *p.TopP
	}
	if p.TopK != nil {
		params["top_k"] = *p.TopK
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

func lastUserMessage(msgs []requestMessage) string {
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role == "user" {
			return extractText(msgs[i].Content)
		}
	}
	return ""
}

// extractText handles the three shapes a message-content field can take in
// the Anthropic API: bare string, [{"type":"text","text":"..."}, ...], or
// a single object {"type":"text","text":"..."}. Tool-use blocks and image
// blocks are intentionally ignored — their span representation lives in
// dedicated TOOL spans the caller is expected to add manually.
func extractText(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	// Try string first.
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s
	}
	// Try []{type,text} next.
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
	// Try single {type,text}.
	var obj struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(raw, &obj); err == nil && obj.Type == "text" {
		return obj.Text
	}
	return ""
}

// responsePayload is a subset of the Message response.
type responsePayload struct {
	ID         string            `json:"id"`
	Role       string            `json:"role"`
	Model      string            `json:"model"`
	StopReason string            `json:"stop_reason"`
	Content    []responseContent `json:"content"`
	Usage      *responseUsage    `json:"usage,omitempty"`
}

type responseContent struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type responseUsage struct {
	InputTokens              int64 `json:"input_tokens"`
	OutputTokens             int64 `json:"output_tokens"`
	CacheCreationInputTokens int64 `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     int64 `json:"cache_read_input_tokens"`
}

func (m *middleware) setResponseAttrs(span trace.Span, body []byte, statusCode int) {
	if statusCode < 200 || statusCode >= 300 {
		span.SetStatus(codes.Error, http.StatusText(statusCode))
		// Do not parse error bodies as success payloads; the API returns
		// {"type":"error","error":{...}} which unmarshals into
		// responsePayload with zero token counts and would pollute the
		// span with misleading "0 tokens" attributes.
		return
	}

	var r responsePayload
	if err := json.Unmarshal(body, &r); err != nil {
		// Surface the parse failure instead of producing a silent
		// missing-attributes span.
		span.RecordError(fmt.Errorf("parse response body: %w", err))
		return
	}

	// Anthropic returns the canonical model name in the response — prefer
	// it over the request (which may be an alias).
	if r.Model != "" {
		span.SetAttributes(attribute.String(semconv.LLMModelName, r.Model))
	}
	if r.StopReason != "" {
		span.SetAttributes(attribute.String(semconv.LLMFinishReason, r.StopReason))
	}

	var text strings.Builder
	for _, c := range r.Content {
		if c.Type == "text" {
			text.WriteString(c.Text)
		}
	}
	if text.Len() > 0 {
		span.SetAttributes(attribute.String(semconv.OutputValue, m.config.MaskOutputValue(text.String())))
		// Output messages: drop the entire llm.output_messages.*
		// structure when either HideOutputs or HideOutputMessages is
		// set, matching Python's TraceConfig.mask. HideOutputText
		// alone keeps the structure but redacts text.
		if !m.config.HideOutputs && !m.config.HideOutputMessages {
			span.SetAttributes(attribute.String(semconv.LLMOutputMessageRoleKey(0), "assistant"))
			if m.config.HideOutputText {
				span.SetAttributes(attribute.String(semconv.LLMOutputMessageContentKey(0), instrumentation.RedactedValue))
			} else {
				span.SetAttributes(attribute.String(semconv.LLMOutputMessageContentKey(0), text.String()))
			}
		}
	}

	if r.Usage != nil {
		span.SetAttributes(
			attribute.Int64(semconv.LLMTokenCountPrompt, r.Usage.InputTokens),
			attribute.Int64(semconv.LLMTokenCountCompletion, r.Usage.OutputTokens),
			attribute.Int64(semconv.LLMTokenCountTotal, r.Usage.InputTokens+r.Usage.OutputTokens),
		)
		if r.Usage.CacheReadInputTokens > 0 {
			span.SetAttributes(attribute.Int64(semconv.LLMTokenCountPromptDetailsCacheRead, r.Usage.CacheReadInputTokens))
		}
		if r.Usage.CacheCreationInputTokens > 0 {
			span.SetAttributes(attribute.Int64(semconv.LLMTokenCountPromptDetailsCacheWrite, r.Usage.CacheCreationInputTokens))
		}
	}
}

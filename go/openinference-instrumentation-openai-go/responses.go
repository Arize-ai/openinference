package openai

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"sync"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"

	"github.com/Arize-ai/openinference/go/openinference-instrumentation"
	"github.com/Arize-ai/openinference/go/openinference-instrumentation-openai-go/internal/httputil"
	"github.com/Arize-ai/openinference/go/openinference-semantic-conventions"
)

// This file maps the Responses API (POST /v1/responses) onto
// OpenInference LLM attributes. The mapping follows the Python
// (_attributes/_responses_api.py) and JS (responsesAttributes.ts)
// OpenAI instrumentors: input and output items become messages,
// function_call items become tool calls, and function_call_output items
// become tool-role messages.

// base64ImageMaxLength matches the Python TraceConfig default
// (DEFAULT_BASE64_IMAGE_MAX_LENGTH): base64 data-URI images longer than
// this are recorded as RedactedValue instead of inline. The Go
// TraceConfig has no equivalent setting yet.
const base64ImageMaxLength = 32_000

type responsesRequestPayload struct {
	Model        string            `json:"model"`
	Instructions json.RawMessage   `json:"instructions,omitempty"`
	Input        json.RawMessage   `json:"input,omitempty"`
	Tools        []json.RawMessage `json:"tools,omitempty"`
}

// responsesItem is the union of the input and output item shapes the
// middleware records. Items are decoded one at a time so that an item
// type with an unexpected field shape is skipped on its own instead of
// failing the whole input or output array.
type responsesItem struct {
	Type      string          `json:"type"`
	ID        string          `json:"id"`
	Role      string          `json:"role"`
	Content   json.RawMessage `json:"content"`
	CallID    string          `json:"call_id"`
	Name      string          `json:"name"`
	Arguments string          `json:"arguments"`
	// Input is the free-form input of a custom_tool_call item.
	Input json.RawMessage `json:"input"`
	// Output is a string or a list of content parts on
	// function_call_output and custom_tool_call_output items.
	Output  json.RawMessage `json:"output"`
	Summary []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"summary"`
	EncryptedContent string `json:"encrypted_content"`
}

// responsesContentPart is one entry of a message's content list
// (input_text, output_text, refusal, input_image, ...).
type responsesContentPart struct {
	Type     string `json:"type"`
	Text     string `json:"text"`
	Refusal  string `json:"refusal"`
	ImageURL string `json:"image_url"`
}

// messageKeys builds the attribute keys and masking for one side of the
// conversation, so the item mapper can be shared by input and output.
type messageKeys struct {
	role        func(i int) string
	content     func(i int) string
	toolCallID  func(i int) string
	toolCall    func(i, j int, child string) string
	contentPart func(i, k int, child string) string
	maskText    func(string) string
	hideImages  bool
}

func (m *middleware) inputMessageKeys() messageKeys {
	return messageKeys{
		role:        semconv.LLMInputMessageRoleKey,
		content:     semconv.LLMInputMessageContentKey,
		toolCallID:  semconv.LLMInputMessageToolCallIDKey,
		toolCall:    semconv.LLMInputMessageToolCallKey,
		contentPart: inputMessageContentKey,
		maskText:    m.config.MaskInputText,
		hideImages:  m.config.ShouldHideInputImages(),
	}
}

func (m *middleware) outputMessageKeys() messageKeys {
	return messageKeys{
		role:        semconv.LLMOutputMessageRoleKey,
		content:     semconv.LLMOutputMessageContentKey,
		toolCallID:  func(i int) string { return outputMessageKey(i, semconv.MessageToolCallID) },
		toolCall:    semconv.LLMOutputMessageToolCallKey,
		contentPart: outputMessageContentKey,
		maskText:    m.config.MaskOutputText,
		hideImages:  m.config.ShouldHideOutputImages(),
	}
}

func (m *middleware) setResponsesRequestAttrs(span trace.Span, body []byte) {
	var p responsesRequestPayload
	if err := json.Unmarshal(body, &p); err != nil {
		return
	}

	// input.value first: if a long conversation pushes the span past the
	// SDK's attribute count limit, the later per-message attributes are
	// the ones dropped, not the top-level input. Under HideInputs the
	// value is redacted and, as in Python, the mime type is omitted
	// because the sentinel is not JSON.
	span.SetAttributes(attribute.String(semconv.InputValue, m.config.MaskInputValue(redactInputImages(body, m.config.ShouldHideInputImages()))))
	if !m.config.HideInputs {
		span.SetAttributes(attribute.String(semconv.InputMimeType, semconv.MimeTypeJSON))
	}

	if p.Model != "" {
		span.SetAttributes(attribute.String(semconv.LLMModelName, p.Model))
	}
	if !m.config.HideLLMInvocationParameters {
		if invocation := responsesInvocationParamsJSON(body); invocation != "" {
			span.SetAttributes(attribute.String(semconv.LLMInvocationParameters, invocation))
		}
	}

	// Same masking semantics as Chat Completions: HideInputs or
	// HideInputMessages drop llm.input_messages.* wholesale, and
	// HideInputText keeps the structure but redacts text.
	if !m.config.HideInputs && !m.config.HideInputMessages {
		keys := m.inputMessageKeys()
		var attrs []attribute.KeyValue
		i := 0
		// instructions becomes a leading system message, then each input
		// item takes the next index. Indices are contiguous, matching JS.
		if instructions := jsonString(p.Instructions); instructions != "" {
			attrs = append(attrs,
				attribute.String(keys.role(i), "system"),
				attribute.String(keys.content(i), keys.maskText(instructions)),
			)
			i++
		}
		if text, ok := jsonStringOK(p.Input); ok {
			attrs = append(attrs,
				attribute.String(keys.role(i), "user"),
				attribute.String(keys.content(i), keys.maskText(text)),
			)
		} else {
			var items []json.RawMessage
			if err := json.Unmarshal(p.Input, &items); err == nil {
				for _, raw := range items {
					attrs = append(attrs, responsesItemAttrs(raw, i, keys)...)
					i++
				}
			}
		}
		span.SetAttributes(attrs...)
	}

	// Tools: dropped under either HideInputs or HideLLMTools, as for
	// Chat Completions.
	if !m.config.HideInputs && !m.config.HideLLMTools {
		for i, tool := range p.Tools {
			var buf bytes.Buffer
			if err := json.Compact(&buf, tool); err == nil {
				span.SetAttributes(attribute.String(semconv.LLMToolKey(i), buf.String()))
			}
		}
	}
}

// responsesInvocationParamsJSON returns the request body minus the
// fields surfaced elsewhere (input, instructions, tools), matching the
// Python instrumentor. Returns "" if the body is not a JSON object or no
// params remain.
func responsesInvocationParamsJSON(body []byte) string {
	var raw map[string]any
	if err := json.Unmarshal(body, &raw); err != nil {
		return ""
	}
	delete(raw, "input")
	delete(raw, "instructions")
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

// redactInputImages returns the request body for input.value with the
// image_url of input_image parts replaced by RedactedValue when images
// are hidden or the URL is an oversized base64 data URI, as Python's
// redact_images_from_request_parameters does. The body is returned
// as-is when nothing needs redacting.
func redactInputImages(body []byte, hideAll bool) string {
	if !bytes.Contains(body, []byte(`"input_image"`)) {
		return string(body)
	}
	dec := json.NewDecoder(bytes.NewReader(body))
	dec.UseNumber() // keep large integers exact when re-encoding
	var req map[string]any
	if err := dec.Decode(&req); err != nil {
		return string(body)
	}
	items, _ := req["input"].([]any)
	changed := false
	for _, item := range items {
		obj, _ := item.(map[string]any)
		parts, _ := obj["content"].([]any)
		for _, part := range parts {
			p, _ := part.(map[string]any)
			if p["type"] != "input_image" {
				continue
			}
			if url, ok := p["image_url"].(string); ok && (hideAll || isOversizedBase64Image(url)) {
				p["image_url"] = instrumentation.RedactedValue
				changed = true
			}
		}
	}
	if !changed {
		return string(body)
	}
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(req); err != nil {
		return string(body)
	}
	return strings.TrimSuffix(buf.String(), "\n")
}

// isOversizedBase64Image reports whether url is a base64 image data URI
// longer than base64ImageMaxLength, using Python's is_base64_url test.
func isOversizedBase64Image(url string) bool {
	return strings.HasPrefix(url, "data:image/") && strings.Contains(url, "base64") && len(url) > base64ImageMaxLength
}

type responsesResponsePayload struct {
	ID     string            `json:"id"`
	Model  string            `json:"model"`
	Output []json.RawMessage `json:"output"`
	Usage  *responsesUsage   `json:"usage,omitempty"`
}

// responsesUsage mirrors the Responses API usage object. Detail counts
// are pointers so a reported zero is still recorded, matching Python and
// JS, while an absent field is not.
type responsesUsage struct {
	InputTokens        int64 `json:"input_tokens"`
	OutputTokens       int64 `json:"output_tokens"`
	TotalTokens        int64 `json:"total_tokens"`
	InputTokensDetails *struct {
		CachedTokens     *int64 `json:"cached_tokens"`
		CacheWriteTokens *int64 `json:"cache_write_tokens"`
	} `json:"input_tokens_details,omitempty"`
	OutputTokensDetails *struct {
		ReasoningTokens *int64 `json:"reasoning_tokens"`
	} `json:"output_tokens_details,omitempty"`
}

func (m *middleware) setResponsesResponseAttrs(span trace.Span, body []byte, statusCode int) {
	if setHTTPErrorStatus(span, statusCode) {
		return
	}
	m.setResponsesOutputAttrs(span, body, true)
}

// observeResponsesStream parses a streamed Responses call as the caller
// reads it and, when the span ends, records the response carried by the
// response.completed event, as Python and JS do. A stream closed before
// that event ends the span with request attributes only.
func (m *middleware) observeResponsesStream(span trace.Span, body *httputil.SpanEndingBody) {
	// Read and Close may run on different goroutines (e.g. a context
	// cancellation closing the body), so guard the parser state.
	var mu sync.Mutex
	var completed []byte
	parser := &httputil.SSEParser{OnEvent: func(data []byte) {
		// Skip the JSON decode for the many delta events.
		if !bytes.Contains(data, []byte(`"response.completed"`)) {
			return
		}
		var event struct {
			Type     string          `json:"type"`
			Response json.RawMessage `json:"response"`
		}
		if err := json.Unmarshal(data, &event); err != nil || event.Type != "response.completed" ||
			!bytes.HasPrefix(bytes.TrimSpace(event.Response), []byte("{")) {
			return
		}
		completed = append([]byte(nil), event.Response...)
	}}
	body.OnRead = func(p []byte) {
		mu.Lock()
		defer mu.Unlock()
		_, _ = parser.Write(p)
	}
	body.BeforeEnd = func(end httputil.StreamEnd) {
		mu.Lock()
		defer mu.Unlock()
		if completed != nil {
			// Don't mark a stream that hit a read or close error as OK:
			// an OK status would overwrite the recorded Error.
			m.setResponsesOutputAttrs(span, completed, !end.Failed)
		}
	}
}

// setResponsesOutputAttrs records the attributes of a Response object,
// either a non-streaming response body or the response carried by a
// stream's response.completed event, and sets the span status to OK
// when markOK is true, as Python and JS do.
func (m *middleware) setResponsesOutputAttrs(span trace.Span, body []byte, markOK bool) {
	var r responsesResponsePayload
	if err := json.Unmarshal(body, &r); err != nil {
		span.RecordError(fmt.Errorf("parse response body: %w", err))
		return
	}

	span.SetAttributes(attribute.String(semconv.OutputValue, m.config.MaskOutputValue(string(body))))
	if !m.config.HideOutputs {
		span.SetAttributes(attribute.String(semconv.OutputMimeType, semconv.MimeTypeJSON))
	}

	if r.Model != "" {
		span.SetAttributes(attribute.String(semconv.LLMModelName, r.Model))
	}

	if !m.config.HideOutputs && !m.config.HideOutputMessages {
		keys := m.outputMessageKeys()
		var attrs []attribute.KeyValue
		for i, raw := range r.Output {
			attrs = append(attrs, responsesItemAttrs(raw, i, keys)...)
		}
		span.SetAttributes(attrs...)
	}

	if u := r.Usage; u != nil {
		span.SetAttributes(
			attribute.Int64(semconv.LLMTokenCountPrompt, u.InputTokens),
			attribute.Int64(semconv.LLMTokenCountCompletion, u.OutputTokens),
			attribute.Int64(semconv.LLMTokenCountTotal, u.TotalTokens),
		)
		if d := u.InputTokensDetails; d != nil {
			if d.CachedTokens != nil {
				span.SetAttributes(attribute.Int64(semconv.LLMTokenCountPromptDetailsCacheRead, *d.CachedTokens))
			}
			if d.CacheWriteTokens != nil {
				span.SetAttributes(attribute.Int64(semconv.LLMTokenCountPromptDetailsCacheWrite, *d.CacheWriteTokens))
			}
		}
		if d := u.OutputTokensDetails; d != nil && d.ReasoningTokens != nil {
			span.SetAttributes(attribute.Int64(semconv.LLMTokenCountCompletionDetailsReasoning, *d.ReasoningTokens))
		}
	}

	if markOK {
		span.SetStatus(codes.Ok, "")
	}
}

// responsesItemAttrs returns the message attributes for one input or
// output item at message index i. Unknown item types and items that fail
// to decode yield no attributes, but the caller still consumes their
// index, as Python and JS do.
func responsesItemAttrs(raw json.RawMessage, i int, keys messageKeys) []attribute.KeyValue {
	var it responsesItem
	if err := json.Unmarshal(raw, &it); err != nil {
		return nil
	}
	var attrs []attribute.KeyValue
	// toolCall records the item as the message's single tool call.
	toolCall := func(id, name, arguments string) {
		if id != "" {
			attrs = append(attrs, attribute.String(keys.toolCall(i, 0, semconv.ToolCallID), id))
		}
		if name != "" {
			attrs = append(attrs, attribute.String(keys.toolCall(i, 0, semconv.ToolCallFunctionName), name))
		}
		if arguments != "" {
			attrs = append(attrs, attribute.String(keys.toolCall(i, 0, semconv.ToolCallFunctionArgumentsJSON), arguments))
		}
	}
	switch it.Type {
	case "":
		// An EasyInputMessage may omit "type"; as in Python, it counts as
		// a message only when it has both a role and content.
		if it.Role == "" || len(it.Content) == 0 {
			return nil
		}
		fallthrough
	case "message":
		if it.Role != "" {
			attrs = append(attrs, attribute.String(keys.role(i), it.Role))
		}
		attrs = append(attrs, responsesContentAttrs(it.Content, i, keys)...)
	case "function_call":
		attrs = append(attrs, attribute.String(keys.role(i), "assistant"))
		toolCall(it.CallID, it.Name, it.Arguments)
	case "custom_tool_call":
		attrs = append(attrs, attribute.String(keys.role(i), "assistant"))
		var args string
		if len(it.Input) > 0 {
			// Custom tools take free-form input; wrap it as
			// {"input": ...} so the arguments stay JSON, as Python does.
			if b, err := json.Marshal(map[string]json.RawMessage{"input": it.Input}); err == nil {
				args = string(b)
			}
		}
		toolCall(it.CallID, it.Name, args)
	case "computer_call":
		// Python records only the call id. JS also records the item type
		// as the name and the action as arguments.
		attrs = append(attrs, attribute.String(keys.role(i), "assistant"))
		toolCall(it.CallID, "", "")
	case "function_call_output", "custom_tool_call_output", "computer_call_output":
		attrs = append(attrs, attribute.String(keys.role(i), "tool"))
		if it.CallID != "" {
			attrs = append(attrs, attribute.String(keys.toolCallID(i), it.CallID))
		}
		// Python records no content for computer_call_output (a
		// screenshot); the other two carry the tool's output.
		if it.Type != "computer_call_output" {
			if output := stringOrJSON(it.Output); output != "" {
				attrs = append(attrs, attribute.String(keys.content(i), keys.maskText(output)))
			}
		}
	case "reasoning":
		attrs = append(attrs,
			attribute.String(keys.role(i), "assistant"),
			attribute.String(keys.contentPart(i, 0, semconv.MessageContentType), "reasoning"),
		)
		var texts []string
		for _, s := range it.Summary {
			if s.Type == "summary_text" && s.Text != "" {
				texts = append(texts, s.Text)
			}
		}
		if len(texts) > 0 {
			attrs = append(attrs, attribute.String(keys.contentPart(i, 0, semconv.MessageContentText), keys.maskText(strings.Join(texts, "\n"))))
		}
		if it.EncryptedContent != "" {
			attrs = append(attrs, attribute.String(keys.contentPart(i, 0, semconv.MessageContentEncryptedContent), it.EncryptedContent))
		}
	case "web_search_call", "file_search_call":
		// Hosted tools: the tool call id is the item id and the item type
		// stands in for the function name, as in Python.
		attrs = append(attrs, attribute.String(keys.role(i), "assistant"))
		toolCall(it.ID, it.Type, "")
	}
	return attrs
}

// responsesContentAttrs handles a message's content, which is either a
// bare string (recorded as message.content) or a list of parts (recorded
// as message.contents.{k}.message_content.*). Text, output text and
// refusal parts are recorded with type "text", and input images with
// type "image" and their URL. Other part types are skipped but keep
// their index, as in Python.
func responsesContentAttrs(raw json.RawMessage, i int, keys messageKeys) []attribute.KeyValue {
	if text, ok := jsonStringOK(raw); ok {
		return []attribute.KeyValue{attribute.String(keys.content(i), keys.maskText(text))}
	}
	var parts []responsesContentPart
	if err := json.Unmarshal(raw, &parts); err != nil {
		return nil
	}
	var attrs []attribute.KeyValue
	for k, part := range parts {
		switch part.Type {
		case "input_text", "output_text", "refusal":
			text := part.Text
			if part.Type == "refusal" {
				text = part.Refusal
			}
			attrs = append(attrs,
				attribute.String(keys.contentPart(i, k, semconv.MessageContentType), "text"),
				attribute.String(keys.contentPart(i, k, semconv.MessageContentText), keys.maskText(text)),
			)
		case "input_image":
			if part.ImageURL == "" {
				continue
			}
			attrs = append(attrs, attribute.String(keys.contentPart(i, k, semconv.MessageContentType), "image"))
			// Hidden images drop the URL but keep the part's type, as
			// Python's TraceConfig.mask does.
			if !keys.hideImages {
				url := part.ImageURL
				if isOversizedBase64Image(url) {
					url = instrumentation.RedactedValue
				}
				attrs = append(attrs, attribute.String(keys.contentPart(i, k, semconv.MessageContentImage+"."+semconv.ImageURL), url))
			}
		}
	}
	return attrs
}

func inputMessageContentKey(i, k int, child string) string {
	return inputMessageKey(i, semconv.MessageContents+"."+strconv.Itoa(k)+"."+child)
}

func outputMessageContentKey(i, k int, child string) string {
	return outputMessageKey(i, semconv.MessageContents+"."+strconv.Itoa(k)+"."+child)
}

// jsonStringOK decodes raw as a JSON string, reporting whether it was
// one.
func jsonStringOK(raw json.RawMessage) (string, bool) {
	if len(raw) == 0 || raw[0] != '"' {
		return "", false
	}
	var s string
	if err := json.Unmarshal(raw, &s); err != nil {
		return "", false
	}
	return s, true
}

func jsonString(raw json.RawMessage) string {
	s, _ := jsonStringOK(raw)
	return s
}

// stringOrJSON returns raw decoded when it is a JSON string, or the
// compacted JSON otherwise (e.g. a function_call_output whose output is
// a list of content parts).
func stringOrJSON(raw json.RawMessage) string {
	if len(raw) == 0 || string(raw) == "null" {
		return ""
	}
	if s, ok := jsonStringOK(raw); ok {
		return s
	}
	var buf bytes.Buffer
	if err := json.Compact(&buf, raw); err != nil {
		return ""
	}
	return buf.String()
}

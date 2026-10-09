package openai

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"

	"github.com/Arize-ai/openinference/go/openinference-instrumentation"
	"github.com/Arize-ai/openinference/go/openinference-semantic-conventions"
)

// This file maps the Responses API (POST /v1/responses) onto
// OpenInference LLM attributes. The mapping follows the Python
// (_attributes/_responses_api.py) and JS (responsesAttributes.ts)
// OpenAI instrumentors: input and output items become messages,
// function_call items become tool calls, and function_call_output items
// become tool-role messages.

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
	Type    string `json:"type"`
	Text    string `json:"text"`
	Refusal string `json:"refusal"`
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
	span.SetAttributes(attribute.String(semconv.InputValue, m.config.MaskInputValue(string(body))))
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
		hideText := m.config.HideInputText
		var attrs []attribute.KeyValue
		i := 0
		// instructions becomes a leading system message, then each input
		// item takes the next index. Indices are contiguous, matching JS.
		if instructions := jsonString(p.Instructions); instructions != "" {
			prefix := messagePrefix(semconv.LLMInputMessages, i)
			attrs = append(attrs,
				attribute.String(prefix+semconv.MessageRole, "system"),
				attribute.String(prefix+semconv.MessageContent, maskText(instructions, hideText)),
			)
			i++
		}
		if text, ok := jsonStringOK(p.Input); ok {
			prefix := messagePrefix(semconv.LLMInputMessages, i)
			attrs = append(attrs,
				attribute.String(prefix+semconv.MessageRole, "user"),
				attribute.String(prefix+semconv.MessageContent, maskText(text, hideText)),
			)
		} else {
			var items []json.RawMessage
			if err := json.Unmarshal(p.Input, &items); err == nil {
				for _, raw := range items {
					attrs = append(attrs, responsesItemAttrs(raw, messagePrefix(semconv.LLMInputMessages, i), hideText)...)
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

type responsesResponsePayload struct {
	ID     string             `json:"id"`
	Model  string             `json:"model"`
	Output []json.RawMessage  `json:"output"`
	Usage  *responsesUsageObj `json:"usage,omitempty"`
}

// responsesUsageObj mirrors the Responses API usage object. Detail
// counts are pointers so a reported zero is still recorded, matching
// Python and JS, while an absent field is not.
type responsesUsageObj struct {
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
		var attrs []attribute.KeyValue
		for i, raw := range r.Output {
			attrs = append(attrs, responsesItemAttrs(raw, messagePrefix(semconv.LLMOutputMessages, i), m.config.HideOutputText)...)
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
}

// responsesItemAttrs returns the message attributes for one input or
// output item under prefix (e.g. "llm.input_messages.2."). Unknown item
// types and items that fail to decode yield no attributes, but the
// caller still consumes their index, as Python and JS do.
func responsesItemAttrs(raw json.RawMessage, prefix string, hideText bool) []attribute.KeyValue {
	var it responsesItem
	if err := json.Unmarshal(raw, &it); err != nil {
		return nil
	}
	toolCall := prefix + semconv.MessageToolCalls + ".0."
	var attrs []attribute.KeyValue
	switch it.Type {
	case "", "message":
		// EasyInputMessage items may omit "type".
		if it.Role == "" {
			return nil
		}
		attrs = append(attrs, attribute.String(prefix+semconv.MessageRole, it.Role))
		attrs = append(attrs, responsesContentAttrs(it.Content, prefix, hideText)...)
	case "function_call":
		attrs = append(attrs, attribute.String(prefix+semconv.MessageRole, "assistant"))
		if it.CallID != "" {
			attrs = append(attrs, attribute.String(toolCall+semconv.ToolCallID, it.CallID))
		}
		if it.Name != "" {
			attrs = append(attrs, attribute.String(toolCall+semconv.ToolCallFunctionName, it.Name))
		}
		if it.Arguments != "" {
			attrs = append(attrs, attribute.String(toolCall+semconv.ToolCallFunctionArgumentsJSON, it.Arguments))
		}
	case "custom_tool_call":
		attrs = append(attrs, attribute.String(prefix+semconv.MessageRole, "assistant"))
		if it.CallID != "" {
			attrs = append(attrs, attribute.String(toolCall+semconv.ToolCallID, it.CallID))
		}
		if it.Name != "" {
			attrs = append(attrs, attribute.String(toolCall+semconv.ToolCallFunctionName, it.Name))
		}
		if len(it.Input) > 0 {
			// Custom tools take free-form input; wrap it as
			// {"input": ...} so the arguments stay JSON, as Python does.
			if args, err := json.Marshal(map[string]json.RawMessage{"input": it.Input}); err == nil {
				attrs = append(attrs, attribute.String(toolCall+semconv.ToolCallFunctionArgumentsJSON, string(args)))
			}
		}
	case "function_call_output", "custom_tool_call_output":
		attrs = append(attrs, attribute.String(prefix+semconv.MessageRole, "tool"))
		if it.CallID != "" {
			attrs = append(attrs, attribute.String(prefix+semconv.MessageToolCallID, it.CallID))
		}
		if output := stringOrJSON(it.Output); output != "" {
			attrs = append(attrs, attribute.String(prefix+semconv.MessageContent, maskText(output, hideText)))
		}
	case "reasoning":
		attrs = append(attrs, attribute.String(prefix+semconv.MessageRole, "assistant"))
		content := prefix + semconv.MessageContents + ".0."
		attrs = append(attrs, attribute.String(content+semconv.MessageContentType, "reasoning"))
		var texts []string
		for _, s := range it.Summary {
			if s.Type == "summary_text" && s.Text != "" {
				texts = append(texts, s.Text)
			}
		}
		if len(texts) > 0 {
			attrs = append(attrs, attribute.String(content+semconv.MessageContentText, maskText(strings.Join(texts, "\n"), hideText)))
		}
		if it.EncryptedContent != "" {
			attrs = append(attrs, attribute.String(content+semconv.MessageContentEncryptedContent, it.EncryptedContent))
		}
	case "web_search_call", "file_search_call":
		// Hosted tools: the tool call id is the item id and the item type
		// stands in for the function name, as in Python and JS.
		attrs = append(attrs, attribute.String(prefix+semconv.MessageRole, "assistant"))
		if it.ID != "" {
			attrs = append(attrs, attribute.String(toolCall+semconv.ToolCallID, it.ID))
		}
		attrs = append(attrs, attribute.String(toolCall+semconv.ToolCallFunctionName, it.Type))
	}
	return attrs
}

// responsesContentAttrs handles a message's content, which is either a
// bare string (recorded as message.content) or a list of parts (recorded
// as message.contents.{k}.message_content.*). Text, output text and
// refusal parts are recorded with type "text"; other part types are
// skipped but keep their index, as in Python.
func responsesContentAttrs(raw json.RawMessage, prefix string, hideText bool) []attribute.KeyValue {
	if text, ok := jsonStringOK(raw); ok {
		return []attribute.KeyValue{attribute.String(prefix+semconv.MessageContent, maskText(text, hideText))}
	}
	var parts []responsesContentPart
	if err := json.Unmarshal(raw, &parts); err != nil {
		return nil
	}
	var attrs []attribute.KeyValue
	for k, part := range parts {
		var text string
		switch part.Type {
		case "input_text", "output_text":
			text = part.Text
		case "refusal":
			text = part.Refusal
		default:
			continue
		}
		content := prefix + semconv.MessageContents + "." + strconv.Itoa(k) + "."
		attrs = append(attrs,
			attribute.String(content+semconv.MessageContentType, "text"),
			attribute.String(content+semconv.MessageContentText, maskText(text, hideText)),
		)
	}
	return attrs
}

// messagePrefix returns e.g. "llm.input_messages.3." for (base, 3).
func messagePrefix(base string, i int) string {
	return base + "." + strconv.Itoa(i) + "."
}

func maskText(text string, hide bool) string {
	if hide {
		return instrumentation.RedactedValue
	}
	return text
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

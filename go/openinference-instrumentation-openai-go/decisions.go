package openai

import (
	"encoding/json"
	"fmt"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"

	"github.com/Arize-ai/openinference/go/openinference-semantic-conventions"
)

// This file maps the Decisions API (POST /v1/decisions) onto an
// OpenInference DECISION span, following the Python
// (_get_attributes_from_decision*) and JS (decisionsAttributes.ts)
// OpenAI instrumentors. A decision model answers typed questions about
// its input instead of generating text, so the span identifies the model
// under decision.* and carries no llm.* model or token attributes. The
// request and response are recorded whole, as JSON, in input.value and
// output.value; they have no message equivalent. See
// https://github.com/Arize-ai/openinference/blob/main/spec/decision_spans.md

// decisionStartAttrs returns the span kind, system and provider
// attributes of a DECISION span.
func decisionStartAttrs(host string) []attribute.KeyValue {
	return []attribute.KeyValue{
		attribute.String(semconv.OpenInferenceSpanKind, semconv.SpanKindDecision),
		attribute.String(semconv.DecisionSystem, semconv.DecisionSystemOpenAI),
		attribute.String(semconv.DecisionProvider, providerForHost(host)),
	}
}

type decisionsRequestPayload struct {
	Model string `json:"model"`
}

func (m *middleware) setDecisionsRequestAttrs(span trace.Span, body []byte) {
	var p decisionsRequestPayload
	if err := json.Unmarshal(body, &p); err != nil {
		return
	}

	// input.value goes through the same image redaction as Responses:
	// decision images are inline base64 data URIs, so HideInputImages and
	// the oversized-image limit both apply. Under HideInputs the value is
	// redacted and the mime type omitted because the sentinel is not JSON.
	span.SetAttributes(attribute.String(semconv.InputValue, m.config.MaskInputValue(redactInputImages(body, m.config.ShouldHideInputImages()))))
	if !m.config.HideInputs {
		span.SetAttributes(attribute.String(semconv.InputMimeType, semconv.MimeTypeJSON))
	}

	// decision.model_name is set from the request so that it is present
	// even when the call fails; a successful response overwrites it with
	// the model that answered.
	if p.Model != "" {
		span.SetAttributes(
			attribute.String(semconv.DecisionRequestModelName, p.Model),
			attribute.String(semconv.DecisionModelName, p.Model),
		)
	}
}

// decisionsResponsePayload holds the response fields the span records
// beyond output.value. Token counts are pointers so a reported zero is
// still recorded (output_tokens is 0 for gpt-6-luna today), while an
// absent field is not, matching Python and JS.
type decisionsResponsePayload struct {
	Model string `json:"model"`
	Usage *struct {
		InputTokens  *int64 `json:"input_tokens"`
		OutputTokens *int64 `json:"output_tokens"`
	} `json:"usage,omitempty"`
}

func (m *middleware) setDecisionsResponseAttrs(span trace.Span, body []byte, statusCode int) {
	if setHTTPErrorStatus(span, statusCode) {
		return
	}

	var r decisionsResponsePayload
	if err := json.Unmarshal(body, &r); err != nil {
		span.RecordError(fmt.Errorf("parse response body: %w", err))
		return
	}

	span.SetAttributes(attribute.String(semconv.OutputValue, m.config.MaskOutputValue(string(body))))
	if !m.config.HideOutputs {
		span.SetAttributes(attribute.String(semconv.OutputMimeType, semconv.MimeTypeJSON))
	}

	if r.Model != "" {
		span.SetAttributes(
			attribute.String(semconv.DecisionResponseModelName, r.Model),
			attribute.String(semconv.DecisionModelName, r.Model),
		)
	}

	// Decision spans have no prompt/completion split and no total, so
	// only the input and output counts are recorded.
	if u := r.Usage; u != nil {
		if u.InputTokens != nil {
			span.SetAttributes(attribute.Int64(semconv.DecisionTokenCountInput, *u.InputTokens))
		}
		if u.OutputTokens != nil {
			span.SetAttributes(attribute.Int64(semconv.DecisionTokenCountOutput, *u.OutputTokens))
		}
	}

	// A parsed 2xx response is a successful call: OK, as Python, JS and
	// Responses set it.
	span.SetStatus(codes.Ok, "")
}

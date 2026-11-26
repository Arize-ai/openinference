"""Pure attribute builders for ``google-genai`` ``decisions.create`` calls.

A Gemini Decisions API call (``client.decisions.create``) is modelled as an OpenInference
DECISION span: ``gemini-decision-1`` scores or selects among the caller's typed questions
rather than generating free-form text, so the span identifies the model under ``decision.*``
instead of ``llm.*`` and records the raw request and response in ``input.value`` /
``output.value``. See ``spec/spans/decision.md`` for the full mapping.

Every function here is side-effect free: it takes the parsed request parameters or the SDK
response object and returns a flat mapping of OpenInference span attributes.
"""

import logging
from collections.abc import Mapping as AbcMapping
from collections.abc import Sequence as AbcSequence
from typing import Any, Dict, Mapping, Optional

from opentelemetry.util.types import AttributeValue

from openinference.instrumentation import (
    REDACTED_VALUE,
    DecisionTokenCount,
    TraceConfig,
    get_decision_attributes,
    get_input_attributes,
    get_output_attributes,
    get_span_kind_attributes,
)
from openinference.instrumentation.google_genai._image_utils import (
    redact_images_from_request_parameters,
)
from openinference.semconv.trace import (
    OpenInferenceDecisionProviderValues,
    OpenInferenceDecisionSystemValues,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _to_builtins(value: Any) -> Any:
    """Converts an SDK object into JSON-compatible builtins.

    The request ``input`` and ``questions`` usually arrive as plain builtins the caller
    passed, while the response ``answers``, ``usage``, and ``prompt_feedback`` are pydantic
    models. Pydantic models are dumped through ``model_dump(mode="json")``; abstract
    containers are materialized; anything else is recorded as its string form.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if callable(model_dump := getattr(value, "model_dump", None)):
        try:
            return model_dump(mode="json")
        except Exception:
            logger.exception("Failed to dump %r", type(value))
    if isinstance(value, AbcMapping):
        return {str(k): _to_builtins(v) for k, v in value.items()}
    if isinstance(value, AbcSequence) and not isinstance(value, (str, bytes)):
        return [_to_builtins(v) for v in value]
    return str(value)


def get_request_attributes(
    request_parameters: Mapping[str, Any],
    config: Optional[TraceConfig] = None,
) -> Dict[str, AttributeValue]:
    """Returns the span attributes known before the request is sent.

    The ``input`` may carry inline images, audio, video, or documents as base64 ``data``.
    When a ``TraceConfig`` is given, those are redacted from ``input.value`` per
    ``hide_input_images`` and ``base64_image_max_length`` so the span does not carry
    megabytes of media. ``hide_inputs`` (hiding the whole value) is applied by ``OITracer``.

    Args:
        request_parameters: The parsed ``create`` arguments, carrying ``model``, ``input``,
            and ``questions``.
        config: The active ``TraceConfig``, used for image redaction.

    Returns:
        The span kind, ``input.value``, and the request-side ``decision.*`` attributes.
    """
    model = request_parameters.get("model")
    body: Dict[str, Any] = {
        "model": model,
        "input": _to_builtins(request_parameters.get("input")),
        "questions": _to_builtins(request_parameters.get("questions")),
    }
    input_attributes = get_input_attributes(body, mime_type=OpenInferenceMimeTypeValues.JSON)
    if config is not None:
        try:
            redacted = redact_images_from_request_parameters(
                body,
                hide_input_images=bool(config.hide_input_images),
                base64_image_max_length=int(config.base64_image_max_length or 0),
            )
            input_attributes = get_input_attributes(
                redacted, mime_type=OpenInferenceMimeTypeValues.JSON
            )
        except Exception:
            logger.exception("Could not redact images from decision request input")
            input_attributes = get_input_attributes(REDACTED_VALUE)
    return {
        **get_span_kind_attributes(OpenInferenceSpanKindValues.DECISION),
        **input_attributes,
        **get_decision_attributes(
            system=OpenInferenceDecisionSystemValues.GOOGLE,
            provider=OpenInferenceDecisionProviderValues.GOOGLE,
            request_model_name=model if isinstance(model, str) else None,
        ),
    }


def get_response_attributes(response: Any) -> Dict[str, AttributeValue]:
    """Returns the span attributes derived from a ``CreateDecisionResponse``.

    The Gemini Decisions API response does not echo the model, so
    ``decision.response.model_name`` is left unset and ``decision.model_name`` keeps the
    requested model. Output tokens are always ``0`` today.

    Args:
        response: The response object returned by ``decisions.create``.

    Returns:
        ``output.value`` and the response-side ``decision.*`` token counts.
    """
    usage = getattr(response, "usage", None)
    body: Dict[str, Any] = {
        "answers": _to_builtins(getattr(response, "answers", None)),
        "usage": _to_builtins(usage),
        "prompt_feedback": _to_builtins(getattr(response, "prompt_feedback", None)),
    }
    return {
        **get_decision_attributes(token_count=_get_token_count(usage)),
        **get_output_attributes(body, mime_type=OpenInferenceMimeTypeValues.JSON),
    }


def _get_token_count(usage: Any) -> Optional[DecisionTokenCount]:
    """Returns the input and output token counts, or ``None`` when unreported.

    Maps the Gemini ``Usage`` fields ``total_input_tokens`` and ``total_output_tokens`` to
    ``decision.token_count.input`` and ``decision.token_count.output``. Decision spans have
    no total, so none is derived.
    """
    input_tokens = getattr(usage, "total_input_tokens", None)
    output_tokens = getattr(usage, "total_output_tokens", None)
    token_count: DecisionTokenCount = {}
    if isinstance(input_tokens, int):
        token_count["input"] = input_tokens
    if isinstance(output_tokens, int):
        token_count["output"] = output_tokens
    return token_count or None

"""Pure attribute builders for TypeSafe AI ``system_one`` calls.

Every function here is side-effect free: it takes the SDK call arguments or the SDK
response object and returns a flat mapping of OpenInference span attributes.

A ``system_one`` call is modelled as an LLM span whose structured output is the map of
typed answers. ``input.value`` and ``output.value`` mirror the wire request and response
bodies, and they are the only place the request's ``state`` and ``questions`` are
recorded, so ``hide_inputs`` alone keeps every part of the request off the span.
``llm.invocation_parameters`` carries only call configuration: the ``model`` and any
``extra_body`` fields. See the package README for the full attribute mapping.

A ``system_one`` call is not a chat exchange: neither side is a message list, so
``llm.input_messages`` and ``llm.output_messages`` are deliberately not recorded.
"""

import logging
from collections.abc import Mapping as AbcMapping
from collections.abc import Sequence as AbcSequence
from typing import Any, Dict, Mapping, Optional

import msgspec
from opentelemetry.util.types import AttributeValue

from openinference.instrumentation import (
    TokenCount,
    get_input_attributes,
    get_llm_attributes,
    get_output_attributes,
    get_span_kind_attributes,
)
from openinference.semconv.trace import OpenInferenceMimeTypeValues, OpenInferenceSpanKindValues

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

LLM_PROVIDER = "typesafe"


def _enc_hook(value: Any) -> Any:
    """Materializes what msgspec cannot encode on its own into JSON-compatible builtins.

    Two kinds of value reach this hook. Pydantic models, which is what the SDK's questions,
    answers, and usage became in ``typesafe-sdk`` 0.7.0, are dumped through their own
    serializer, the one the SDK encodes with. Abstract containers, which the SDK accepts
    anywhere it declares a ``Mapping`` or a ``Sequence``, are materialized the way the SDK's
    own encoder fallback materializes them. Anything else is recorded as its string form.
    """
    if callable(model_dump := getattr(value, "model_dump", None)):
        try:
            return model_dump(mode="json")
        except Exception:
            logger.exception("Failed to dump %r", type(value))
    if isinstance(value, AbcMapping):
        return dict(value)
    if isinstance(value, AbcSequence):
        return list(value)
    return str(value)


def _to_builtins(value: Any) -> Any:
    """Converts the SDK's question, answer, and usage objects into JSON-compatible builtins.

    Args:
        value: Any value the SDK may accept or hand back: a msgspec struct (``typesafe-sdk``
            0.6.x), a pydantic model (0.7.0 and later), or a plain builtin.

    Returns:
        The value as JSON-compatible builtins, or ``str(value)`` if conversion fails.
    """
    try:
        return msgspec.to_builtins(value, str_keys=True, enc_hook=_enc_hook)
    except Exception:
        logger.exception("Failed to convert %r to builtins", type(value))
        return str(value)


def get_request_attributes(
    *,
    state: Any,
    questions: Any,
    model: Optional[str],
    extra_body: Optional[Mapping[str, Any]] = None,
) -> Dict[str, AttributeValue]:
    """Returns the span attributes that are known before the request is sent.

    Args:
        state: The state the questions are asked about, a string or a JSON-compatible value.
        questions: The map of typed questions, as SDK objects or raw dictionaries.
        model: The requested model, or ``None`` when the client default applies.
        extra_body: Extra top-level request body fields, if any.

    Returns:
        The span kind, ``input.value``, and the request-side ``llm.*`` attributes.
    """
    body: Dict[str, Any] = {
        "state": _to_builtins(state),
        "model": model,
        "questions": _to_builtins(questions),
    }
    if extra_body:
        # extra_body values are JSONValue, so they may hold abstract Mapping / Sequence
        # containers that need the same conversion as state and questions.
        body.update(_to_builtins(extra_body))
    # The questions map is request content, not call configuration: it carries the caller's
    # instructions. Recording it once, in input.value, keeps one masking flag in charge of
    # everything the request says.
    invocation_parameters = {
        k: v for k, v in body.items() if k not in ("state", "questions") and v is not None
    }
    return {
        **get_span_kind_attributes(OpenInferenceSpanKindValues.LLM),
        **get_input_attributes(body, mime_type=OpenInferenceMimeTypeValues.JSON),
        **get_llm_attributes(
            provider=LLM_PROVIDER,
            request_model_name=model,
            invocation_parameters=invocation_parameters,
        ),
    }


def get_response_attributes(response: Any) -> Dict[str, AttributeValue]:
    """Returns the span attributes derived from a ``SystemOneResponse``.

    Args:
        response: The response object returned by ``system_one``.

    Returns:
        ``output.value``, the resolved response model name, and token counts.
    """
    model = getattr(response, "model", None)
    usage = getattr(response, "usage", None)
    body: Dict[str, Any] = {
        "model": model,
        "answers": _to_builtins(getattr(response, "answers", {})),
        "usage": _to_builtins(usage),
    }
    return {
        **get_llm_attributes(
            response_model_name=model,
            token_count=_get_token_count(usage),
        ),
        **get_output_attributes(body, mime_type=OpenInferenceMimeTypeValues.JSON),
    }


def _get_token_count(usage: Any) -> Optional[TokenCount]:
    """Returns the prompt, completion, and total token counts, or ``None`` when unreported.

    Args:
        usage: The ``usage`` object on a ``SystemOneResponse``, or ``None``.

    Returns:
        A ``TokenCount`` with whichever counts the response reported, else ``None``. The
        total is only derived when both the prompt and completion counts are present.
    """
    prompt = getattr(usage, "input_tokens", None)
    completion = getattr(usage, "output_tokens", None)
    token_count: TokenCount = {}
    if isinstance(prompt, int):
        token_count["prompt"] = prompt
    if isinstance(completion, int):
        token_count["completion"] = completion
    if isinstance(prompt, int) and isinstance(completion, int):
        token_count["total"] = prompt + completion
    return token_count or None

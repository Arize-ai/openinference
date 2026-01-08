import logging
import warnings
from typing import Any, Iterable, Iterator, Mapping, NamedTuple, Optional, Sequence, Tuple

from opentelemetry import trace as trace_api
from opentelemetry.util.types import AttributeValue

from openinference.instrumentation import safe_json_dumps
from openinference.instrumentation.google_genai._with_span import _WithSpan
from openinference.semconv.trace import OpenInferenceMimeTypeValues, SpanAttributes

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Audio modality identifier (matches both enum and string representations)
_AUDIO_MODALITY = "AUDIO"


def _get_token_count_attributes_from_usage_metadata(
    usage_metadata: Mapping[str, Any],
) -> Iterator[Tuple[str, AttributeValue]]:
    """
    Extract token count attributes from usage metadata.

    Works with both typed objects (converted via model_dump()) and raw dicts.
    """
    if total_token_count := usage_metadata.get("total_token_count"):
        yield SpanAttributes.LLM_TOKEN_COUNT_TOTAL, int(total_token_count)

    # Extract prompt details audio tokens
    if prompt_tokens_details := usage_metadata.get("prompt_tokens_details"):
        prompt_details_audio = 0
        for modality_token_count in prompt_tokens_details:
            modality = modality_token_count.get("modality")
            # Handle both enum (via .value) and string representations
            modality_str = getattr(modality, "value", None) or modality
            if modality_str == _AUDIO_MODALITY and modality_token_count.get("token_count"):
                prompt_details_audio += modality_token_count["token_count"]
        if prompt_details_audio:
            yield SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_AUDIO, int(prompt_details_audio)

    # Calculate total prompt tokens (base + tool use)
    prompt_token_count = 0
    if base_prompt_tokens := usage_metadata.get("prompt_token_count"):
        prompt_token_count += base_prompt_tokens
    if tool_use_prompt_tokens := usage_metadata.get("tool_use_prompt_token_count"):
        prompt_token_count += tool_use_prompt_tokens
    if prompt_token_count:
        yield SpanAttributes.LLM_TOKEN_COUNT_PROMPT, int(prompt_token_count)

    # Extract completion details audio tokens
    if candidates_tokens_details := usage_metadata.get("candidates_tokens_details"):
        completion_details_audio = 0
        for modality_token_count in candidates_tokens_details:
            modality = modality_token_count.get("modality")
            modality_str = getattr(modality, "value", None) or modality
            if modality_str == _AUDIO_MODALITY and modality_token_count.get("token_count"):
                completion_details_audio += modality_token_count["token_count"]
        if completion_details_audio:
            yield (
                SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_AUDIO,
                int(completion_details_audio),
            )

    # Calculate total completion tokens (candidates + thoughts/reasoning)
    completion_token_count = 0
    if candidates_token_count := usage_metadata.get("candidates_token_count"):
        completion_token_count += candidates_token_count
    if thoughts_token_count := usage_metadata.get("thoughts_token_count"):
        yield SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING, int(thoughts_token_count)
        completion_token_count += thoughts_token_count
    if completion_token_count:
        yield SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, int(completion_token_count)


class _ValueAndType(NamedTuple):
    value: str
    type: OpenInferenceMimeTypeValues


def _io_value_and_type(obj: Any) -> _ValueAndType:
    if hasattr(obj, "model_dump_json") and callable(obj.model_dump_json):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # `warnings=False` in `model_dump_json()` is only supported in Pydantic v2
                value = obj.model_dump_json(exclude_unset=True)
            assert isinstance(value, str)
        except Exception:
            logger.exception("Failed to get model dump json")
        else:
            return _ValueAndType(value, OpenInferenceMimeTypeValues.JSON)
    if not isinstance(obj, str) and isinstance(obj, (Sequence, Mapping)):
        try:
            value = safe_json_dumps(obj)
        except Exception:
            logger.exception("Failed to dump json")
        else:
            return _ValueAndType(value, OpenInferenceMimeTypeValues.JSON)

    return _ValueAndType(str(obj), OpenInferenceMimeTypeValues.TEXT)


def _as_input_attributes(
    value_and_type: Optional[_ValueAndType],
) -> Iterator[Tuple[str, AttributeValue]]:
    if not value_and_type:
        return
    yield SpanAttributes.INPUT_VALUE, value_and_type.value
    # It's assumed to be TEXT by default, so we can skip to save one attribute.
    if value_and_type.type is not OpenInferenceMimeTypeValues.TEXT:
        yield SpanAttributes.INPUT_MIME_TYPE, value_and_type.type.value


def _as_output_attributes(
    value_and_type: Optional[_ValueAndType],
) -> Iterator[Tuple[str, AttributeValue]]:
    if not value_and_type:
        return
    yield SpanAttributes.OUTPUT_VALUE, value_and_type.value
    # It's assumed to be TEXT by default, so we can skip to save one attribute.
    if value_and_type.type is not OpenInferenceMimeTypeValues.TEXT:
        yield SpanAttributes.OUTPUT_MIME_TYPE, value_and_type.type.value


def _finish_tracing(
    with_span: _WithSpan,
    attributes: Iterable[Tuple[str, AttributeValue]],
    extra_attributes: Iterable[Tuple[str, AttributeValue]],
    status: Optional[trace_api.Status] = None,
) -> None:
    try:
        attributes_dict = dict(attributes)
    except Exception:
        logger.exception("Failed to get attributes")
    try:
        extra_attributes_dict = dict(extra_attributes)
    except Exception:
        logger.exception("Failed to get extra attributes")
    try:
        with_span.finish_tracing(
            status=status,
            attributes=attributes_dict,
            extra_attributes=extra_attributes_dict,
        )
    except Exception:
        logger.exception("Failed to finish tracing")

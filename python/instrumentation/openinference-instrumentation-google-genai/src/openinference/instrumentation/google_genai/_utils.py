import logging
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    Iterator,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

from opentelemetry import trace as trace_api
from opentelemetry.util.types import AttributeValue

from openinference.instrumentation import safe_json_dumps
from openinference.instrumentation.google_genai._with_span import _WithSpan
from openinference.semconv.trace import OpenInferenceMimeTypeValues, SpanAttributes

if TYPE_CHECKING:
    from google.genai import types

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _get_token_count_attributes_from_usage_metadata(
    usage_metadata: "types.GenerateContentResponseUsageMetadata",
) -> Iterator[Tuple[str, AttributeValue]]:
    """Extract token count attributes from usage metadata."""
    from google.genai import types

    if usage_metadata.total_token_count:
        yield SpanAttributes.LLM_TOKEN_COUNT_TOTAL, usage_metadata.total_token_count

    # Extract prompt details audio tokens
    if usage_metadata.prompt_tokens_details:
        prompt_details_audio = 0
        for modality_token_count in usage_metadata.prompt_tokens_details:
            if (
                modality_token_count.modality is types.MediaModality.AUDIO
                and modality_token_count.token_count
            ):
                prompt_details_audio += modality_token_count.token_count
        if prompt_details_audio:
            yield SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_AUDIO, prompt_details_audio

    # Calculate total prompt tokens (base + tool use)
    prompt_token_count = 0
    if usage_metadata.prompt_token_count:
        prompt_token_count += usage_metadata.prompt_token_count
    if usage_metadata.tool_use_prompt_token_count:
        prompt_token_count += usage_metadata.tool_use_prompt_token_count
    if prompt_token_count:
        yield SpanAttributes.LLM_TOKEN_COUNT_PROMPT, prompt_token_count

    # Extract completion details audio tokens
    if usage_metadata.candidates_tokens_details:
        completion_details_audio = 0
        for modality_token_count in usage_metadata.candidates_tokens_details:
            if (
                modality_token_count.modality is types.MediaModality.AUDIO
                and modality_token_count.token_count
            ):
                completion_details_audio += modality_token_count.token_count
        if completion_details_audio:
            yield SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_AUDIO, completion_details_audio

    # Calculate total completion tokens (candidates + thoughts/reasoning)
    completion_token_count = 0
    if usage_metadata.candidates_token_count:
        completion_token_count += usage_metadata.candidates_token_count
    if usage_metadata.thoughts_token_count:
        yield (
            SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING,
            usage_metadata.thoughts_token_count,
        )
        completion_token_count += usage_metadata.thoughts_token_count
    if completion_token_count:
        yield SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, completion_token_count


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

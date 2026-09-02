import logging
from typing import TYPE_CHECKING, Any, Iterator, NamedTuple, Optional, Protocol, Tuple

from openinference.semconv.trace import OpenInferenceMimeTypeValues, SpanAttributes
from opentelemetry import trace as trace_api
from opentelemetry.util.types import Attributes, AttributeValue

from openinference.instrumentation import safe_json_dumps
from openinference.instrumentation.anthropic._with_span import _WithSpan

if TYPE_CHECKING:
    from anthropic.types import Usage

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _get_token_counts(usage: "Usage") -> Iterator[Tuple[str, AttributeValue]]:
    """Yields token count attributes from an Anthropic `Usage`.

    Shared by the streaming and non-streaming paths so both emit the same
    attributes. See
    https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching#tracking-cache-performance
    - cache_creation_input_tokens: Number of tokens written to the cache when creating a new entry.
    - cache_read_input_tokens: Number of tokens retrieved from the cache for this request.
    - input_tokens: Number of input tokens which were not read from or used to create a cache.
    """
    prompt_tokens = (
        (usage.input_tokens or 0)
        + (usage.cache_creation_input_tokens or 0)
        + (usage.cache_read_input_tokens or 0)
    )
    if prompt_tokens:
        yield SpanAttributes.LLM_TOKEN_COUNT_PROMPT, prompt_tokens
    if usage.output_tokens:
        yield SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, usage.output_tokens
    if usage.cache_read_input_tokens:
        yield (
            SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ,
            usage.cache_read_input_tokens,
        )
    if usage.cache_creation_input_tokens:
        yield (
            SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE,
            usage.cache_creation_input_tokens,
        )
    # Anthropic's Usage has no total field, so derive it from prompt + completion.
    if total_tokens := prompt_tokens + (usage.output_tokens or 0):
        yield SpanAttributes.LLM_TOKEN_COUNT_TOTAL, total_tokens


class _ValueAndType(NamedTuple):
    value: str
    type: OpenInferenceMimeTypeValues


class _HasAttributes(Protocol):
    def get_attributes(self) -> Iterator[Tuple[str, AttributeValue]]: ...


def _finish_tracing(
    with_span: _WithSpan,
    has_attributes: _HasAttributes,
    status: Optional[trace_api.Status] = None,
) -> None:
    try:
        attributes: Attributes = dict(has_attributes.get_attributes())
    except Exception:
        logger.exception("Failed to get attributes")
        attributes = None
    try:
        with_span.finish_tracing(
            status=status,
            attributes=attributes,
        )
    except Exception:
        logger.exception("Failed to finish tracing")


def _io_value_and_type(obj: Any) -> _ValueAndType:
    try:
        return _ValueAndType(safe_json_dumps(obj), OpenInferenceMimeTypeValues.JSON)
    except Exception:
        logger.exception("Failed to get input attributes from request parameters.")
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

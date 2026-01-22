import logging
import warnings
from typing import Any, Iterable, Iterator, Mapping, NamedTuple, Optional, Sequence, Tuple

from opentelemetry import trace as trace_api
from opentelemetry.util.types import AttributeValue

from openinference.instrumentation import safe_json_dumps
from openinference.instrumentation.portkey._with_span import _WithSpan
from openinference.semconv.trace import (
    OpenInferenceLLMSystemValues,
    OpenInferenceMimeTypeValues,
    SpanAttributes,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class _ValueAndType(NamedTuple):
    value: str
    type: OpenInferenceMimeTypeValues


def _io_value_and_type(obj: Any) -> _ValueAndType:
    if hasattr(obj, "model_dump_json") and callable(obj.model_dump_json):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
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
    if value_and_type.type is not OpenInferenceMimeTypeValues.TEXT:
        yield SpanAttributes.INPUT_MIME_TYPE, value_and_type.type.value


def _as_output_attributes(
    value_and_type: Optional[_ValueAndType],
) -> Iterator[Tuple[str, AttributeValue]]:
    if not value_and_type:
        return
    yield SpanAttributes.OUTPUT_VALUE, value_and_type.value
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
    except Exception as e:
        print(e)
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


def infer_llm_system_from_model(
    model_name: Optional[str] = None,
) -> Optional[OpenInferenceLLMSystemValues]:
    """Infer the LLM system from a model identifier when possible."""
    if not model_name:
        return None

    model = model_name.lower()

    if model.startswith(
        (
            "gpt-",
            "gpt.",
            "o1",
            "o3",
            "o4",
            "text-embedding",
            "davinci",
            "curie",
            "babbage",
            "ada",
            "azure_openai",
            "azure_ai",
            "azure",
        )
    ):
        return OpenInferenceLLMSystemValues.OPENAI

    if model.startswith(("anthropic.claude", "anthropic/", "claude-", "google_anthropic_vertex")):
        return OpenInferenceLLMSystemValues.ANTHROPIC

    if model.startswith(("cohere.command", "command", "cohere")):
        return OpenInferenceLLMSystemValues.COHERE

    if model.startswith(("mistralai", "mixtral", "mistral", "pixtral")):
        return OpenInferenceLLMSystemValues.MISTRALAI

    if model.startswith(
        ("google_vertexai", "google_genai", "vertexai", "vertex_ai", "vertex", "gemini", "google")
    ):
        return OpenInferenceLLMSystemValues.VERTEXAI

    return None

import base64
import logging
import struct
import warnings
from typing import Any, Iterable, Iterator, Mapping, Sequence, Tuple

from opentelemetry.util.types import AttributeValue

from openinference.instrumentation import safe_json_dumps
from openinference.semconv.trace import (
    EmbeddingAttributes,
    MessageAttributes,
    SpanAttributes,
    ToolCallAttributes,
)

__all__ = ("_ResponseAttributesExtractor",)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class _ResponseAttributesExtractor:
    def get_attributes(self, response: Any) -> Iterator[Tuple[str, AttributeValue]]:
        try:
            with warnings.catch_warnings():
                # Pydantic emits serializer warnings for union-typed response fields
                # whose server-sent shape is not an exact match. `warnings=False` on
                # `model_dump_json()` is Pydantic v2 only, so suppress them here to
                # keep instrumentation from polluting the caller's stderr.
                warnings.simplefilter("ignore")
                value = response.model_dump_json(exclude_unset=True)
        except Exception:
            yield SpanAttributes.OUTPUT_VALUE, str(response)
            return
        yield SpanAttributes.OUTPUT_VALUE, value
        yield SpanAttributes.OUTPUT_MIME_TYPE, "application/json"

    def get_extra_attributes(
        self,
        response: Any,
    ) -> Iterator[Tuple[str, AttributeValue]]:
        yield from self._get_attributes_from_usage(getattr(response, "usage", None))
        if message := getattr(response, "message", None):
            for key, value in self._get_attributes_from_response_message(message):
                yield f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{key}", value
        if (finish_reason := getattr(response, "finish_reason", None)) is not None:
            yield SpanAttributes.LLM_FINISH_REASON, finish_reason

    def get_extra_attributes_from_embed_response(
        self,
        response: Any,
    ) -> Iterator[Tuple[str, AttributeValue]]:
        if meta := _get_attribute(response, "meta"):
            yield from self._get_attributes_from_usage(meta)
        embeddings = _get_attribute(response, "embeddings")
        for index, vector in enumerate(_embedding_vectors(embeddings)):
            yield (
                f"{SpanAttributes.EMBEDDING_EMBEDDINGS}.{index}."
                f"{EmbeddingAttributes.EMBEDDING_VECTOR}",
                vector,
            )

    def _get_attributes_from_response_message(
        self,
        message: Any,
    ) -> Iterator[Tuple[str, AttributeValue]]:
        if role := getattr(message, "role", None):
            yield MessageAttributes.MESSAGE_ROLE, role
        if (content := getattr(message, "content", None)) is not None:
            if text := _content_text(content):
                yield MessageAttributes.MESSAGE_CONTENT, text
        if (tool_calls := getattr(message, "tool_calls", None)) and isinstance(
            tool_calls, Iterable
        ):
            for index, tool_call in enumerate(tool_calls):
                if (tool_call_id := getattr(tool_call, "id", None)) is not None:
                    yield (
                        f"{MessageAttributes.MESSAGE_TOOL_CALLS}.{index}."
                        f"{ToolCallAttributes.TOOL_CALL_ID}",
                        tool_call_id,
                    )
                if function := getattr(tool_call, "function", None):
                    if name := getattr(function, "name", None):
                        yield (
                            f"{MessageAttributes.MESSAGE_TOOL_CALLS}.{index}."
                            f"{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}",
                            name,
                        )
                    if (arguments := getattr(function, "arguments", None)) is not None:
                        yield (
                            f"{MessageAttributes.MESSAGE_TOOL_CALLS}.{index}."
                            f"{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}",
                            arguments if isinstance(arguments, str) else safe_json_dumps(arguments),
                        )

    def _get_attributes_from_usage(
        self,
        usage: Any,
    ) -> Iterator[Tuple[str, AttributeValue]]:
        # Cohere nests token counts under ``usage.tokens``.
        tokens = getattr(usage, "tokens", None)
        if tokens is None:
            return
        prompt_tokens = getattr(tokens, "input_tokens", None)
        completion_tokens = getattr(tokens, "output_tokens", None)
        if prompt_tokens is not None:
            yield SpanAttributes.LLM_TOKEN_COUNT_PROMPT, int(prompt_tokens)
        if completion_tokens is not None:
            yield SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, int(completion_tokens)
        if prompt_tokens is not None or completion_tokens is not None:
            yield (
                SpanAttributes.LLM_TOKEN_COUNT_TOTAL,
                int(prompt_tokens or 0) + int(completion_tokens or 0),
            )


def _content_text(content: Any) -> str:
    """Collapse Cohere content (a string or a list of typed blocks) into a string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            text = item.get("text") if isinstance(item, Mapping) else getattr(item, "text", None)
            if text:
                parts.append(text)
        return "".join(parts)
    return ""


def _get_attribute(obj: Any, name: str) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def _embedding_vectors(embeddings: Any) -> Iterator[Sequence[float]]:
    """Return Cohere vectors as floats, preferring its lossless response formats."""
    if embeddings is None:
        return

    float_vectors = _get_attribute(embeddings, "float_") or _get_attribute(embeddings, "float")
    if float_vectors is not None:
        yield from _numeric_vectors(float_vectors)
        return

    if base64_vectors := _get_attribute(embeddings, "base64"):
        for encoded_vector in base64_vectors:
            if not isinstance(encoded_vector, str):
                continue
            try:
                raw_vector = base64.b64decode(encoded_vector, validate=True)
                if not raw_vector or len(raw_vector) % 4:
                    continue
                size = len(raw_vector) // 4
                yield struct.unpack(f"<{size}f", raw_vector)
            except (ValueError, struct.error):
                logger.exception("Failed to decode a base64 Cohere embedding")
        return

    # Quantized responses are still embedding vectors. Converting their numeric
    # components to floats keeps the semantic-convention value type consistent.
    for embedding_type in ("int8", "uint8", "binary", "ubinary"):
        if vectors := _get_attribute(embeddings, embedding_type):
            yield from _numeric_vectors(vectors)
            return


def _numeric_vectors(vectors: Any) -> Iterator[Sequence[float]]:
    if not isinstance(vectors, Sequence) or isinstance(vectors, (str, bytes)):
        return
    for vector in vectors:
        if not isinstance(vector, Sequence) or isinstance(vector, (str, bytes)):
            continue
        try:
            yield tuple(float(component) for component in vector)
        except (TypeError, ValueError):
            logger.exception("Failed to convert a Cohere embedding to floats")

import logging
from enum import Enum
from typing import Any, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from opentelemetry.util.types import AttributeValue

from openinference.instrumentation import (
    get_input_attributes,
    get_reranker_attributes,
    safe_json_dumps,
)
from openinference.semconv.trace import (
    EmbeddingAttributes,
    MessageAttributes,
    OpenInferenceLLMProviderValues,
    OpenInferenceLLMSystemValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolAttributes,
    ToolCallAttributes,
)

__all__ = ("_RequestAttributesExtractor",)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class _RequestAttributesExtractor:
    __slots__ = ()

    def get_attributes_from_request(
        self,
        request_parameters: Mapping[str, Any],
    ) -> Iterator[Tuple[str, AttributeValue]]:
        yield SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.LLM.value
        yield SpanAttributes.LLM_PROVIDER, OpenInferenceLLMProviderValues.COHERE.value
        yield SpanAttributes.LLM_SYSTEM, OpenInferenceLLMSystemValues.COHERE.value
        # The chat response does not echo the model, so it is read from the request.
        # Recording it here rather than from the response also keeps it on error spans.
        if isinstance(request_parameters, Mapping) and (model := request_parameters.get("model")):
            yield SpanAttributes.LLM_MODEL_NAME, model
        try:
            yield from get_input_attributes(request_parameters).items()
        except Exception:
            logger.exception(
                f"Failed to get input attributes from request parameters of "
                f"type {type(request_parameters)}"
            )

    def get_extra_attributes_from_request(
        self,
        request_parameters: Mapping[str, Any],
    ) -> Iterator[Tuple[str, AttributeValue]]:
        if not isinstance(request_parameters, Mapping):
            return
        invocation_params = dict(request_parameters)
        invocation_params.pop("messages", None)  # Remove LLM input messages
        invocation_params.pop("model", None)  # Captured separately as the model name

        if (tools := _replayable_sequence(invocation_params.pop("tools", None))) is not None:
            for i, tool in enumerate(tools):
                yield (
                    f"{SpanAttributes.LLM_TOOLS}.{i}.{ToolAttributes.TOOL_JSON_SCHEMA}",
                    safe_json_dumps(_as_jsonable(tool)),
                )

        yield SpanAttributes.LLM_INVOCATION_PARAMETERS, safe_json_dumps(invocation_params)

        if (input_messages := _replayable_sequence(request_parameters.get("messages"))) is not None:
            for index, input_message in reversed(list(enumerate(input_messages))):
                # Use reversed() to get the last message first. This is because OTEL has a default
                # limit of 128 attributes per span, and flattening increases the number of
                # attributes very quickly.
                for key, value in self._get_attributes_from_message_param(input_message):
                    yield f"{SpanAttributes.LLM_INPUT_MESSAGES}.{index}.{key}", value

    def _get_attributes_from_message_param(
        self,
        message: Any,
    ) -> Iterator[Tuple[str, AttributeValue]]:
        if role := get_attribute(message, "role"):
            yield (
                MessageAttributes.MESSAGE_ROLE,
                role.value if isinstance(role, Enum) else role,
            )
        if (content := get_attribute(message, "content")) is not None:
            if text := _content_text(content):
                yield MessageAttributes.MESSAGE_CONTENT, text
        # Links a tool result message back to the call that produced it.
        if (tool_call_id := get_attribute(message, "tool_call_id")) is not None:
            yield MessageAttributes.MESSAGE_TOOL_CALL_ID, tool_call_id

        if (tool_calls := get_attribute(message, "tool_calls")) and isinstance(
            tool_calls, Iterable
        ):
            for index, tool_call in enumerate(tool_calls):
                if (tool_call_id := get_attribute(tool_call, "id")) is not None:
                    yield (
                        f"{MessageAttributes.MESSAGE_TOOL_CALLS}.{index}."
                        f"{ToolCallAttributes.TOOL_CALL_ID}",
                        tool_call_id,
                    )
                if function := get_attribute(tool_call, "function"):
                    if name := get_attribute(function, "name"):
                        yield (
                            f"{MessageAttributes.MESSAGE_TOOL_CALLS}.{index}."
                            f"{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}",
                            name,
                        )
                    if (arguments := get_attribute(function, "arguments")) is not None:
                        yield (
                            f"{MessageAttributes.MESSAGE_TOOL_CALLS}.{index}."
                            f"{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}",
                            arguments if isinstance(arguments, str) else safe_json_dumps(arguments),
                        )

    def get_attributes_from_embed_request(
        self,
        request_parameters: Mapping[str, Any],
    ) -> Iterator[Tuple[str, AttributeValue]]:
        yield SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.EMBEDDING.value
        if isinstance(request_parameters, Mapping) and (model := request_parameters.get("model")):
            yield SpanAttributes.EMBEDDING_MODEL_NAME, model

        invocation_parameters = dict(request_parameters)
        for input_parameter in ("texts", "images", "inputs"):
            invocation_parameters.pop(input_parameter, None)
        yield (
            SpanAttributes.EMBEDDING_INVOCATION_PARAMETERS,
            safe_json_dumps(invocation_parameters),
        )

        if (texts := _replayable_sequence(request_parameters.get("texts"))) is not None:
            for index, text in enumerate(texts):
                if isinstance(text, str):
                    yield (
                        f"{SpanAttributes.EMBEDDING_EMBEDDINGS}.{index}."
                        f"{EmbeddingAttributes.EMBEDDING_TEXT}",
                        text,
                    )
        elif (inputs := _replayable_sequence(request_parameters.get("inputs"))) is not None:
            for index, embed_input in enumerate(inputs):
                if text := _embed_input_text(embed_input):
                    yield (
                        f"{SpanAttributes.EMBEDDING_EMBEDDINGS}.{index}."
                        f"{EmbeddingAttributes.EMBEDDING_TEXT}",
                        text,
                    )
        try:
            yield from get_input_attributes(request_parameters).items()
        except Exception:
            logger.exception(
                f"Failed to get input attributes from embed request parameters of "
                f"type {type(request_parameters)}"
            )

    def get_attributes_from_rerank_request(
        self,
        request_parameters: Mapping[str, Any],
    ) -> Iterator[Tuple[str, AttributeValue]]:
        yield SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.RERANKER.value
        query = request_parameters.get("query")
        model_name = request_parameters.get("model")
        top_k = request_parameters.get("top_n")
        yield from get_reranker_attributes(
            query=query if isinstance(query, str) else None,
            model_name=model_name if isinstance(model_name, str) else None,
            top_k=top_k if isinstance(top_k, int) and not isinstance(top_k, bool) else None,
        ).items()
        try:
            yield from get_input_attributes(request_parameters).items()
        except Exception:
            logger.exception(
                f"Failed to get input attributes from rerank request parameters of "
                f"type {type(request_parameters)}"
            )

    def get_extra_attributes_from_rerank_request(
        self,
        request_parameters: Mapping[str, Any],
    ) -> Iterator[Tuple[str, AttributeValue]]:
        documents = _replayable_sequence(request_parameters.get("documents"))
        if documents is None:
            return
        yield from get_reranker_attributes(
            input_documents=_as_reranker_documents(documents)
        ).items()


def get_attribute(obj: Any, attr_name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(attr_name, default)
    return getattr(obj, attr_name, default)


def _replayable_sequence(obj: Any) -> Optional[Sequence[Any]]:
    """Return ``obj`` if it can be iterated without consuming the caller's value.

    Attributes are extracted before the wrapped call runs, so iterating a one-shot
    iterator here would hand the SDK an exhausted argument. Only re-iterable
    sequences are safe to read; a generator is left untraced rather than drained.
    """
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        return obj
    return None


def _as_reranker_documents(documents: Sequence[Any]) -> List[Any]:
    """Convert Cohere's string or structured documents to OI document mappings."""
    converted: List[Any] = []
    for document in documents:
        if isinstance(document, str):
            converted.append({"content": document})
            continue

        jsonable_document = _as_jsonable(document)
        converted_document = {"content": safe_json_dumps(jsonable_document)}
        if isinstance(jsonable_document, Mapping):
            if (document_id := jsonable_document.get("id")) is not None:
                converted_document["id"] = document_id
            if (metadata := jsonable_document.get("metadata")) is not None:
                converted_document["metadata"] = metadata
        converted.append(converted_document)
    return converted


def _as_jsonable(obj: Any) -> Any:
    """Unwrap a pydantic model so it serializes as JSON rather than as its repr.

    ``safe_json_dumps`` falls back to ``str`` for unknown types, which would turn
    the SDK's typed objects (e.g. ``ToolV2``) into Python reprs.
    """
    if callable(model_dump := getattr(obj, "model_dump", None)):
        try:
            return model_dump(exclude_unset=True)
        except Exception:
            logger.exception(f"Failed to dump object of type {type(obj)}")
    return obj


def _content_text(content: Any) -> str:
    """Cohere content may be a plain string or a list of typed content blocks.

    Collapse either form into a single string. Blocks carrying ``text`` contribute
    it directly; other block types (e.g. documents in a tool result) are serialized
    so their payload still reaches the trace instead of being dropped.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence):
        parts: List[str] = []
        for item in content:
            if text := get_attribute(item, "text"):
                parts.append(text if isinstance(text, str) else safe_json_dumps(text))
            elif item is not None:
                parts.append(safe_json_dumps(_as_jsonable(item)))
        return "".join(parts)
    return ""


def _embed_input_text(embed_input: Any) -> str:
    """Collect only text blocks from a multimodal Cohere ``EmbedInput``."""
    content = get_attribute(embed_input, "content")
    if not isinstance(content, Sequence) or isinstance(content, (str, bytes)):
        return ""
    parts: List[str] = []
    for item in content:
        if isinstance(text := get_attribute(item, "text"), str):
            parts.append(text)
    return "".join(parts)

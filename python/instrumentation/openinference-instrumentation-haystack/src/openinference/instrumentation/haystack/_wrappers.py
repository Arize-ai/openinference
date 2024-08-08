from abc import ABC
from enum import Enum
from typing import Any, Callable, Iterator, List, Mapping, Tuple

import opentelemetry.context as context_api
from openinference.instrumentation import get_attributes_from_context, safe_json_dumps
from openinference.semconv.trace import (
    DocumentAttributes,
    EmbeddingAttributes,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)
from opentelemetry import trace as trace_api
from opentelemetry.util.types import AttributeValue


def _flatten(mapping: Mapping[str, Any]) -> Iterator[Tuple[str, AttributeValue]]:
    for key, value in mapping.items():
        if value is None:
            continue
        if isinstance(value, Mapping):
            for sub_key, sub_value in _flatten(value):
                yield f"{key}.{sub_key}", sub_value
        elif isinstance(value, List) and any(isinstance(item, Mapping) for item in value):
            for index, sub_mapping in enumerate(value):
                for sub_key, sub_value in _flatten(sub_mapping):
                    yield f"{key}.{index}.{sub_key}", sub_value
        else:
            if isinstance(value, Enum):
                value = value.value
            yield key, value


class _WithTracer(ABC):
    """
    Base class for wrappers that need a tracer.
    """

    def __init__(self, tracer: trace_api.Tracer, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._tracer = tracer


class _ComponentWrapper(_WithTracer):
    """
    Wrapper for the pipeline processing
    Captures all calls to the pipeline
    """

    def __call__(
            self,
            wrapped: Callable[..., Any],
            instance: Any,
            args: Tuple[Any, ...],
            kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        component_type = args[0]
        input_data = args[1]

        # Diving into the instance to retrieve the Component instance
        component = instance.graph.nodes._nodes[component_type]["instance"]

        component_name = component.__class__.__name__

        # Prepare invocation parameters by merging args and kwargs
        invocation_parameters = {}
        for arg in args:
            if arg and isinstance(arg, dict):
                invocation_parameters.update(arg)
        invocation_parameters.update(kwargs)

        with self._tracer.start_as_current_span(name=component_name) as span:
            span.set_attributes(dict(get_attributes_from_context()))
            if "Generator" in component_name or component_name == "VertexAIImageQA":
                span.set_attributes(
                    dict(
                        _flatten(
                            {
                                SpanAttributes.OPENINFERENCE_SPAN_KIND: LLM,
                                SpanAttributes.INPUT_VALUE: safe_json_dumps(input_data),
                                SpanAttributes.INPUT_MIME_TYPE: OpenInferenceMimeTypeValues.JSON,
                            }
                        )
                    )
                )
                if "prompt_builder" in str(instance):
                    if "ChatPromptBuilder" in str(instance):
                        span.set_attributes(
                            dict(
                                _flatten(
                                    {
                                        SpanAttributes.LLM_INPUT_MESSAGES: input_data["messages"],
                                    }
                                )
                            )
                        )
                    else:
                        span.set_attribute(
                            SpanAttributes.LLM_PROMPT_TEMPLATE,
                            instance.graph.nodes._nodes["prompt_builder"][
                                "instance"
                            ]._template_string,
                        )

            elif "Embedder" in component_name:
                span.set_attributes(
                    dict(
                        _flatten(
                            {
                                SpanAttributes.EMBEDDING_MODEL_NAME: component.model,
                                SpanAttributes.OPENINFERENCE_SPAN_KIND: EMBEDDING,
                                SpanAttributes.INPUT_VALUE: safe_json_dumps(invocation_parameters),
                                SpanAttributes.INPUT_MIME_TYPE: OpenInferenceMimeTypeValues.JSON,
                            }
                        )
                    )
                )
            elif "Retriever" in component_name:
                span.set_attributes(
                    dict(
                        _flatten(
                            {
                                SpanAttributes.OPENINFERENCE_SPAN_KIND: RETRIEVER,
                            }
                        )
                    )
                )
                if "query_embedding" in input_data:
                    emb_len = len(input_data["query_embedding"])
                    span.set_attributes(
                        dict(
                            _flatten(
                                {
                                    SpanAttributes.INPUT_MIME_TYPE: TEXT,
                                    SpanAttributes.INPUT_VALUE: f"<{emb_len} dimensional vector>",
                                }
                            )
                        )
                    )
                if "query" in input_data:
                    span.set_attributes(
                        dict(
                            _flatten(
                                {
                                    SpanAttributes.INPUT_MIME_TYPE: TEXT,
                                    SpanAttributes.INPUT_VALUE: input_data["query"],
                                }
                            )
                        )
                    )
            elif "PromptBuilder" in component_name:
                span.set_attributes(
                    dict(
                        _flatten(
                            {
                                SpanAttributes.OPENINFERENCE_SPAN_KIND: CHAIN,
                                SpanAttributes.INPUT_VALUE: safe_json_dumps(invocation_parameters),
                                SpanAttributes.INPUT_MIME_TYPE: OpenInferenceMimeTypeValues.JSON,
                            }
                        )
                    )
                )
                if "ChatPromptBuilder" in component_name:
                    temp_vars = safe_json_dumps(input_data["template_variables"])
                    msg_conts = [m.content for m in input_data["template"]]
                    span.set_attributes(
                        dict(
                            _flatten(
                                {
                                    SpanAttributes.LLM_INPUT_MESSAGES: msg_conts,
                                    SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES: temp_vars,
                                }
                            )
                        )
                    )
                else:
                    span.set_attribute(
                        SpanAttributes.LLM_PROMPT_TEMPLATE,
                        instance.graph.nodes._nodes["prompt_builder"]["instance"]._template_string,
                    )
                if "documents" in invocation_parameters:
                    span.set_attributes(
                        dict(
                            _flatten(
                                {
                                    SpanAttributes.RETRIEVAL_DOCUMENTS: safe_json_dumps(
                                        invocation_parameters["documents"]
                                    ),
                                }
                            )
                        )
                    )
            else:
                span.set_attributes(
                    dict(
                        _flatten(
                            {
                                SpanAttributes.OPENINFERENCE_SPAN_KIND: CHAIN,
                                SpanAttributes.INPUT_VALUE: safe_json_dumps(invocation_parameters),
                                SpanAttributes.INPUT_MIME_TYPE: OpenInferenceMimeTypeValues.JSON,
                            }
                        )
                    )
                )

            try:
                response = wrapped(*args, **kwargs)
            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                raise
            span.set_status(trace_api.StatusCode.OK)

            if "Generator" in component_name or component_name == "VertexAIImageQA":
                if "Chat" in component.__class__.__name__:
                    replies = response.get("replies")
                    if replies is None or len(replies) == 0:
                        pass
                    reply = replies[0]
                    usage = reply.meta.get("usage", {})
                    if "meta" in str(reply):
                        span.set_attributes(
                            dict(
                                _flatten(
                                    {
                                        SpanAttributes.LLM_OUTPUT_MESSAGES: response["replies"],
                                        SpanAttributes.LLM_MODEL_NAME: reply.meta["model"],
                                        SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: usage[
                                            "completion_tokens"
                                        ],
                                        SpanAttributes.LLM_TOKEN_COUNT_PROMPT: usage[
                                            "prompt_tokens"
                                        ],
                                        SpanAttributes.LLM_TOKEN_COUNT_TOTAL: usage["total_tokens"],
                                    }
                                )
                            )
                        )
                else:
                    span.set_attributes(
                        dict(
                            _flatten(
                                {
                                    SpanAttributes.LLM_MODEL_NAME: response["meta"][0]["model"],
                                    SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: response["meta"][0][
                                        "usage"
                                    ]["completion_tokens"],
                                    SpanAttributes.LLM_TOKEN_COUNT_PROMPT: response["meta"][0][
                                        "usage"
                                    ]["prompt_tokens"],
                                    SpanAttributes.LLM_TOKEN_COUNT_TOTAL: response["meta"][0][
                                        "usage"
                                    ]["total_tokens"],
                                    SpanAttributes.LLM_OUTPUT_MESSAGES: response["replies"],
                                }
                            )
                        )
                    )
            elif "Embedder" in component_name:
                emb_len = len(response["embedding"])
                emb_vec_0 = f"{SpanAttributes.EMBEDDING_EMBEDDINGS}.0."

                span.set_attributes(
                    dict(
                        _flatten(
                            {
                                f"{emb_vec_0}{EMBEDDING_VECTOR}": f"<{emb_len} dimensional vector>",
                                f"{emb_vec_0}{EMBEDDING_TEXT}": invocation_parameters["text"],
                            }
                        )
                    )
                )
            elif "Retriever" in component_name:
                if "documents" in response:
                    span.set_attributes(
                        dict(
                            _flatten(
                                {
                                    SpanAttributes.OUTPUT_VALUE: safe_json_dumps(
                                        response["documents"]
                                    ),
                                    SpanAttributes.OUTPUT_MIME_TYPE: JSON,
                                }
                            )
                        )
                    )

                for i, document in enumerate(response["documents"]):
                    span.set_attributes(
                        {
                            f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.{i}."
                            f"{DocumentAttributes.DOCUMENT_CONTENT}": document.content,
                            f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.{i}."
                            f"{DocumentAttributes.DOCUMENT_ID}": document.id,
                            f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.{i}."
                            f"{DocumentAttributes.DOCUMENT_SCORE}": document.score,
                            f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.{i}."
                            f"{DocumentAttributes.DOCUMENT_METADATA}": safe_json_dumps(
                                document.meta
                            ),
                        }
                    )
            else:
                span.set_attributes(
                    dict(
                        _flatten(
                            {
                                SpanAttributes.OUTPUT_VALUE: safe_json_dumps(response),
                                SpanAttributes.OUTPUT_MIME_TYPE: OpenInferenceMimeTypeValues.JSON,
                            }
                        )
                    )
                )
        return response


class _PipelineWrapper(_WithTracer):
    """
    Wrapper for the pipeline processing
    Captures all calls to the pipeline
    """

    def __call__(
            self,
            wrapped: Callable[..., Any],
            instance: Any,
            args: Tuple[Any, ...],
            kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        # Prepare invocation parameters by merging args and kwargs
        invocation_parameters = {}
        for arg in args:
            if arg and isinstance(arg, dict):
                invocation_parameters.update(arg)
        invocation_parameters.update(kwargs)

        span_name = "Pipeline"
        with self._tracer.start_as_current_span(
                span_name,
                attributes=dict(
                    _flatten(
                        {
                            SpanAttributes.OPENINFERENCE_SPAN_KIND: CHAIN,
                            SpanAttributes.INPUT_VALUE: safe_json_dumps(invocation_parameters),
                            SpanAttributes.INPUT_MIME_TYPE: OpenInferenceMimeTypeValues.JSON,
                        }
                    )
                ),
        ) as span:
            span.set_attributes(dict(get_attributes_from_context()))
            try:
                response = wrapped(*args, **kwargs)
            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                raise
            span.set_attributes(
                dict(
                    _flatten(
                        {
                            SpanAttributes.OUTPUT_VALUE: safe_json_dumps(response),
                            SpanAttributes.OUTPUT_MIME_TYPE: OpenInferenceMimeTypeValues.JSON,
                        }
                    )
                )
            )
            span.set_status(trace_api.StatusCode.OK)

        return response


CHAIN = OpenInferenceSpanKindValues.CHAIN
RETRIEVER = OpenInferenceSpanKindValues.RETRIEVER
EMBEDDING = OpenInferenceSpanKindValues.EMBEDDING
JSON = OpenInferenceMimeTypeValues.JSON
TEXT = OpenInferenceMimeTypeValues.TEXT
LLM = OpenInferenceSpanKindValues.LLM
EMBEDDING_VECTOR = EmbeddingAttributes.EMBEDDING_VECTOR
EMBEDDING_TEXT = EmbeddingAttributes.EMBEDDING_TEXT


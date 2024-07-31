from abc import ABC
from enum import Enum
from typing import Any, Callable, Iterator, List, Mapping, Tuple

import opentelemetry.context as context_api
from openinference.instrumentation import get_attributes_from_context, safe_json_dumps
from openinference.semconv.trace import (
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



class _CompletionsWrapper(_WithTracer):
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
        llm_invocation_params = kwargs
        llm_messages = dict(kwargs).pop("messages", None)
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        # Prepare invocation parameters by merging args and kwargs
        invocation_parameters = {}
        for arg in args:
            if arg and isinstance(arg, dict):
                invocation_parameters.update(arg)
        invocation_parameters.update(kwargs)

        span_name = "Completions"
        with self._tracer.start_as_current_span(
                span_name,
                attributes= {}
        ) as span:
            span.set_attributes(dict(get_attributes_from_context()))

            attributes=dict(
                _flatten(
                    {
                        SpanAttributes.OPENINFERENCE_SPAN_KIND: LLM,
                        SpanAttributes.LLM_INPUT_MESSAGES: llm_messages,
                        SpanAttributes.LLM_INVOCATION_PARAMETERS: safe_json_dumps(llm_invocation_params),
                        SpanAttributes.LLM_MODEL_NAME: llm_invocation_params.get("model"),
                        SpanAttributes.INPUT_MIME_TYPE: OpenInferenceMimeTypeValues.JSON,
                    }
                )
            )
            try:
                response = wrapped(*args, **kwargs)
            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                raise
            attributes.update(
                _flatten(
                    {
                        SpanAttributes.OUTPUT_VALUE: response.choices[0].message.content,
                        SpanAttributes.OUTPUT_MIME_TYPE: OpenInferenceMimeTypeValues.TEXT,
                        SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: response.usage.completion_tokens,
                        SpanAttributes.LLM_TOKEN_COUNT_PROMPT: response.usage.prompt_tokens,
                        SpanAttributes.LLM_TOKEN_COUNT_TOTAL: response.usage.total_tokens,
                    }
                )
            )

            span.set_attributes(attributes)
            span.set_status(trace_api.StatusCode.OK)

        return response

class _BothCompletionsWrapper(_WithTracer):
    """
    Wrapper for the pipeline processing
    Captures all calls to the pipeline
    """

    async def __call__(
            self,
            wrapped: Callable[..., Any],
            instance: Any,
            args: Tuple[Any, ...],
            kwargs: Mapping[str, Any],
    ) -> Any:
        comp_type = instance.__class__.__name__
        llm_invocation_params = kwargs
        llm_messages = dict(kwargs).pop("messages", None)
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            if comp_type == "Completions":
                return wrapped(*args, **kwargs)
            elif comp_type == "AsyncCompletions":
                return await wrapped(*args, **kwargs)

        # Prepare invocation parameters by merging args and kwargs
        invocation_parameters = {}
        for arg in args:
            if arg and isinstance(arg, dict):
                invocation_parameters.update(arg)
        invocation_parameters.update(kwargs)

        if comp_type == "Completions":
            span_name = "Completions"
        elif comp_type == "AsyncCompletions":
            span_name = "AsyncCompletions"
        with self._tracer.start_as_current_span(
                span_name,
                attributes= {}
        ) as span:
            span.set_attributes(dict(get_attributes_from_context()))

            attributes=dict(
                _flatten(
                    {
                        SpanAttributes.OPENINFERENCE_SPAN_KIND: LLM,
                        SpanAttributes.LLM_INPUT_MESSAGES: llm_messages,
                        SpanAttributes.LLM_INVOCATION_PARAMETERS: safe_json_dumps(llm_invocation_params),
                        SpanAttributes.LLM_MODEL_NAME: llm_invocation_params.get("model"),
                        SpanAttributes.INPUT_MIME_TYPE: OpenInferenceMimeTypeValues.JSON,
                    }
                )
            )
            try:
                if comp_type == "Completions":
                    response = wrapped(*args, **kwargs)
                elif comp_type == "AsyncCompletions":
                    response = await wrapped(*args, **kwargs)
            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                raise
            attributes.update(
                _flatten(
                    {
                        SpanAttributes.OUTPUT_VALUE: response.choices[0].message.content,
                        SpanAttributes.OUTPUT_MIME_TYPE: OpenInferenceMimeTypeValues.TEXT,
                        SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: response.usage.completion_tokens,
                        SpanAttributes.LLM_TOKEN_COUNT_PROMPT: response.usage.prompt_tokens,
                        SpanAttributes.LLM_TOKEN_COUNT_TOTAL: response.usage.total_tokens,
                    }
                )
            )

            span.set_attributes(attributes)
            span.set_status(trace_api.StatusCode.OK)

        return response


CHAIN = OpenInferenceSpanKindValues.CHAIN
RETRIEVER = OpenInferenceSpanKindValues.RETRIEVER
EMBEDDING = OpenInferenceSpanKindValues.EMBEDDING
LLM = OpenInferenceSpanKindValues.LLM
EMBEDDING_VECTOR = EmbeddingAttributes.EMBEDDING_VECTOR
EMBEDDING_TEXT = EmbeddingAttributes.EMBEDDING_TEXT

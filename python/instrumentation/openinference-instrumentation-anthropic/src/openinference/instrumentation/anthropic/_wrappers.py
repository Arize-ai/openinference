from abc import ABC
from enum import Enum
from typing import Any, Callable, Iterator, List, Mapping, Tuple

import opentelemetry.context as context_api
from openinference.instrumentation import get_attributes_from_context, safe_json_dumps
from openinference.semconv.trace import (
    EmbeddingAttributes,
    MessageAttributes,
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


class _CompletionsGenerationWrapper(_WithTracer):
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

        llm_invocation_params = kwargs
        llm_prompt = dict(kwargs).pop("prompt", None)

        # Prepare invocation parameters by merging args and kwargs
        invocation_parameters = {}
        for arg in args:
            if arg and isinstance(arg, dict):
                invocation_parameters.update(arg)
        invocation_parameters.update(kwargs)

        span_name = "Completions"
        with self._tracer.start_as_current_span(
            span_name,
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            span.set_attributes(dict(get_attributes_from_context()))

            span.set_attributes(
                dict(
                    _flatten(
                        {
                            SpanAttributes.OPENINFERENCE_SPAN_KIND: LLM,
                            SpanAttributes.LLM_INPUT_MESSAGES: llm_prompt,
                            SpanAttributes.LLM_INVOCATION_PARAMETERS: safe_json_dumps(
                                llm_invocation_params
                            ),
                            SpanAttributes.LLM_MODEL_NAME: llm_invocation_params.get("model"),
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
            span.set_attributes(
                dict(
                    _flatten(
                        {
                            f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}": response.completion,
                            f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}": "assistant",
                            OUTPUT_VALUE: response.completion,
                            OUTPUT_MIME_TYPE: OpenInferenceMimeTypeValues.TEXT,
                        }
                    )
                )
            )

            span.set_status(trace_api.StatusCode.OK)
        return response


class _AsyncCompletionsGenerationWrapper(_WithTracer):
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
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return await wrapped(*args, **kwargs)

        llm_invocation_params = kwargs
        llm_prompt = dict(kwargs).pop("prompt", None)

        # Prepare invocation parameters by merging args and kwargs
        invocation_parameters = {}
        for arg in args:
            if arg and isinstance(arg, dict):
                invocation_parameters.update(arg)
        invocation_parameters.update(kwargs)

        span_name = "AsyncCompletions"
        with self._tracer.start_as_current_span(
            span_name,
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            span.set_attributes(dict(get_attributes_from_context()))

            span.set_attributes(
                dict(
                    _flatten(
                        {
                            SpanAttributes.OPENINFERENCE_SPAN_KIND: LLM,
                            SpanAttributes.LLM_INPUT_MESSAGES: llm_prompt,
                            SpanAttributes.LLM_INVOCATION_PARAMETERS: safe_json_dumps(
                                llm_invocation_params
                            ),
                            SpanAttributes.LLM_MODEL_NAME: llm_invocation_params.get("model"),
                            SpanAttributes.INPUT_MIME_TYPE: OpenInferenceMimeTypeValues.JSON,
                        }
                    )
                )
            )
            try:
                response = await wrapped(*args, **kwargs)
            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                raise
            span.set_attributes(
                dict(
                    _flatten(
                        {
                            f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}": response.completion,
                            f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}": "assistant",
                            OUTPUT_VALUE: response.completion,
                            OUTPUT_MIME_TYPE: OpenInferenceMimeTypeValues.TEXT,
                        }
                    )
                )
            )

            span.set_status(trace_api.StatusCode.OK)

        return response


class _CompletionsChatWrapper(_WithTracer):
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

        llm_invocation_params = kwargs
        llm_input_messages = dict(kwargs).pop("messages", None)

        # Prepare invocation parameters by merging args and kwargs
        invocation_parameters = {}
        for arg in args:
            if arg and isinstance(arg, dict):
                invocation_parameters.update(arg)
        invocation_parameters.update(kwargs)

        span_name = "Messages"
        with self._tracer.start_as_current_span(
            span_name,
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            span.set_attributes(dict(get_attributes_from_context()))

            span.set_attributes(
                dict(
                    _flatten(
                        {
                            SpanAttributes.OPENINFERENCE_SPAN_KIND: LLM,
                            SpanAttributes.LLM_INPUT_MESSAGES: llm_input_messages,
                            SpanAttributes.LLM_INVOCATION_PARAMETERS: safe_json_dumps(
                                llm_invocation_params
                            ),
                            SpanAttributes.LLM_MODEL_NAME: llm_invocation_params.get("model"),
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
            span.set_attributes(
                dict(
                    _flatten(
                        {
                            f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}": response.content[0],
                            f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}": response.role,
                            OUTPUT_VALUE: response.content[0],
                            OUTPUT_MIME_TYPE: OpenInferenceMimeTypeValues.TEXT,
                        }
                    )
                )
            )

            span.set_status(trace_api.StatusCode.OK)
        return response


class _AsyncCompletionsChatWrapper(_WithTracer):
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
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return await wrapped(*args, **kwargs)

        llm_invocation_params = kwargs
        llm_input_messages = dict(kwargs).pop("messages", None)

        # Prepare invocation parameters by merging args and kwargs
        invocation_parameters = {}
        for arg in args:
            if arg and isinstance(arg, dict):
                invocation_parameters.update(arg)
        invocation_parameters.update(kwargs)

        span_name = "AsyncMessages"
        with self._tracer.start_as_current_span(
            span_name,
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            span.set_attributes(dict(get_attributes_from_context()))

            span.set_attributes(
                dict(
                    _flatten(
                        {
                            SpanAttributes.OPENINFERENCE_SPAN_KIND: LLM,
                            SpanAttributes.LLM_INPUT_MESSAGES: llm_input_messages,
                            SpanAttributes.LLM_INVOCATION_PARAMETERS: safe_json_dumps(
                                llm_invocation_params
                            ),
                            SpanAttributes.LLM_MODEL_NAME: llm_invocation_params.get("model"),
                            SpanAttributes.INPUT_MIME_TYPE: OpenInferenceMimeTypeValues.JSON,
                        }
                    )
                )
            )
            try:
                response = await wrapped(*args, **kwargs)
            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                raise
            span.set_attributes(
                dict(
                    _flatten(
                        {
                            f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}": response.content[0],
                            f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}": response.role,
                            OUTPUT_VALUE: response.content[0],
                            OUTPUT_MIME_TYPE: OpenInferenceMimeTypeValues.TEXT,
                        }
                    )
                )
            )

            span.set_status(trace_api.StatusCode.OK)

        return response


CHAIN = OpenInferenceSpanKindValues.CHAIN
RETRIEVER = OpenInferenceSpanKindValues.RETRIEVER
EMBEDDING = OpenInferenceSpanKindValues.EMBEDDING
LLM = OpenInferenceSpanKindValues.LLM
EMBEDDING_VECTOR = EmbeddingAttributes.EMBEDDING_VECTOR
EMBEDDING_TEXT = EmbeddingAttributes.EMBEDDING_TEXT
LLM_TOKEN_COUNT_COMPLETION = SpanAttributes.LLM_TOKEN_COUNT_COMPLETION
LLM_OUTPUT_MESSAGES = SpanAttributes.LLM_OUTPUT_MESSAGES
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
OUTPUT_MIME_TYPE = SpanAttributes.OUTPUT_MIME_TYPE
MESSAGE_CONTENT = MessageAttributes.MESSAGE_CONTENT
MESSAGE_ROLE = MessageAttributes.MESSAGE_ROLE

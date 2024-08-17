from abc import ABC
from enum import Enum
from typing import Any, Callable, Dict, Iterator, List, Mapping, Tuple

import opentelemetry.context as context_api
from openinference.instrumentation import get_attributes_from_context, safe_json_dumps
from openinference.semconv.trace import (
    DocumentAttributes,
    EmbeddingAttributes,
    MessageAttributes,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolCallAttributes,
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
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        arguments = kwargs
        llm_prompt = dict(arguments).pop("prompt", None)

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
                            OPENINFERENCE_SPAN_KIND: LLM,
                            LLM_PROMPTS: [llm_prompt],
                            INPUT_VALUE: safe_json_dumps(arguments),
                            INPUT_MIME_TYPE: OpenInferenceMimeTypeValues.JSON,
                        }
                    )
                )
            )
            if model_name := arguments.get("model"):
                span.set_attribute(LLM_MODEL_NAME, model_name)
            try:
                response = wrapped(*args, **kwargs)
            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                raise
            span.set_status(trace_api.StatusCode.OK)
            span.set_attributes(
                dict(
                    _flatten(
                        {
                            OUTPUT_VALUE: response.model_dump_json(),
                            OUTPUT_MIME_TYPE: JSON,
                        }
                    )
                )
            )

        return response


class _AsyncCompletionsWrapper(_WithTracer):
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

        arguments = kwargs
        llm_prompt = dict(arguments).pop("prompt", None)

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
                            OPENINFERENCE_SPAN_KIND: LLM,
                            LLM_PROMPTS: [llm_prompt],
                            INPUT_VALUE: safe_json_dumps(arguments),
                            INPUT_MIME_TYPE: OpenInferenceMimeTypeValues.JSON,
                        }
                    )
                )
            )
            if model_name := arguments.get("model"):
                span.set_attribute(LLM_MODEL_NAME, model_name)
            try:
                response = await wrapped(*args, **kwargs)
            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                raise
            span.set_status(trace_api.StatusCode.OK)
            span.set_attributes(
                dict(
                    _flatten(
                        {
                            OUTPUT_VALUE: response.to_json(indent=None),
                            OUTPUT_MIME_TYPE: JSON,
                        }
                    )
                )
            )
            return response


class _MessagesWrapper(_WithTracer):
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

        arguments = kwargs

        # llm_invocation_params = kwargs
        llm_input_messages = dict(arguments).pop("messages", None)

        invocation_parameters = _get_invocation_parameters(arguments)

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
                            OPENINFERENCE_SPAN_KIND: LLM,
                            **dict(_get_input_messages(llm_input_messages)),
                            LLM_INVOCATION_PARAMETERS: invocation_parameters,
                            LLM_MODEL_NAME: arguments.get("model"),
                            INPUT_VALUE: safe_json_dumps(arguments),
                            INPUT_MIME_TYPE: OpenInferenceMimeTypeValues.JSON,
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
            span.set_attributes(
                dict(
                    _flatten(
                        {
                            f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}": response.content[0].text,
                            f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}": response.role,
                            LLM_TOKEN_COUNT_PROMPT: response.usage.input_tokens,
                            LLM_TOKEN_COUNT_COMPLETION: response.usage.output_tokens,
                            OUTPUT_VALUE: response.model_dump_json(),
                            OUTPUT_MIME_TYPE: OpenInferenceMimeTypeValues.JSON,
                        }
                    )
                )
            )

        return response


class _AsyncMessagesWrapper(_WithTracer):
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

        arguments = kwargs

        llm_input_messages = dict(arguments).pop("messages", None)

        invocation_parameters = _get_invocation_parameters(arguments)

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
                            OPENINFERENCE_SPAN_KIND: LLM,
                            **dict(_get_input_messages(llm_input_messages)),
                            LLM_INVOCATION_PARAMETERS: invocation_parameters,
                            LLM_MODEL_NAME: arguments.get("model"),
                            INPUT_VALUE: safe_json_dumps(arguments),
                            INPUT_MIME_TYPE: OpenInferenceMimeTypeValues.JSON,
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
            span.set_status(trace_api.StatusCode.OK)
            span.set_attributes(
                dict(
                    _flatten(
                        {
                            f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}": response.content[0].text,
                            f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}": response.role,
                            LLM_TOKEN_COUNT_PROMPT: response.usage.input_tokens,
                            LLM_TOKEN_COUNT_COMPLETION: response.usage.output_tokens,
                            OUTPUT_VALUE: response.model_dump_json(),
                            OUTPUT_MIME_TYPE: OpenInferenceMimeTypeValues.JSON,
                        }
                    )
                )
            )

        return response


def _get_input_messages(messages: List[Dict[str, str]]) -> Any:
    """
    Extracts the messages from the chat response
    """
    for i in range(len(messages)):
        if content := messages[i].get("content"):
            yield f"{LLM_INPUT_MESSAGES}.{i}.{MESSAGE_CONTENT}", content
        if role := messages[i].get("role"):
            yield f"{LLM_INPUT_MESSAGES}.{i}.{MESSAGE_ROLE}", role


def _get_output_message(response: Any) -> Any:
    if response.content:
        yield f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}", response.content[0].text
    if response.role:
        yield f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}", response.role


def _get_invocation_parameters(kwargs: Mapping[str, Any]) -> Any:
    """
    Extracts the invocation parameters from the call
    """
    invocation_parameters = {}
    for key, value in kwargs.items():
        if isinstance(value, dict) and _validate_invocation_parameter(key):
            invocation_parameters[key] = value
    return invocation_parameters


def _validate_invocation_parameter(parameter: Any) -> bool:
    """
    Validates the invocation parameters
    """
    valid_params = (
        "max_tokens",
        "messages",
        "model",
        "metadata",
        "stop_sequences",
        "stream",
        "system",
        "temperature",
        "tool_choice",
        "tools",
        "top_k",
        "top_p",
    )

    return parameter in valid_params


CHAIN = OpenInferenceSpanKindValues.CHAIN.value
EMBEDDING = OpenInferenceSpanKindValues.EMBEDDING.value
LLM = OpenInferenceSpanKindValues.LLM.value
RETRIEVER = OpenInferenceSpanKindValues.RETRIEVER.value

JSON = OpenInferenceMimeTypeValues.JSON.value
TEXT = OpenInferenceMimeTypeValues.TEXT.value

DOCUMENT_CONTENT = DocumentAttributes.DOCUMENT_CONTENT
DOCUMENT_ID = DocumentAttributes.DOCUMENT_ID
DOCUMENT_SCORE = DocumentAttributes.DOCUMENT_SCORE
DOCUMENT_METADATA = DocumentAttributes.DOCUMENT_METADATA
EMBEDDING_EMBEDDINGS = SpanAttributes.EMBEDDING_EMBEDDINGS
EMBEDDING_MODEL_NAME = SpanAttributes.EMBEDDING_MODEL_NAME
EMBEDDING_TEXT = EmbeddingAttributes.EMBEDDING_TEXT
EMBEDDING_VECTOR = EmbeddingAttributes.EMBEDDING_VECTOR
INPUT_MIME_TYPE = SpanAttributes.INPUT_MIME_TYPE
INPUT_VALUE = SpanAttributes.INPUT_VALUE
LLM_INPUT_MESSAGES = SpanAttributes.LLM_INPUT_MESSAGES
LLM_INVOCATION_PARAMETERS = SpanAttributes.LLM_INVOCATION_PARAMETERS
LLM_MODEL_NAME = SpanAttributes.LLM_MODEL_NAME
LLM_OUTPUT_MESSAGES = SpanAttributes.LLM_OUTPUT_MESSAGES
LLM_PROMPTS = SpanAttributes.LLM_PROMPTS
LLM_PROMPT_TEMPLATE = SpanAttributes.LLM_PROMPT_TEMPLATE
LLM_PROMPT_TEMPLATE_VARIABLES = SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES
LLM_PROMPT_TEMPLATE_VERSION = SpanAttributes.LLM_PROMPT_TEMPLATE_VERSION
LLM_TOKEN_COUNT_COMPLETION = SpanAttributes.LLM_TOKEN_COUNT_COMPLETION
LLM_TOKEN_COUNT_PROMPT = SpanAttributes.LLM_TOKEN_COUNT_PROMPT
LLM_TOKEN_COUNT_TOTAL = SpanAttributes.LLM_TOKEN_COUNT_TOTAL
MESSAGE_CONTENT = MessageAttributes.MESSAGE_CONTENT
MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON = MessageAttributes.MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON
MESSAGE_FUNCTION_CALL_NAME = MessageAttributes.MESSAGE_FUNCTION_CALL_NAME
MESSAGE_ROLE = MessageAttributes.MESSAGE_ROLE
MESSAGE_TOOL_CALLS = MessageAttributes.MESSAGE_TOOL_CALLS
METADATA = SpanAttributes.METADATA
OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
OUTPUT_MIME_TYPE = SpanAttributes.OUTPUT_MIME_TYPE
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
RETRIEVAL_DOCUMENTS = SpanAttributes.RETRIEVAL_DOCUMENTS
SESSION_ID = SpanAttributes.SESSION_ID
TAG_TAGS = SpanAttributes.TAG_TAGS
TOOL_CALL_FUNCTION_ARGUMENTS_JSON = ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON
TOOL_CALL_FUNCTION_NAME = ToolCallAttributes.TOOL_CALL_FUNCTION_NAME
USER_ID = SpanAttributes.USER_ID

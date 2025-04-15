from __future__ import annotations

import json
from typing import Any, Callable

import wrapt
from opentelemetry import context as context_api
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.trace import Status, StatusCode, Tracer

from openinference.instrumentation.bedrock._response_accumulator import _ResponseAccumulator
from openinference.instrumentation.bedrock.utils import _EventStream, _use_span
from openinference.instrumentation.bedrock.utils.anthropic._messages import (
    _AnthropicMessagesCallback,
)
from openinference.semconv.trace import (
    ImageAttributes,
    MessageAttributes,
    MessageContentAttributes,
    OpenInferenceLLMProviderValues,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)


class _WithTracer:
    def __init__(self, tracer: Tracer):
        self._tracer = tracer


class _InvokeModelWithResponseStream(_WithTracer):
    _name = "bedrock.invoke_model_with_response_stream"

    @wrapt.decorator  # type: ignore[misc]
    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)
        with self._tracer.start_as_current_span(
            self._name,
            end_on_exit=False,
        ) as span:
            response = wrapped(*args, **kwargs)
            from botocore.eventstream import EventStream

            kwargs["body"] = json.loads(kwargs["body"])
            if isinstance(response["body"], EventStream):
                if "anthropic_version" in kwargs["body"]:
                    response["body"] = _EventStream(
                        response["body"],
                        _AnthropicMessagesCallback(span, kwargs),
                        _use_span(span),
                    )
                    return response
            span.set_attribute(LLM_INVOCATION_PARAMETERS, kwargs["body"])
            span.set_attribute(INPUT_MIME_TYPE, JSON)
            span.set_attribute(INPUT_VALUE, kwargs["body"])
            span.set_attribute(OPENINFERENCE_SPAN_KIND, LLM)
            span.end()
            return response


class _InvokeAgentWithResponseStream(_WithTracer):
    _name = "bedrock_agent.invoke_agent"

    @wrapt.decorator  # type: ignore[misc]
    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)
        # span = self._tracer.start_span(self._name)
        with self._tracer.start_as_current_span(
            self._name,
            end_on_exit=False,
        ) as span:
            attributes = {
                SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.AGENT.value,
                SpanAttributes.LLM_PROVIDER: OpenInferenceLLMProviderValues.AWS.value,
            }
            if input_text := kwargs.get("inputText"):
                attributes[SpanAttributes.INPUT_VALUE] = input_text
            span.set_attributes({k: v for k, v in attributes.items() if v is not None})
            try:
                response = wrapped(*args, **kwargs)
                response["completion"] = _EventStream(
                    response["completion"],
                    _ResponseAccumulator(span, self._tracer, kwargs),
                    _use_span(span),
                )
                return response
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.end()
                raise e


IMAGE_URL = ImageAttributes.IMAGE_URL
INPUT_MIME_TYPE = SpanAttributes.INPUT_MIME_TYPE
INPUT_VALUE = SpanAttributes.INPUT_VALUE
JSON = OpenInferenceMimeTypeValues.JSON.value
LLM = OpenInferenceSpanKindValues.LLM.value
LLM_INPUT_MESSAGES = SpanAttributes.LLM_INPUT_MESSAGES
LLM_INVOCATION_PARAMETERS = SpanAttributes.LLM_INVOCATION_PARAMETERS
LLM_MODEL_NAME = SpanAttributes.LLM_MODEL_NAME
LLM_TOKEN_COUNT_COMPLETION = SpanAttributes.LLM_TOKEN_COUNT_COMPLETION
LLM_TOKEN_COUNT_PROMPT = SpanAttributes.LLM_TOKEN_COUNT_PROMPT
LLM_TOKEN_COUNT_TOTAL = SpanAttributes.LLM_TOKEN_COUNT_TOTAL
MESSAGE_CONTENT = MessageAttributes.MESSAGE_CONTENT
MESSAGE_CONTENT_IMAGE = MessageContentAttributes.MESSAGE_CONTENT_IMAGE
MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON = MessageAttributes.MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON
MESSAGE_FUNCTION_CALL_NAME = MessageAttributes.MESSAGE_FUNCTION_CALL_NAME
MESSAGE_NAME = MessageAttributes.MESSAGE_NAME
OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
OUTPUT_MIME_TYPE = SpanAttributes.OUTPUT_MIME_TYPE
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE

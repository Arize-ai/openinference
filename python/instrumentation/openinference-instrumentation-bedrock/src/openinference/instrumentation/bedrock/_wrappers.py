from __future__ import annotations

import asyncio
import inspect
import json
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Dict, Mapping, Tuple, cast

import wrapt
from opentelemetry import context as context_api
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.trace import Span, Status, StatusCode, Tracer
from typing_extensions import override

from openinference.instrumentation import get_attributes_from_context, safe_json_dumps
from openinference.instrumentation.bedrock._attribute_extractor import AttributeExtractor
from openinference.instrumentation.bedrock._converse_attributes import (
    get_attributes_from_request_data,
)

if TYPE_CHECKING:
    from mypy_boto3_bedrock_agent_runtime.type_defs import RetrieveAndGenerateRequestTypeDef
    from mypy_boto3_bedrock_runtime.type_defs import ConverseStreamRequestTypeDef
from openinference.instrumentation.bedrock._converse_stream_callback import _ConverseStreamCallback
from openinference.instrumentation.bedrock._rag_wrappers import _RagEventStream
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


def _is_async_at_decoration(wrapped: Callable[..., Any]) -> bool:
    """
    Decide sync vs async at decoration time (same heuristic as call-time check).

    Botocore builds client methods as sync def that return self._make_api_call(...).
    In aiobotocore, _make_api_call is async, so the method returns a coroutine but
    is not itself a coroutine function. We detect async by checking whether the
    bound instance's _make_api_call is a coroutine function.
    """
    if inspect.iscoroutinefunction(wrapped):
        return True
    instance = getattr(wrapped, "__self__", None)
    if instance is not None and hasattr(instance, "_make_api_call"):
        if asyncio.iscoroutinefunction(instance._make_api_call):
            return True
    return False


class _WithTracer:
    def __init__(self, tracer: Tracer):
        self._tracer = tracer

    def __call__(self, wrapped: Callable[..., Any]) -> Any:
        """Route to sync or async wrapper at decoration time."""
        if _is_async_at_decoration(wrapped):
            return self._wrap_async(wrapped)
        return self._wrap_sync(wrapped)

    def _wrap_sync(self, wrapped: Callable[..., Any]) -> Any:
        raise NotImplementedError("Subclasses must implement _wrap_sync")

    def _wrap_async(self, wrapped: Callable[..., Any]) -> Any:
        raise NotImplementedError("Subclasses must implement _wrap_async")


class _InvokeModelWithResponseStream(_WithTracer):
    _name = "bedrock.invoke_model_with_response_stream"

    @override
    def _wrap_sync(self, wrapped: Callable[..., Any]) -> Any:
        @wrapt.decorator  # type: ignore[misc,attr-defined,unused-ignore]
        def _impl(
            wrapped_fn: Callable[..., Any],
            instance: Any,
            args: Tuple[Any, ...],
            kwargs: Mapping[str, Any],
        ) -> Any:
            return self._sync_call(wrapped_fn, args, kwargs)

        return _impl(wrapped)

    @override
    def _wrap_async(self, wrapped: Callable[..., Any]) -> Any:
        @wrapt.decorator  # type: ignore[misc,attr-defined,unused-ignore]
        async def _impl(
            wrapped_fn: Callable[..., Any],
            instance: Any,
            args: Tuple[Any, ...],
            kwargs: Mapping[str, Any],
        ) -> Any:
            return await self._async_call(wrapped_fn, args, kwargs)

        return _impl(wrapped)

    @staticmethod
    def handle_response(response: Any, kwargs: Mapping[str, Any], span: Span) -> Any:
        from botocore.eventstream import EventStream

        # Request body is InvokeModel payload (blob: str or bytes); json.loads accepts both.
        # Construct a parsed view of the request rather than mutating kwargs in place.
        body = json.loads(kwargs["body"])
        if isinstance(response["body"], EventStream):
            if "anthropic_version" in body:
                parsed_request: Mapping[str, Any] = {**kwargs, "body": body}
                response["body"] = _EventStream(
                    response["body"],
                    _AnthropicMessagesCallback(span, parsed_request),
                    _use_span(span),
                )
                return response
        span.set_attribute(LLM_INVOCATION_PARAMETERS, body)
        span.set_attribute(INPUT_MIME_TYPE, JSON)
        span.set_attribute(INPUT_VALUE, body)
        span.set_attribute(OPENINFERENCE_SPAN_KIND, LLM)
        span.end()

    def _sync_call(
        self,
        wrapped: Callable[..., Any],
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)
        with self._tracer.start_as_current_span(
            self._name,
            end_on_exit=False,
        ) as span:
            response = wrapped(*args, **kwargs)
            self.handle_response(response, kwargs, span)
            return response

    async def _async_call(
        self,
        wrapped: Callable[..., Any],
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return await wrapped(*args, **kwargs)
        with self._tracer.start_as_current_span(
            self._name,
            end_on_exit=False,
        ) as span:
            try:
                response = await wrapped(*args, **kwargs)
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.end()
                raise
            self.handle_response(response, kwargs, span)
            return response


class _ConverseStream(_WithTracer):
    """
    A decorator class for tracing Bedrock 'converse_stream' API calls with OpenTelemetry.

    This class wraps the Bedrock converse_stream API, starting a tracing span for each invocation.
    It attaches request attributes to the span, and if the response contains a streaming
    EventStream, it wraps the stream to enable span-aware streaming instrumentation.
    The span is ended when the stream is fully consumed or if the response is not a stream.
    """

    _name = "bedrock.converse_stream"

    @override
    def _wrap_sync(self, wrapped: Callable[..., Any]) -> Any:
        @wrapt.decorator  # type: ignore[misc,attr-defined,unused-ignore]
        def _impl(
            wrapped_fn: Callable[..., Any],
            instance: Any,
            args: Tuple[Any, ...],
            kwargs: Mapping[str, Any],
        ) -> Any:
            return self._sync_call(wrapped_fn, args, kwargs)

        return _impl(wrapped)

    @override
    def _wrap_async(self, wrapped: Callable[..., Any]) -> Any:
        @wrapt.decorator  # type: ignore[misc,attr-defined,unused-ignore]
        async def _impl(
            wrapped_fn: Callable[..., Any],
            instance: Any,
            args: Tuple[Any, ...],
            kwargs: Mapping[str, Any],
        ) -> Any:
            return await self._async_call(wrapped_fn, args, kwargs)

        return _impl(wrapped)

    @staticmethod
    def handle_response(response: Any, kwargs: Mapping[str, Any], span: Span) -> Any:
        from botocore.eventstream import EventStream

        if isinstance(response["stream"], EventStream):
            response["stream"] = _EventStream(
                response["stream"],
                _ConverseStreamCallback(span, cast("ConverseStreamRequestTypeDef", kwargs)),
                _use_span(span),
            )
            return response
        span.set_attribute(OPENINFERENCE_SPAN_KIND, LLM)
        span.end()

    def _sync_call(
        self,
        wrapped: Callable[..., Any],
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)
        with self._tracer.start_as_current_span(
            self._name,
            attributes=get_attributes_from_request_data(
                cast("ConverseStreamRequestTypeDef", kwargs)
            ),
            end_on_exit=False,
        ) as span:
            response = wrapped(*args, **kwargs)
            self.handle_response(response, kwargs, span)
            return response

    async def _async_call(
        self,
        wrapped: Callable[..., Any],
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return await wrapped(*args, **kwargs)
        with self._tracer.start_as_current_span(
            self._name,
            attributes=get_attributes_from_request_data(
                cast("ConverseStreamRequestTypeDef", kwargs)
            ),
            end_on_exit=False,
        ) as span:
            try:
                response = await wrapped(*args, **kwargs)
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.end()
                raise
            self.handle_response(response, kwargs, span)
            return response


class _InvokeAgentWithResponseStream(_WithTracer):
    _name = "bedrock_agent.invoke_agent"

    @override
    def _wrap_sync(self, wrapped: Callable[..., Any]) -> Any:
        @wrapt.decorator  # type: ignore[misc,attr-defined,unused-ignore]
        def _impl(
            wrapped_fn: Callable[..., Any],
            instance: Any,
            args: Tuple[Any, ...],
            kwargs: Mapping[str, Any],
        ) -> Any:
            return self._sync_call(wrapped_fn, args, kwargs)

        return _impl(wrapped)

    @override
    def _wrap_async(self, wrapped: Callable[..., Any]) -> Any:
        @wrapt.decorator  # type: ignore[misc,attr-defined,unused-ignore]
        async def _impl(
            wrapped_fn: Callable[..., Any],
            instance: Any,
            args: Tuple[Any, ...],
            kwargs: Mapping[str, Any],
        ) -> Any:
            return await self._async_call(wrapped_fn, args, kwargs)

        return _impl(wrapped)

    @staticmethod
    def handle_request_attributes(span: Span, kwargs: Mapping[str, Any]) -> None:
        attributes = {
            SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.AGENT.value,
            SpanAttributes.LLM_PROVIDER: OpenInferenceLLMProviderValues.AWS.value,
        }
        if input_text := kwargs.get("inputText"):
            attributes[SpanAttributes.INPUT_VALUE] = input_text
        span.set_attributes({k: v for k, v in attributes.items() if v is not None})

    def _sync_call(
        self, wrapped: Callable[..., Any], args: Tuple[Any, ...], kwargs: Mapping[str, Any]
    ) -> Any:
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)
        with self._tracer.start_as_current_span(
            f"bedrock_agent.{wrapped.__name__}",
            end_on_exit=False,
        ) as span:
            self.handle_request_attributes(span, kwargs)
            try:
                response = wrapped(*args, **kwargs)
                response["completion"] = _EventStream(
                    response["completion"],
                    _ResponseAccumulator(span, self._tracer, kwargs),
                    _use_span(span),
                )
                return response
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.end()
                raise

    async def _async_call(
        self, wrapped: Callable[..., Any], args: Tuple[Any, ...], kwargs: Mapping[str, Any]
    ) -> Any:
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return await wrapped(*args, **kwargs)
        with self._tracer.start_as_current_span(
            f"bedrock_agent.{wrapped.__name__}",
            end_on_exit=False,
        ) as span:
            self.handle_request_attributes(span, kwargs)
            try:
                response = await wrapped(*args, **kwargs)
                response["completion"] = _EventStream(
                    response["completion"],
                    _ResponseAccumulator(span, self._tracer, kwargs),
                    _use_span(span),
                )
                return response
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.end()
                raise


class _RetrieveAndGenerateStream(_WithTracer):
    _name = "bedrock_agent.retrieve_and_generate_stream"

    @override
    def _wrap_sync(self, wrapped: Callable[..., Any]) -> Any:
        @wrapt.decorator  # type: ignore[misc,attr-defined,unused-ignore]
        def _impl(
            wrapped_fn: Callable[..., Any],
            instance: Any,
            args: Tuple[Any, ...],
            kwargs: Mapping[str, Any],
        ) -> Any:
            return self._sync_call(wrapped_fn, args, kwargs)

        return _impl(wrapped)

    @override
    def _wrap_async(self, wrapped: Callable[..., Any]) -> Any:
        @wrapt.decorator  # type: ignore[misc,attr-defined,unused-ignore]
        async def _impl(
            wrapped_fn: Callable[..., Any],
            instance: Any,
            args: Tuple[Any, ...],
            kwargs: Mapping[str, Any],
        ) -> Any:
            return await self._async_call(wrapped_fn, args, kwargs)

        return _impl(wrapped)

    def _sync_call(
        self,
        wrapped: Callable[..., Any],
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)
        with self._tracer.start_as_current_span(
            self._name,
            end_on_exit=False,
        ) as span:
            try:
                span.set_attributes(
                    AttributeExtractor.extract_bedrock_rag_input_attributes(
                        cast("RetrieveAndGenerateRequestTypeDef", kwargs)
                    )
                )
                response = wrapped(*args, **kwargs)
                response["stream"] = _EventStream(
                    response["stream"],
                    _RagEventStream(span, self._tracer, kwargs),
                    _use_span(span),
                )
                return response
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.end()
                raise

    async def _async_call(
        self,
        wrapped: Callable[..., Any],
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return await wrapped(*args, **kwargs)
        with self._tracer.start_as_current_span(
            self._name,
            end_on_exit=False,
        ) as span:
            try:
                span.set_attributes(
                    AttributeExtractor.extract_bedrock_rag_input_attributes(
                        cast("RetrieveAndGenerateRequestTypeDef", kwargs)
                    )
                )
                response = await wrapped(*args, **kwargs)
                response["stream"] = _EventStream(
                    response["stream"],
                    _RagEventStream(span, self._tracer, kwargs),
                    _use_span(span),
                )
                return response
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.end()
                raise


def _apply_guardrail_wrapper(tracer: Tracer) -> Callable[[Any], Callable[..., Any]]:
    """
    Wraps bedrock-runtime apply_guardrail: one span per call with GUARDRAIL span kind.

    Captures:
    - input.value: selected requested params in json format
    - output.value: joined text from the outputs response field
    - metadata: guardrailIdentifier, guardrailVersion, source (from request),
                action and actionReason (from response, when present)
    """

    def _guardrail_wrapper(wrapped_client: Any) -> Callable[..., Any]:
        def _serialize_guardrail_content(record: Mapping[str, Any]) -> Dict[str, Any]:
            content_block: Dict[str, Any] = {}
            if "text" in record and record["text"] is not None:
                content_block["text"] = record["text"]
            if isinstance(image := record.get("image"), Mapping):
                image_block: Dict[str, Any] = {}
                if "format" in image and image["format"] is not None:
                    image_block["format"] = image["format"]
                if isinstance(source := image.get("source"), Mapping):
                    source_block: Dict[str, Any] = {}
                    for key, value in source.items():
                        if value is None:
                            continue
                        if key == "bytes":
                            try:
                                source_block["bytes"] = f"<{len(value)} bytes>"
                            except TypeError:
                                source_block["bytes"] = "<bytes>"
                        else:
                            source_block[key] = value
                    if source_block:
                        image_block["source"] = source_block
                if image_block:
                    content_block["image"] = image_block
            return content_block

        def _set_input_attributes(span: Span, kwargs: Mapping[str, Any]) -> None:
            span.set_attribute(
                SpanAttributes.OPENINFERENCE_SPAN_KIND,
                OpenInferenceSpanKindValues.GUARDRAIL.value,
            )
            input_values: Dict[str, Any] = {}
            if source := kwargs.get("source"):
                input_values["source"] = source
            if content := kwargs.get("content"):
                serialized_content = [
                    serialized_block
                    for record in content
                    if isinstance(record, Mapping)
                    if (serialized_block := _serialize_guardrail_content(record))
                ]
                if serialized_content:
                    input_values["content"] = serialized_content
            if output_scope := kwargs.get("outputScope"):
                input_values["outputScope"] = output_scope
            span.set_attribute(SpanAttributes.INPUT_VALUE, safe_json_dumps(input_values))
            span.set_attribute(SpanAttributes.INPUT_MIME_TYPE, JSON)

        def _response_is_blocked(response: Mapping[str, Any]) -> bool:
            assessments = response.get("assessments")
            if not isinstance(assessments, list):
                return False
            return AttributeExtractor.is_blocked_guardrail(
                cast(
                    Any,
                    [
                        {"inputAssessments": [assessment]}
                        for assessment in assessments
                        if isinstance(assessment, Mapping)
                    ],
                )
            )

        def _set_response_attributes(span: Span, kwargs: Mapping[str, Any], response: Any) -> None:
            metadata: Dict[str, Any] = {
                "guardrailIdentifier": kwargs.get("guardrailIdentifier"),
                "guardrailVersion": kwargs.get("guardrailVersion"),
            }
            output: Dict[str, Any] = {}
            if isinstance(response, Mapping):
                if action := response.get("action"):
                    output["action"] = action
                if action_reason := response.get("actionReason"):
                    output["actionReason"] = action_reason
                if outputs := response.get("outputs"):
                    output["outputs"] = outputs
                if guardrail_coverage := response.get("guardrailCoverage"):
                    metadata["guardrailCoverage"] = guardrail_coverage
                if usage := response.get("usage"):
                    metadata["usage"] = usage
            if metadata:
                span.set_attribute(SpanAttributes.METADATA, safe_json_dumps(metadata))
            if response is not None:
                span.set_attribute(SpanAttributes.OUTPUT_VALUE, safe_json_dumps(output))
                span.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE, JSON)
            span.set_attributes(dict(get_attributes_from_context()))
            if isinstance(response, Mapping) and _response_is_blocked(response):
                span.set_status(
                    Status(StatusCode.ERROR, cast(str, response.get("actionReason", "")))
                )
            else:
                span.set_status(Status(StatusCode.OK))

        if _is_async_at_decoration(wrapped_client.apply_guardrail):

            @wraps(wrapped_client.apply_guardrail)
            async def async_instrumented_response(*args: Any, **kwargs: Any) -> Any:
                if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
                    return await wrapped_client._unwrapped_apply_guardrail(*args, **kwargs)

                with tracer.start_as_current_span("bedrock.apply_guardrail") as span:
                    _set_input_attributes(span, kwargs)
                    try:
                        response = await wrapped_client._unwrapped_apply_guardrail(*args, **kwargs)
                        _set_response_attributes(span, kwargs, response)
                        return response
                    except Exception as e:
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        raise

            return async_instrumented_response

        @wraps(wrapped_client.apply_guardrail)
        def sync_instrumented_response(*args: Any, **kwargs: Any) -> Any:
            if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
                return wrapped_client._unwrapped_apply_guardrail(*args, **kwargs)

            with tracer.start_as_current_span("bedrock.apply_guardrail") as span:
                _set_input_attributes(span, kwargs)
                try:
                    response = wrapped_client._unwrapped_apply_guardrail(*args, **kwargs)
                    _set_response_attributes(span, kwargs, response)
                    return response
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

        return sync_instrumented_response

    return _guardrail_wrapper


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

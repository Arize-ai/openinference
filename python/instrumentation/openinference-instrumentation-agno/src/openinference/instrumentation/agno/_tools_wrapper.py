from enum import Enum
from types import AsyncGeneratorType, GeneratorType
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Iterator,
    Mapping,
    Optional,
    Tuple,
    Union,
)

from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.util.types import AttributeValue

from agno.run.agent import RunContentEvent, RunOutputEvent
from agno.run.team import RunContentEvent as TeamRunContentEvent
from agno.run.team import TeamRunOutputEvent
from agno.tools.function import FunctionCall, ToolResult
from openinference.instrumentation import get_attributes_from_context, safe_json_dumps
from openinference.semconv.trace import (
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)


def _flatten(mapping: Optional[Mapping[str, Any]]) -> Iterator[Tuple[str, AttributeValue]]:
    if not mapping:
        return
    for key, value in mapping.items():
        if value is None:
            continue
        if isinstance(value, Mapping):
            for sub_key, sub_value in _flatten(value):
                yield f"{key}.{sub_key}", sub_value
        elif isinstance(value, list) and any(isinstance(item, Mapping) for item in value):
            for index, sub_mapping in enumerate(value):
                for sub_key, sub_value in _flatten(sub_mapping):
                    yield f"{key}.{index}.{sub_key}", sub_value
        else:
            if isinstance(value, Enum):
                value = value.value
            yield key, value


def _function_call_attributes(function_call: FunctionCall) -> Iterator[Tuple[str, Any]]:
    function = function_call.function
    function_name = function.name
    function_arguments = function_call.arguments

    yield TOOL_NAME, function_name

    if function_description := getattr(function, "description", None):
        yield TOOL_DESCRIPTION, function_description
    yield TOOL_PARAMETERS, safe_json_dumps(function_arguments)


def _input_value_and_mime_type_for_tool_span(
    arguments: Mapping[str, Any],
) -> Iterator[Tuple[str, Any]]:
    yield INPUT_MIME_TYPE, JSON
    yield INPUT_VALUE, safe_json_dumps(arguments)


def _output_value_and_mime_type_for_tool_span(result: Any) -> Iterator[Tuple[str, Any]]:
    yield OUTPUT_VALUE, str(result)
    yield OUTPUT_MIME_TYPE, TEXT


class _FunctionCallWrapper:
    def __init__(self, tracer: trace_api.Tracer) -> None:
        self._tracer = tracer

    def run(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        function_call = instance
        function = function_call.function
        function_name = function.name
        function_arguments = function_call.arguments

        span_name = f"{function_name}"

        span = self._tracer.start_span(
            span_name,
            attributes={
                OPENINFERENCE_SPAN_KIND: TOOL,
                **dict(_input_value_and_mime_type_for_tool_span(function_arguments)),
                **dict(_function_call_attributes(function_call)),
                **dict(get_attributes_from_context()),
            },
        )

        try:
            with trace_api.use_span(span, end_on_exit=False):
                response = wrapped(*args, **kwargs)

            if response.status == "success":
                if isinstance(function_call.result, (GeneratorType, Iterator)):
                    function_call.result = self._handle_success_streaming(
                        function_call.result, span
                    )
                    response.result = function_call.result
                    return response
                else:
                    self._handle_success(function_call.result, span)
                    span.end()

            elif response.status == "failure":
                function_error_message = function_call.error
                span.set_status(trace_api.StatusCode.ERROR, function_error_message)
                span.set_attribute(OUTPUT_VALUE, function_error_message)
                span.set_attribute(OUTPUT_MIME_TYPE, TEXT)
            else:
                span.set_status(trace_api.StatusCode.ERROR, "Unknown function call status")

        except Exception as e:
            span.set_status(trace_api.StatusCode.ERROR, str(e))
            span.record_exception(e)
            span.end()
            raise

        return response

    async def arun(
        self,
        wrapped: Callable[..., Awaitable[Any]],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return await wrapped(*args, **kwargs)

        function_call = instance
        function = function_call.function
        function_name = function.name
        function_arguments = function_call.arguments

        span_name = f"{function_name}"

        span = self._tracer.start_span(
            span_name,
            attributes={
                OPENINFERENCE_SPAN_KIND: TOOL,
                **dict(_input_value_and_mime_type_for_tool_span(function_arguments)),
                **dict(_function_call_attributes(function_call)),
                **dict(get_attributes_from_context()),
            },
        )

        try:
            with trace_api.use_span(span, end_on_exit=False):
                response = await wrapped(*args, **kwargs)

            if response.status == "success":
                if isinstance(function_call.result, (AsyncGeneratorType, AsyncIterator)):
                    function_call.result = self._handle_async_success_streaming(
                        function_call.result, span
                    )
                    response.result = function_call.result
                    return response
                elif isinstance(function_call.result, (GeneratorType, Iterator)):
                    function_call.result = self._handle_success_streaming(
                        function_call.result, span
                    )
                    response.result = function_call.result
                    return response
                else:
                    self._handle_success(function_call.result, span)
                    span.end()
            elif response.status == "failure":
                function_error_message = function_call.error
                span.set_status(trace_api.StatusCode.ERROR, function_error_message)
                span.set_attribute(OUTPUT_VALUE, function_error_message)
                span.set_attribute(OUTPUT_MIME_TYPE, TEXT)
            else:
                span.set_status(trace_api.StatusCode.ERROR, "Unknown function call status")

        except Exception as e:
            span.set_status(trace_api.StatusCode.ERROR, str(e))
            span.record_exception(e)
            span.end()
            raise

        return response

    def _handle_success(self, result: Any, span: trace_api.Span) -> None:
        """Handle non-streaming success case: set span attributes and close span."""
        function_result = ""
        try:
            if isinstance(result, ToolResult):
                function_result = result.content
            else:
                function_result = result
        except Exception:
            # Don't break if observability fails
            pass

        span.set_status(trace_api.StatusCode.OK)
        span.set_attributes(
            dict(
                _output_value_and_mime_type_for_tool_span(
                    result=function_result,
                )
            )
        )

    def _handle_success_streaming(
        self, result: Iterator[Union[RunOutputEvent, TeamRunOutputEvent]], span: trace_api.Span
    ) -> Iterator[Union[RunOutputEvent, TeamRunOutputEvent]]:
        """Handle streaming success case: set span attributes and close span."""
        function_result = ""
        try:
            for item in result:
                yield item
                if isinstance(item, (RunContentEvent, TeamRunContentEvent)):
                    function_result += self._parse_content(item.content)
                elif isinstance(item, str):
                    function_result += str(item)
        except Exception:
            # Don't break if observability fails
            pass
        span.set_status(trace_api.StatusCode.OK)
        span.set_attributes(
            dict(
                _output_value_and_mime_type_for_tool_span(
                    result=function_result,
                )
            )
        )
        span.end()

    async def _handle_async_success_streaming(
        self, result: AsyncIterator[Union[RunOutputEvent, TeamRunOutputEvent]], span: trace_api.Span
    ) -> AsyncIterator[Union[RunOutputEvent, TeamRunOutputEvent]]:
        """Handle streaming success case: set span attributes and close span."""
        function_result = ""
        async for item in result:
            yield item
            try:
                if isinstance(item, (RunContentEvent, TeamRunContentEvent)):
                    function_result += self._parse_content(item.content)
                elif isinstance(item, str):
                    function_result += str(item)
            except Exception:
                # Don't break if observability fails
                pass

        span.set_status(trace_api.StatusCode.OK)
        span.set_attributes(
            dict(
                _output_value_and_mime_type_for_tool_span(
                    result=function_result,
                )
            )
        )
        span.end()

    def _parse_content(self, content: Any) -> str:
        from pydantic import BaseModel

        if content is not None and isinstance(content, BaseModel):
            return str(content.model_dump_json())
        else:
            # Capture output
            return str(content) if content else ""


# span attributes
INPUT_MIME_TYPE = SpanAttributes.INPUT_MIME_TYPE
INPUT_VALUE = SpanAttributes.INPUT_VALUE
OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
OUTPUT_MIME_TYPE = SpanAttributes.OUTPUT_MIME_TYPE
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
TOOL_DESCRIPTION = SpanAttributes.TOOL_DESCRIPTION
TOOL_NAME = SpanAttributes.TOOL_NAME
TOOL_PARAMETERS = SpanAttributes.TOOL_PARAMETERS
USER_ID = SpanAttributes.USER_ID
GRAPH_NODE_ID = SpanAttributes.GRAPH_NODE_ID
GRAPH_NODE_NAME = SpanAttributes.GRAPH_NODE_NAME
GRAPH_NODE_PARENT_ID = SpanAttributes.GRAPH_NODE_PARENT_ID

# mime types
TEXT = OpenInferenceMimeTypeValues.TEXT.value
JSON = OpenInferenceMimeTypeValues.JSON.value

# span kinds
TOOL = OpenInferenceSpanKindValues.TOOL.value

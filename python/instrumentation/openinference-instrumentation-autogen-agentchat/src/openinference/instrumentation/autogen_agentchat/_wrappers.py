import json
import logging
from abc import ABC
from enum import Enum
from inspect import signature
from typing import Any, AsyncGenerator, Callable, Iterator, List, Mapping, Optional, Tuple, Union

from autogen_core.models import CreateResult
from autogen_ext.models.openai import BaseOpenAIChatCompletionClient
from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.context import _RUNTIME_CONTEXT
from opentelemetry.trace.propagation import _SPAN_KEY
from opentelemetry.util.types import AttributeValue

from autogen_agentchat.agents import AssistantAgent, BaseChatAgent
from autogen_agentchat.base import Response, TaskResult
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage
from autogen_agentchat.teams import BaseGroupChat
from openinference.instrumentation import (
    get_attributes_from_context,
    get_output_attributes,
    safe_json_dumps,
)
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class _WithTracer(ABC):
    def __init__(
        self,
        tracer: trace_api.Tracer,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._tracer = tracer


class SafeJSONEncoder(json.JSONEncoder):
    """
    Safely encodes non-JSON-serializable objects.
    """

    def default(self, o: Any) -> Any:
        try:
            return super().default(o)
        except TypeError:
            if hasattr(o, "dict") and callable(o.dict):  # pydantic v1 models, e.g., from Cohere
                return o.dict()
            return repr(o)


def _flatten(mapping: Optional[Mapping[str, Any]]) -> Iterator[Tuple[str, AttributeValue]]:
    if not mapping:
        return
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


def _get_input_value(method: Callable[..., Any], *args: Any, **kwargs: Any) -> str:
    """
    Parses a method call's inputs into a JSON string. Ensures a consistent
    output regardless of whether those inputs are passed as positional or
    keyword arguments.
    """

    # For typical class methods, the corresponding instance of inspect.Signature
    # does not include the self parameter. However, the inspect.Signature
    # instance for __call__ does include the self parameter.
    method_signature = signature(method)
    first_parameter_name = next(iter(method_signature.parameters), None)
    signature_contains_self_parameter = first_parameter_name in ["self"]
    bound_arguments = method_signature.bind(
        *(
            [None]  # the value bound to the method's self argument is discarded below, so pass None
            if signature_contains_self_parameter
            else []  # no self parameter, so `no need to pass a value
        ),
        *args,
        **kwargs,
    )
    return safe_json_dumps(
        {
            **{
                argument_name: argument_value
                for argument_name, argument_value in bound_arguments.arguments.items()
                if argument_name not in ["self", "kwargs"]
            },
            **bound_arguments.arguments.get("kwargs", {}),
        },
        cls=SafeJSONEncoder,
    )


class _AssistantAgentOnMessagesStreamWrapper(_WithTracer):
    def __call__(
        self,
        wrapped: Callable[..., AsyncGenerator[BaseAgentEvent | BaseChatMessage | Response, None]],
        instance: AssistantAgent,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        generator = wrapped(*args, **kwargs)
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return generator

        tracer = self._tracer
        agent_name = instance.name if instance else "AssistantAgent"

        span_name = f"{agent_name}.on_messages_stream"
        attributes = dict(get_attributes_from_context())
        attributes[SpanAttributes.OPENINFERENCE_SPAN_KIND] = OpenInferenceSpanKindValues.AGENT.value
        attributes[SpanAttributes.INPUT_VALUE] = _get_input_value(
            wrapped,
            *args,
            **kwargs,
        )

        async def wrapped_generator() -> AsyncGenerator[
            BaseAgentEvent | BaseChatMessage | Response, None
        ]:
            span = tracer.start_span(
                name=span_name,
                attributes=attributes,
            )
            span.set_status(trace_api.StatusCode.OK)
            token = context_api.attach(context_api.set_value(_SPAN_KEY, span))
            try:
                async for event in generator:
                    yield event
            except GeneratorExit:
                raise
            except BaseException as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                raise
            finally:
                span.end()
                try:
                    _RUNTIME_CONTEXT.detach(token)
                except Exception:
                    # If the context is already detached, we can ignore the error.
                    pass

        return wrapped_generator()


class _BaseChatAgentOnMessagesStreamWrapper(_WithTracer):
    def __call__(
        self,
        wrapped: Callable[..., AsyncGenerator[BaseAgentEvent | BaseChatMessage | Response, None]],
        instance: BaseChatAgent,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        generator = wrapped(*args, **kwargs)
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return generator

        tracer = self._tracer
        agent_name = instance.name if instance else "BaseChatAgent"

        span_name = f"{agent_name}.on_messages_stream"
        attributes = dict(get_attributes_from_context())
        attributes[SpanAttributes.OPENINFERENCE_SPAN_KIND] = OpenInferenceSpanKindValues.AGENT.value
        attributes[SpanAttributes.INPUT_VALUE] = _get_input_value(
            wrapped,
            *args,
            **kwargs,
        )

        async def wrapped_generator() -> AsyncGenerator[
            BaseAgentEvent | BaseChatMessage | Response, None
        ]:
            span = tracer.start_span(
                name=span_name,
                attributes=attributes,
            )
            span.set_status(trace_api.StatusCode.OK)
            token = context_api.attach(context_api.set_value(_SPAN_KEY, span))
            try:
                async for event in generator:
                    yield event
            except GeneratorExit:
                raise
            except BaseException as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                raise
            finally:
                span.end()
                try:
                    _RUNTIME_CONTEXT.detach(token)
                except Exception:
                    # If the context is already detached, we can ignore the error.
                    pass

        return wrapped_generator()


class _BaseAgentRunWrapper:
    def __init__(self, tracer: trace_api.Tracer) -> None:
        self._tracer = tracer

    async def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)
        agent = instance
        if agent:
            span_name = f"{agent.__class__.__name__}.{wrapped.__name__}"
        else:
            span_name = wrapped.__name__
        with self._tracer.start_as_current_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.AGENT,
                        SpanAttributes.INPUT_VALUE: _get_input_value(
                            wrapped,
                            *args,
                            **kwargs,
                        ),
                    }
                )
            ),
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            if agent:
                span.set_attribute("agent_name", agent.name)
                span.set_attribute("agent_description", agent.description)
            try:
                response = await wrapped(*args, **kwargs)
            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                raise
            span.set_status(trace_api.StatusCode.OK)
            span.set_attributes(dict(get_output_attributes(response)))
            span.set_attributes(dict(get_attributes_from_context()))
        return response


class _ToolsRunJSONWrapper:
    def __init__(self, tracer: trace_api.Tracer) -> None:
        self._tracer = tracer

    async def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return await wrapped(*args, **kwargs)
        if instance:
            span_name = f"{instance.__class__.__name__}.{wrapped.__name__}"
        else:
            span_name = wrapped.__name__

        with self._tracer.start_as_current_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.TOOL,
                        SpanAttributes.INPUT_VALUE: _get_input_value(
                            wrapped,
                            *args,
                            **kwargs,
                        ),
                    }
                )
            ),
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            if instance:
                span.set_attribute("tool_name", instance.name)
                span.set_attribute("tool_description", instance.description)
            try:
                response = await wrapped(*args, **kwargs)
            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                raise
            span.set_status(trace_api.StatusCode.OK)
            span.set_attributes(dict(get_output_attributes(response)))
            span.set_attributes(dict(get_attributes_from_context()))
        return response


class _BaseGroupChatRunStreamWrapper:
    def __init__(self, tracer: trace_api.Tracer) -> None:
        self._tracer = tracer

    async def __call__(
        self,
        wrapped: Callable[..., AsyncGenerator[BaseAgentEvent | BaseChatMessage | TaskResult, None]],
        instance: BaseGroupChat,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            async for res in wrapped(*args, **kwargs):
                yield res
            return

        span_name = f"{instance.__class__.__name__}.run_stream"
        with self._tracer.start_as_current_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.CHAIN,
                        SpanAttributes.INPUT_VALUE: _get_input_value(
                            wrapped,
                            *args,
                            **kwargs,
                        ),
                    }
                )
            ),
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            group_chat = instance
            team_id = getattr(group_chat, "_team_id", None)
            participant_names = getattr(group_chat, "_participant_names", None)
            participant_descriptions = getattr(group_chat, "_participant_descriptions", None)

            if team_id:
                span.set_attribute("team_id", team_id)
            if participant_names:
                span.set_attribute("participant_names", participant_names)
            if participant_descriptions:
                span.set_attribute("participant_descriptions", participant_descriptions)

            try:
                async for res in wrapped(*args, **kwargs):
                    if isinstance(res, TaskResult):
                        span.set_attributes(dict(get_output_attributes(res)))
                    yield res
            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                raise
            span.set_status(trace_api.StatusCode.OK)
            span.set_attributes(dict(get_attributes_from_context()))


class _BaseOpenAIChatCompletionClientCreateWrapper(_WithTracer):
    async def __call__(
        self,
        wrapped: Callable[..., CreateResult],
        instance: BaseOpenAIChatCompletionClient,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return await wrapped(*args, **kwargs)

        span_name = f"{instance.__class__.__name__}.create"
        with self._tracer.start_as_current_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.LLM,
                        SpanAttributes.INPUT_VALUE: _get_input_value(
                            wrapped,
                            *args,
                            **kwargs,
                        ),
                    }
                )
            ),
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            try:
                res = await wrapped(*args, **kwargs)
                span.set_attributes(dict(get_output_attributes(res)))
                return res
            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                raise
            finally:
                span.set_attributes(dict(get_attributes_from_context()))


class _BaseOpenAIChatCompletionClientCreateStreamWrapper(_WithTracer):
    async def __call__(
        self,
        wrapped: Callable[..., AsyncGenerator[Union[str, CreateResult], None]],
        instance: BaseOpenAIChatCompletionClient,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            async for res in wrapped(*args, **kwargs):
                yield res
            return
        span_name = f"{instance.__class__.__name__}.create_stream"
        with self._tracer.start_as_current_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.LLM,
                        SpanAttributes.INPUT_VALUE: _get_input_value(
                            wrapped,
                            *args,
                            **kwargs,
                        ),
                    }
                )
            ),
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            try:
                async for res in wrapped(*args, **kwargs):
                    if isinstance(res, CreateResult):
                        span.set_attributes(dict(get_output_attributes(res)))
                    yield res
            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                raise
            span.set_status(trace_api.StatusCode.OK)
            span.set_attributes(dict(get_attributes_from_context()))


class _BaseAgentOnMessageWrapper:
    def __init__(self, tracer: trace_api.Tracer) -> None:
        self._tracer = tracer

    async def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return await wrapped(*args, **kwargs)

        span_name = f"{instance.__class__.__name__}.on_message"
        with self._tracer.start_as_current_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.CHAIN,
                        SpanAttributes.INPUT_VALUE: _get_input_value(
                            wrapped,
                            *args,
                            **kwargs,
                        ),
                    }
                )
            ),
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            try:
                response = await wrapped(*args, **kwargs)
            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                raise
            span.set_status(trace_api.StatusCode.OK)
            span.set_attributes(dict(get_output_attributes(response)))
            span.set_attributes(dict(get_attributes_from_context()))
        return response


INPUT_VALUE = SpanAttributes.INPUT_VALUE
OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
OUTPUT_MIME_TYPE = SpanAttributes.OUTPUT_MIME_TYPE

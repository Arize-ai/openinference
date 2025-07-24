import json
import logging
from abc import ABC
from enum import Enum
from inspect import signature
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    Iterator,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from autogen_core.models import (
    CreateResult,
    LLMMessage,
)
from autogen_core.tools import Tool
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
from openinference.semconv.trace import (
    ImageAttributes,
    MessageAttributes,
    MessageContentAttributes,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolAttributes,
    ToolCallAttributes,
)

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
        elif isinstance(value, list) and any(isinstance(item, Mapping) for item in value):
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

    # Filter out kwargs that aren't in the method signature to prevent binding errors
    valid_kwargs = {}
    for param_name, param_value in kwargs.items():
        if param_name in method_signature.parameters:
            valid_kwargs[param_name] = param_value

    bound_arguments = method_signature.bind(
        *(
            [None]  # the value bound to the method's self argument is discarded below, so pass None
            if signature_contains_self_parameter
            else []  # no self parameter, so no need to pass a value
        ),
        *args,
        **valid_kwargs,
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


def _get_input_source(args: Tuple[Any, ...], kwargs: Mapping[str, Any]) -> Optional[str]:
    last_message = None
    if args:
        if (
            len(args) > 0
            and isinstance(args[0], list)
            and len(args[0]) > 0
            and isinstance(args[0][-1], BaseChatMessage)
        ):
            last_message = args[0][-1]

    if kwargs:
        if (
            "messages" in kwargs
            and isinstance(kwargs["messages"], list)
            and len(kwargs["messages"]) > 0
            and isinstance(kwargs["messages"][-1], BaseChatMessage)
        ):
            last_message = kwargs["messages"][-1]

    if last_message:
        if last_message.source == "user":
            return "start"
        else:
            return last_message.source

    return None


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
        parent_agent_name = _get_input_source(args, kwargs)

        span_name = f"{agent_name}.on_messages_stream"
        attributes = dict(get_attributes_from_context())
        attributes[SpanAttributes.OPENINFERENCE_SPAN_KIND] = OpenInferenceSpanKindValues.AGENT.value
        attributes[SpanAttributes.INPUT_VALUE] = _get_input_value(
            wrapped,
            *args,
            **kwargs,
        )
        attributes[SpanAttributes.GRAPH_NODE_ID] = agent_name
        if parent_agent_name:
            attributes[SpanAttributes.GRAPH_NODE_PARENT_ID] = parent_agent_name

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


class _BaseGroupChatRunStreamWrapper(_WithTracer):
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
        wrapped: Callable[..., Any],
        instance: BaseOpenAIChatCompletionClient,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            # Filter kwargs even when instrumentation is suppressed to prevent API errors
            method_signature = signature(wrapped)
            valid_kwargs = {}
            for param_name, param_value in kwargs.items():
                if param_name in method_signature.parameters:
                    valid_kwargs[param_name] = param_value
            return await wrapped(*args, **valid_kwargs)

        arguments = _bind_arguments(wrapped, *args, **kwargs)
        span_name = f"{instance.__class__.__name__}.create"

        messages = arguments.get("messages")
        tools = arguments.get("tools")

        # Filter kwargs to only include valid parameters for the wrapped method
        method_signature = signature(wrapped)
        valid_kwargs = {}
        for param_name, param_value in kwargs.items():
            if param_name in method_signature.parameters:
                valid_kwargs[param_name] = param_value

        with self._tracer.start_as_current_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        OPENINFERENCE_SPAN_KIND: LLM,
                        **dict(_llm_messages_attributes(messages, "input")),
                        **dict(_get_llm_tool_attributes(tools)),
                        **dict(get_attributes_from_context()),
                    }
                )
            ),
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            try:
                result = await wrapped(*args, **valid_kwargs)
                span.set_status(trace_api.StatusCode.OK)
                span.set_attributes(
                    dict(
                        _flatten(
                            {
                                **dict(get_output_attributes(result)),
                            }
                        )
                    )
                )

            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                raise
        return result


class _BaseOpenAIChatCompletionClientCreateStreamWrapper(_WithTracer):
    async def __call__(
        self,
        wrapped: Callable[..., AsyncGenerator[Union[str, CreateResult], None]],
        instance: BaseOpenAIChatCompletionClient,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            # Filter kwargs even when instrumentation is suppressed to prevent API errors
            method_signature = signature(wrapped)
            valid_kwargs = {}
            for param_name, param_value in kwargs.items():
                if param_name in method_signature.parameters:
                    valid_kwargs[param_name] = param_value
            async for res in wrapped(*args, **valid_kwargs):
                yield res
            return
        arguments = _bind_arguments(wrapped, *args, **kwargs)
        messages = arguments.get("messages", None)
        tools = arguments.get("tools", None)

        # Filter kwargs to only include valid parameters for the wrapped method
        method_signature = signature(wrapped)
        valid_kwargs = {}
        for param_name, param_value in kwargs.items():
            if param_name in method_signature.parameters:
                valid_kwargs[param_name] = param_value

        span_name = f"{instance.__class__.__name__}.create_stream"
        with self._tracer.start_as_current_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        OPENINFERENCE_SPAN_KIND: LLM,
                        **dict(_llm_messages_attributes(messages, "input")),
                        **dict(_get_llm_tool_attributes(tools)),
                        **dict(get_attributes_from_context()),
                    }
                )
            ),
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            try:
                async for res in wrapped(*args, **valid_kwargs):
                    if isinstance(res, CreateResult):
                        span.set_attributes(
                            dict(
                                _flatten(
                                    {
                                        **dict(get_output_attributes(res)),
                                    }
                                )
                            )
                        )
                    yield res
            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                raise
            span.set_status(trace_api.StatusCode.OK)


def _bind_arguments(method: Callable[..., Any], *args: Any, **kwargs: Any) -> dict[str, Any]:
    """
    Bind arguments to a method signature, filtering out invalid kwargs.

    This function is designed to be resilient against upstream API changes
    where parameters are added or removed. It filters out any keyword arguments
    that are not accepted by the method signature before attempting to bind.

    Args:
        method: The method whose signature to bind against
        *args: Positional arguments to bind
        **kwargs: Keyword arguments to bind (will be filtered)

    Returns:
        dict: Bound arguments with defaults applied
    """
    method_signature = signature(method)

    # Filter out kwargs that aren't in the method signature to prevent
    # "unexpected keyword argument" errors when upstream APIs change
    valid_kwargs = {}
    for param_name, param_value in kwargs.items():
        if param_name in method_signature.parameters:
            valid_kwargs[param_name] = param_value

    bound_args = method_signature.bind(*args, **valid_kwargs)
    bound_args.apply_defaults()
    return bound_args.arguments


def _get_llm_tool_attributes(
    tools: Optional["Sequence[Tool]"],
) -> "Mapping[str, AttributeValue]":
    attributes: Dict[str, AttributeValue] = {}
    if not isinstance(tools, Sequence):
        return {}
    for tool_index, tool in enumerate(tools):
        if not isinstance(tool, Tool):
            continue
        if isinstance(tool_json_schema := getattr(tool, "schema"), str):
            attributes[f"{LLM_TOOLS}.{tool_index}.{TOOL_JSON_SCHEMA}"] = tool_json_schema
        elif isinstance(tool_json_schema, dict):
            attributes[f"{LLM_TOOLS}.{tool_index}.{TOOL_JSON_SCHEMA}"] = safe_json_dumps(
                tool_json_schema
            )
    return attributes


def _llm_messages_attributes(
    messages: Optional["Sequence[LLMMessage]"],
    message_type: Literal["input", "output"],
) -> Iterator[Tuple[str, AttributeValue]]:
    base_key = LLM_INPUT_MESSAGES if message_type == "input" else LLM_OUTPUT_MESSAGES
    if not isinstance(messages, Sequence):
        return
    for message_index, message in enumerate(messages):
        # Determine role based on message type
        role = None
        if hasattr(message, "type"):
            if message.type == "SystemMessage":
                role = "developer"
            elif message.type == "UserMessage":
                role = "user"
            elif message.type == "AssistantMessage":
                role = "assistant"
            elif message.type == "FunctionExecutionResultMessage":
                role = "function"
        if role is None:
            continue
        yield f"{base_key}.{message_index}.{MESSAGE_ROLE}", role

        # SystemMessage and UserMessage: content is str or list
        if message.type in ("SystemMessage", "UserMessage"):
            content = getattr(message, "content", None)
            if content is not None:
                # If content is not a primitive, serialize it
                if not isinstance(content, (str, bool, int, float, bytes, type(None))):
                    content = safe_json_dumps(content)
                yield f"{base_key}.{message_index}.{MESSAGE_CONTENT}", content

        # AssistantMessage: content is str or list[FunctionCall], may have thought
        elif message.type == "AssistantMessage":
            content = getattr(message, "content", None)
            if content is not None:
                # If content is a list, serialize each item if not primitive
                if isinstance(content, list):
                    serialized_content = [
                        item
                        if isinstance(item, (str, bool, int, float, bytes, type(None)))
                        else safe_json_dumps(item)
                        for item in content
                    ]
                    yield (
                        f"{base_key}.{message_index}.{MESSAGE_CONTENT}",
                        safe_json_dumps(serialized_content),
                    )
                else:
                    if not isinstance(content, (str, bool, int, float, bytes, type(None))):
                        content = safe_json_dumps(content)
                    yield f"{base_key}.{message_index}.{MESSAGE_CONTENT}", content
            # thought
            thought = getattr(message, "thought", None)
            if thought is not None:
                yield f"{base_key}.{message_index}.thought", thought

        # FunctionExecutionResultMessage: content is list[FunctionExecutionResult]
        elif message.type == "FunctionExecutionResultMessage":
            content = getattr(message, "content", None)
            if isinstance(content, list):
                # First yield the message role
                yield f"{base_key}.{message_index}.message.role", "function"

                # Then yield the function results
                for func_index, func_result in enumerate(content):
                    # Create a function object with all attributes
                    function_obj = {}

                    name = getattr(func_result, "name", None)
                    if name is not None:
                        function_obj["name"] = name

                    func_content = getattr(func_result, "content", None)
                    if func_content is not None:
                        if not isinstance(func_content, (str, bool, int, float, bytes, type(None))):
                            func_content = safe_json_dumps(func_content)
                        function_obj["content"] = func_content

                    call_id = getattr(func_result, "call_id", None)
                    if call_id is not None:
                        function_obj["call_id"] = call_id

                    is_error = getattr(func_result, "is_error", None)
                    if is_error is not None:
                        function_obj["is_error"] = is_error

                    # Yield the complete function object
                    yield (
                        f"{base_key}.{message_index}.function.{func_index}",
                        safe_json_dumps(function_obj),
                    )


def _get_attribute(obj: Any, attr_name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(attr_name, default)
    return getattr(obj, attr_name, default)


INPUT_VALUE = SpanAttributes.INPUT_VALUE
OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
OUTPUT_MIME_TYPE = SpanAttributes.OUTPUT_MIME_TYPE
INPUT_MIME_TYPE = SpanAttributes.INPUT_MIME_TYPE
JSON = OpenInferenceMimeTypeValues.JSON.value
LLM_INPUT_MESSAGES = SpanAttributes.LLM_INPUT_MESSAGES
LLM_OUTPUT_MESSAGES = SpanAttributes.LLM_OUTPUT_MESSAGES
MESSAGE_ROLE = MessageAttributes.MESSAGE_ROLE
MESSAGE_CONTENT = MessageAttributes.MESSAGE_CONTENT
LLM = OpenInferenceSpanKindValues.LLM.value
LLM_TOOLS = SpanAttributes.LLM_TOOLS
TOOL_JSON_SCHEMA = ToolAttributes.TOOL_JSON_SCHEMA
MESSAGE_CONTENTS = MessageAttributes.MESSAGE_CONTENTS
MESSAGE_TOOL_CALL_ID = MessageAttributes.MESSAGE_TOOL_CALL_ID
MESSAGE_TOOL_CALLS = MessageAttributes.MESSAGE_TOOL_CALLS
MESSAGE_CONTENT_TYPE = MessageContentAttributes.MESSAGE_CONTENT_TYPE
MESSAGE_CONTENT_TEXT = MessageContentAttributes.MESSAGE_CONTENT_TEXT
TOOL_CALL_ID = ToolCallAttributes.TOOL_CALL_ID
TOOL_CALL_FUNCTION_NAME = ToolCallAttributes.TOOL_CALL_FUNCTION_NAME
TOOL_CALL_FUNCTION_ARGUMENTS_JSON = ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON
MESSAGE_CONTENT_IMAGE = MessageContentAttributes.MESSAGE_CONTENT_IMAGE
IMAGE_URL = ImageAttributes.IMAGE_URL

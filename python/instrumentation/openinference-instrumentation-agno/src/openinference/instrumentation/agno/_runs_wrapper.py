from enum import Enum
from inspect import signature
from secrets import token_hex
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Iterator,
    Mapping,
    Optional,
    OrderedDict,
    Tuple,
    Union,
    cast,
)

from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.context.context import Context
from opentelemetry.util.types import AttributeValue

from agno.agent import Agent
from agno.models.message import Message
from agno.run.agent import RunOutput
from agno.run.messages import RunMessages
from agno.run.team import TeamRunOutput
from agno.team import Team
from agno.tools.function import Function
from agno.tools.toolkit import Toolkit
from openinference.instrumentation import get_attributes_from_context
from openinference.semconv.trace import (
    MessageAttributes,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)

_AGNO_PARENT_NODE_CONTEXT_KEY = context_api.create_key("agno_parent_node_id")


def _get_attr(obj: Any, key: str, default: Any = None) -> Any:
    """Helper function to get attribute from either dict or object."""
    if obj is None:
        return default
    if isinstance(obj, dict):  # It's a dict
        return obj.get(key, default)
    else:  # It's an object with attributes
        return getattr(obj, key, default)


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


def _get_user_message_content(method: Callable[..., Any], *args: Any, **kwargs: Any) -> str:
    arguments = _bind_arguments(method, *args, **kwargs)
    arguments = _strip_method_args(arguments)

    # Try to get input from run_response.input.input_content
    run_response: Optional[RunOutput] = arguments.get("run_response")
    if run_response and hasattr(run_response, "input") and run_response.input:
        if hasattr(run_response.input, "input_content") and run_response.input.input_content:
            if isinstance(run_response.input.input_content, str):
                return run_response.input.input_content
            elif isinstance(run_response.input.input_content, list):
                list_content = ""
                for item in run_response.input.input_content:
                    if isinstance(item, Message):
                        list_content += str(item.content) + "\n"
                    else:
                        list_content += str(item) + "\n"
                return list_content
            elif isinstance(run_response.input.input_content, dict):
                import json

                return json.dumps(run_response.input.input_content, indent=2, ensure_ascii=False)
            elif isinstance(run_response.input.input_content, Message):
                return str(run_response.input.input_content.content)

            return str(run_response.input.input_content)

    # Fallback: try run_messages approach
    run_messages: Optional[RunMessages] = arguments.get("run_messages")
    if run_messages and run_messages.user_message:
        return str(run_messages.user_message.content)

    return ""


def _extract_run_response_output(run_response: Union[RunOutput, TeamRunOutput]) -> str:
    if run_response and run_response.content:
        if isinstance(run_response.content, str):
            return run_response.content
        else:
            return str(run_response.content.model_dump_json())
    return ""


def _bind_arguments(method: Callable[..., Any], *args: Any, **kwargs: Any) -> Dict[str, Any]:
    method_signature = signature(method)
    bound_args = method_signature.bind(*args, **kwargs)
    bound_args.apply_defaults()
    arguments = bound_args.arguments
    arguments = OrderedDict(
        {key: value for key, value in arguments.items() if value is not None and value != {}}
    )
    return arguments


def _strip_method_args(arguments: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in arguments.items() if key not in ("self", "cls")}


def _generate_node_id() -> str:
    return token_hex(8)  # Generates 16 hex characters (8 bytes)


def _run_arguments(arguments: Mapping[str, Any]) -> Iterator[Tuple[str, AttributeValue]]:
    user_id = arguments.get("user_id")
    session_id = arguments.get("session_id")

    # For agno v2: session_id might be in the session object for internal _run method
    session = arguments.get("session")
    if session and hasattr(session, "session_id"):
        session_id = session.session_id

    if session_id:
        yield SESSION_ID, session_id
    if user_id:
        yield USER_ID, user_id


def _agent_run_attributes(
    agent: Union[Agent, Team], key_suffix: str = ""
) -> Iterator[Tuple[str, AttributeValue]]:
    # Get parent from execution context instead of structural parent
    context_parent_id = context_api.get_value(_AGNO_PARENT_NODE_CONTEXT_KEY)

    if isinstance(agent, Team):
        # Set graph attributes for team
        if agent.name:
            yield GRAPH_NODE_NAME, agent.name

        if hasattr(agent, "id") and agent.id:
            yield "agno.team.id", agent.id

        if hasattr(agent, "user_id") and agent.user_id:
            yield USER_ID, agent.user_id

        # Use context parent instead of structural parent
        if context_parent_id:
            yield GRAPH_NODE_PARENT_ID, cast(str, context_parent_id)

        # Set legacy team attributes
        yield f"agno{key_suffix}.team", agent.name or ""
        for member in agent.members:
            yield from _agent_run_attributes(member, f".{member.name}")

    elif isinstance(agent, Agent):
        # Set graph attributes for agent
        if agent.name:
            yield GRAPH_NODE_NAME, agent.name

        if hasattr(agent, "id") and agent.id:
            yield "agno.agent.id", agent.id

        if hasattr(agent, "user_id") and agent.user_id:
            yield USER_ID, agent.user_id

        # Use context parent instead of structural parent
        if context_parent_id:
            yield GRAPH_NODE_PARENT_ID, cast(str, context_parent_id)

        # Set legacy agent attributes
        if agent.name:
            yield f"agno{key_suffix}.agent", agent.name or ""

        if agent.knowledge:
            yield f"agno{key_suffix}.knowledge", agent.knowledge.__class__.__name__

        if agent.tools:
            tool_names = []
            for tool in agent.tools:
                if isinstance(tool, Function):
                    tool_names.append(tool.name)
                elif isinstance(tool, Toolkit):
                    tool_names.extend([f for f in tool.functions.keys()])
                elif callable(tool):
                    tool_names.append(tool.__name__)
                else:
                    tool_names.append(str(tool))
            yield f"agno{key_suffix}.tools", tool_names


def _setup_team_context(
    agent: Union[Agent, Team], node_id: str
) -> Tuple[Optional[Any], Optional[Context]]:
    if isinstance(agent, Team):
        team_ctx = context_api.set_value(_AGNO_PARENT_NODE_CONTEXT_KEY, node_id)
        return context_api.attach(team_ctx), team_ctx
    return None, None


class _RunWrapper:
    def __init__(self, tracer: trace_api.Tracer) -> None:
        self._tracer = tracer

    """
    We need to keep track of parent/child relationships for agent logging. We do this by:
    1. Each run() method generates a unique node_id and sets it directly as GRAPH_NODE_ID in span
    attributes
    2. Team.run() sets _AGNO_PARENT_NODE_CONTEXT_KEY for child agents
    3. Agent.run() inherits _AGNO_PARENT_NODE_CONTEXT_KEY from team context for parent relationships
    4. _agent_run_attributes() uses _AGNO_PARENT_NODE_CONTEXT_KEY to set GRAPH_NODE_PARENT_ID
    5. This ensures correct parent-child relationships with unique node IDs for each execution
    """

    def run(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)
        if hasattr(instance, "name") and instance.name:
            agent_name = instance.name.replace(" ", "_").replace("-", "_")
        else:
            if isinstance(instance, Team):
                agent_name = "Team"
            else:
                agent_name = "Agent"
        span_name = f"{agent_name}.run"

        # Generate unique node ID for this execution
        node_id = _generate_node_id()

        arguments = _bind_arguments(wrapped, *args, **kwargs)

        span = self._tracer.start_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        OPENINFERENCE_SPAN_KIND: AGENT,
                        GRAPH_NODE_ID: node_id,
                        INPUT_VALUE: _get_user_message_content(
                            wrapped,
                            *args,
                            **kwargs,
                        ),
                        **dict(_agent_run_attributes(instance)),
                        **dict(_run_arguments(arguments)),
                        **dict(get_attributes_from_context()),
                    }
                )
            ),
        )

        try:
            with trace_api.use_span(span, end_on_exit=False):
                team_token, team_ctx = _setup_team_context(instance, node_id)
                run_response: RunOutput = wrapped(*args, **kwargs)
            span.set_status(trace_api.StatusCode.OK)
            span.set_attribute(OUTPUT_VALUE, _extract_run_response_output(run_response))
            span.set_attribute(OUTPUT_MIME_TYPE, JSON)

            if hasattr(run_response, "run_id") and run_response.run_id:
                span.set_attribute("agno.run.id", run_response.run_id)

            return run_response

        except Exception as e:
            span.set_status(trace_api.StatusCode.ERROR, str(e))
            span.record_exception(e)
            raise

        finally:
            if team_token:
                try:
                    context_api.detach(team_token)
                except Exception:
                    pass
            span.end()

    def run_stream(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        if hasattr(instance, "name") and instance.name:
            agent_name = instance.name.replace(" ", "_").replace("-", "_")
        else:
            if isinstance(instance, Team):
                agent_name = "Team"
            else:
                agent_name = "Agent"
        span_name = f"{agent_name}.run"

        # Generate unique node ID for this execution
        node_id = _generate_node_id()
        arguments = _bind_arguments(wrapped, *args, **kwargs)

        span = self._tracer.start_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        OPENINFERENCE_SPAN_KIND: AGENT,
                        GRAPH_NODE_ID: node_id,
                        INPUT_VALUE: _get_user_message_content(
                            wrapped,
                            *args,
                            **kwargs,
                        ),
                        **dict(_agent_run_attributes(instance)),
                        **dict(_run_arguments(arguments)),
                        **dict(get_attributes_from_context()),
                    }
                )
            ),
        )

        try:
            current_run_id = None
            yield_run_output_set = False
            if "yield_run_output" not in kwargs:
                yield_run_output_set = True
                kwargs["yield_run_output"] = True  # type: ignore

            run_response = None
            with trace_api.use_span(span, end_on_exit=False):
                team_token, team_ctx = _setup_team_context(instance, node_id)
                for response in wrapped(*args, **kwargs):
                    if hasattr(response, "run_id"):
                        current_run_id = response.run_id
                        if current_run_id:
                            span.set_attribute("agno.run.id", current_run_id)

                    if isinstance(response, (RunOutput, TeamRunOutput)):
                        run_response = response
                        if yield_run_output_set:
                            continue

                    yield response

            if run_response is not None:
                if run_response.content is not None:
                    span.set_attribute(OUTPUT_VALUE, _extract_run_response_output(run_response))
                span.set_attribute(OUTPUT_MIME_TYPE, JSON)
                span.set_status(trace_api.StatusCode.OK)

        except Exception as e:
            span.set_status(trace_api.StatusCode.ERROR, str(e))
            span.record_exception(e)
            raise

        finally:
            if team_token:
                try:
                    context_api.detach(team_token)
                except Exception:
                    pass
            span.end()

    async def arun(
        self,
        wrapped: Callable[..., Awaitable[Any]],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            response = await wrapped(*args, **kwargs)
            return response

        if hasattr(instance, "name") and instance.name:
            agent_name = instance.name.replace(" ", "_").replace("-", "_")
        else:
            if isinstance(instance, Team):
                agent_name = "Team"
            else:
                agent_name = "Agent"
        span_name = f"{agent_name}.arun"

        # Generate unique node ID for this execution
        node_id = _generate_node_id()

        arguments = _bind_arguments(wrapped, *args, **kwargs)

        span = self._tracer.start_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        OPENINFERENCE_SPAN_KIND: AGENT,
                        GRAPH_NODE_ID: node_id,
                        INPUT_VALUE: _get_user_message_content(
                            wrapped,
                            *args,
                            **kwargs,
                        ),
                        **dict(_agent_run_attributes(instance)),
                        **dict(_run_arguments(arguments)),
                        **dict(get_attributes_from_context()),
                    }
                )
            ),
        )

        try:
            with trace_api.use_span(span, end_on_exit=False):
                team_token, team_ctx = _setup_team_context(instance, node_id)
                run_response = await wrapped(*args, **kwargs)
            span.set_status(trace_api.StatusCode.OK)
            span.set_attribute(OUTPUT_VALUE, _extract_run_response_output(run_response))
            span.set_attribute(OUTPUT_MIME_TYPE, JSON)

            if hasattr(run_response, "run_id") and run_response.run_id:
                span.set_attribute("agno.run.id", run_response.run_id)

            return run_response
        except Exception as e:
            span.set_status(trace_api.StatusCode.ERROR, str(e))
            span.record_exception(e)
            raise

        finally:
            if team_token:
                try:
                    context_api.detach(team_token)
                except Exception:
                    pass
            span.end()

    async def arun_stream(
        self,
        wrapped: Callable[..., Awaitable[Any]],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            async for response in await wrapped(*args, **kwargs):
                yield response

        if hasattr(instance, "name") and instance.name:
            agent_name = instance.name.replace(" ", "_").replace("-", "_")
        else:
            if isinstance(instance, Team):
                agent_name = "Team"
            else:
                agent_name = "Agent"
        span_name = f"{agent_name}.arun"

        # Generate unique node ID for this execution
        node_id = _generate_node_id()

        arguments = _bind_arguments(wrapped, *args, **kwargs)

        span = self._tracer.start_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        OPENINFERENCE_SPAN_KIND: AGENT,
                        GRAPH_NODE_ID: node_id,
                        INPUT_VALUE: _get_user_message_content(
                            wrapped,
                            *args,
                            **kwargs,
                        ),
                        **dict(_agent_run_attributes(instance)),
                        **dict(_run_arguments(arguments)),
                        **dict(get_attributes_from_context()),
                    }
                )
            ),
        )

        try:
            current_run_id = None
            yield_run_output_set = False
            if "yield_run_output" not in kwargs:
                yield_run_output_set = True
                kwargs["yield_run_output"] = True  # type: ignore
            run_response = None
            with trace_api.use_span(span, end_on_exit=False):
                team_token, team_ctx = _setup_team_context(instance, node_id)
                async for response in wrapped(*args, **kwargs):  # type: ignore
                    if hasattr(response, "run_id"):
                        current_run_id = response.run_id
                        if current_run_id:
                            span.set_attribute("agno.run.id", current_run_id)

                    if isinstance(response, (RunOutput, TeamRunOutput)):
                        run_response = response
                        if yield_run_output_set:
                            continue

                    yield response

            if run_response is not None:
                if run_response.content is not None:
                    span.set_attribute(OUTPUT_VALUE, _extract_run_response_output(run_response))
                span.set_attribute(OUTPUT_MIME_TYPE, JSON)
                span.set_status(trace_api.StatusCode.OK)

        except Exception as e:
            span.set_status(trace_api.StatusCode.ERROR, str(e))
            span.record_exception(e)
            raise

        finally:
            if team_token:
                try:
                    context_api.detach(team_token)
                except Exception:
                    pass
            span.end()


# span attributes
INPUT_MIME_TYPE = SpanAttributes.INPUT_MIME_TYPE
INPUT_VALUE = SpanAttributes.INPUT_VALUE
SESSION_ID = SpanAttributes.SESSION_ID
OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
OUTPUT_MIME_TYPE = SpanAttributes.OUTPUT_MIME_TYPE
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
USER_ID = SpanAttributes.USER_ID
GRAPH_NODE_ID = SpanAttributes.GRAPH_NODE_ID
GRAPH_NODE_NAME = SpanAttributes.GRAPH_NODE_NAME
GRAPH_NODE_PARENT_ID = SpanAttributes.GRAPH_NODE_PARENT_ID

# message attributes
MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON = MessageAttributes.MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON
MESSAGE_FUNCTION_CALL_NAME = MessageAttributes.MESSAGE_FUNCTION_CALL_NAME

# mime types
TEXT = OpenInferenceMimeTypeValues.TEXT.value
JSON = OpenInferenceMimeTypeValues.JSON.value

# span kinds
AGENT = OpenInferenceSpanKindValues.AGENT.value

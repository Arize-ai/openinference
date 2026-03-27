import json
import logging
from typing import (
    Any,
    Awaitable,
    Callable,
    Iterator,
    List,
    Mapping,
    Optional,
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
from openinference.instrumentation.agno.utils import (
    _AGNO_PARENT_NODE_CONTEXT_KEY,
    _bind_arguments,
    _flatten,
    _generate_node_id,
)
from openinference.semconv.trace import (
    MessageAttributes,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)

logger = logging.getLogger(__name__)


def _get_attr(obj: Any, key: str, default: Any = None) -> Any:
    """Helper function to get attribute from either dict or object."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _get_user_message_content(method: Callable[..., Any], *args: Any, **kwargs: Any) -> str:
    arguments = _bind_arguments(method, *args, **kwargs)
    arguments = _strip_method_args(arguments)

    run_response: Optional[RunOutput] = arguments.get("run_response")
    if run_response and hasattr(run_response, "input") and run_response.input:
        input_content = getattr(run_response.input, "input_content", None)
        if input_content:
            if isinstance(input_content, str):
                return input_content
            elif isinstance(input_content, list):
                return "".join(
                    str(item.content) + "\n" if isinstance(item, Message) else str(item) + "\n"
                    for item in input_content
                )
            elif isinstance(input_content, dict):
                return json.dumps(input_content, indent=2, ensure_ascii=False)
            elif isinstance(input_content, Message):
                return str(input_content.content)
            return str(input_content)

    run_messages: Optional[RunMessages] = arguments.get("run_messages")
    if run_messages and run_messages.user_message:
        return str(run_messages.user_message.content)

    return ""


def _extract_run_response_output(run_response: Union[RunOutput, TeamRunOutput]) -> str:
    if run_response and run_response.content:
        if isinstance(run_response.content, str):
            return run_response.content
        return str(run_response.content.model_dump_json())
    return ""


def _strip_method_args(arguments: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in arguments.items() if key not in ("self", "cls")}


def _run_arguments(arguments: Mapping[str, Any]) -> Iterator[Tuple[str, AttributeValue]]:
    user_id = arguments.get("user_id")
    session_id = arguments.get("session_id")

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
    context_parent_id = context_api.get_value(_AGNO_PARENT_NODE_CONTEXT_KEY)

    if isinstance(agent, Team):
        if agent.name:
            yield GRAPH_NODE_NAME, agent.name
            yield SpanAttributes.AGENT_NAME, agent.name
        if hasattr(agent, "id") and agent.id:
            yield "agno.team.id", agent.id
        if hasattr(agent, "user_id") and agent.user_id:
            yield USER_ID, agent.user_id
        if context_parent_id:
            yield GRAPH_NODE_PARENT_ID, cast(str, context_parent_id)
        yield f"agno{key_suffix}.team", agent.name or ""
        if hasattr(agent, "metadata") and agent.metadata:
            yield METADATA, json.dumps(agent.metadata, default=str)

    elif isinstance(agent, Agent):
        if agent.name:
            yield GRAPH_NODE_NAME, agent.name
            yield SpanAttributes.AGENT_NAME, agent.name
        if hasattr(agent, "id") and agent.id:
            yield "agno.agent.id", agent.id
        if hasattr(agent, "user_id") and agent.user_id:
            yield USER_ID, agent.user_id
        if context_parent_id:
            yield GRAPH_NODE_PARENT_ID, cast(str, context_parent_id)
        if agent.name:
            yield f"agno{key_suffix}.agent", agent.name
        if hasattr(agent, "metadata") and agent.metadata:
            yield METADATA, json.dumps(agent.metadata, default=str)
        if agent.knowledge:
            yield f"agno{key_suffix}.knowledge", agent.knowledge.__class__.__name__
        if agent.tools:
            tools = agent.tools() if callable(agent.tools) else agent.tools
            tool_names: List[str] = []
            for tool in tools:
                if isinstance(tool, Function):
                    tool_names.append(tool.name)
                elif isinstance(tool, Toolkit):
                    tool_names.extend(tool.functions.keys())
                elif callable(tool):
                    tool_names.append(tool.__name__)
                else:
                    tool_names.append(str(tool))
            yield f"agno{key_suffix}.tools", tool_names


def _setup_team_context(
    agent_or_team: Optional[Union[Agent, Team]], node_id: str
) -> Optional[Any]:
    """Attach a context carrying the current team's node_id and return the token.
    Returns None for non-Team instances."""
    if isinstance(agent_or_team, Team):
        team_ctx = context_api.set_value(_AGNO_PARENT_NODE_CONTEXT_KEY, node_id)
        return context_api.attach(team_ctx)
    return None


def _get_agent_or_team(
    instance: Any, args: Tuple[Any, ...], kwargs: Mapping[str, Any]
) -> Optional[Union[Agent, Team]]:
    """Extract the Agent or Team from the arguments.
    For module-level functions: first arg is the Agent/Team."""
    if args and isinstance(args[0], (Agent, Team)):
        return args[0]
    if "agent" in kwargs and isinstance(kwargs["agent"], Agent):
        return kwargs["agent"]
    if "team" in kwargs and isinstance(kwargs["team"], Team):
        return kwargs["team"]
    return None


def _get_team_span_context(agent_or_team: Optional[Union[Agent, Team]]) -> Optional[Context]:
    """Determine the appropriate span context for Team instances.

    Returns:
        - INVALID_SPAN context if this is a top-level Team (no active span, no parent team)
        - None otherwise (nest naturally under the current span)

    This ensures:
    - Sequential team.run() calls create separate top-level traces
    - Nested teams (teams as members) properly nest under parent teams
    """
    if not isinstance(agent_or_team, Team):
        return None
    if trace_api.get_current_span().is_recording():
        return None  # active span exists — nest under it
    if context_api.get_value(_AGNO_PARENT_NODE_CONTEXT_KEY) is None:
        return trace_api.set_span_in_context(trace_api.INVALID_SPAN)
    return None


def _resolve_span_name(agent_or_team: Optional[Union[Agent, Team]], suffix: str) -> str:
    """Build the span name from the agent/team name."""
    if agent_or_team and getattr(agent_or_team, "name", None):
        safe = agent_or_team.name.replace(" ", "_").replace("-", "_")
        return f"{safe}.{suffix}"
    return f"{'Team' if isinstance(agent_or_team, Team) else 'Agent'}.{suffix}"


def _build_span_attributes(
    wrapped: Callable[..., Any],
    agent_or_team: Optional[Union[Agent, Team]],
    node_id: str,
    arguments: Mapping[str, Any],
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    return dict(
        _flatten(
            {
                OPENINFERENCE_SPAN_KIND: AGENT,
                GRAPH_NODE_ID: node_id,
                INPUT_VALUE: _get_user_message_content(wrapped, *args, **kwargs),
                **dict(_agent_run_attributes(agent_or_team) if agent_or_team else {}),
                **dict(_run_arguments(arguments)),
                **dict(get_attributes_from_context()),
            }
        )
    )


def _safe_detach(token: Any, label: str) -> None:
    """Detach a context token, logging any error instead of silently swallowing it."""
    try:
        context_api.detach(token)
    except Exception as e:
        logger.warning("Failed to detach %s context\n%s", label, e)


class _RunWrapper:
    """
    Wraps Agent/Team run methods to emit OpenTelemetry spans.

    Parent/child relationships are tracked via contextvars:
    1. Each run() generates a unique node_id set as GRAPH_NODE_ID on the span.
    2. Team.run() pushes node_id onto _AGNO_PARENT_NODE_CONTEXT_KEY for child agents.
    3. Agent.run() reads _AGNO_PARENT_NODE_CONTEXT_KEY to set GRAPH_NODE_PARENT_ID.
    """

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

        agent_or_team = _get_agent_or_team(instance, args, kwargs)
        node_id = _generate_node_id()
        arguments = _bind_arguments(wrapped, *args, **kwargs)

        span = self._tracer.start_span(
            _resolve_span_name(agent_or_team, "run"),
            context=_get_team_span_context(agent_or_team),
            attributes=_build_span_attributes(wrapped, agent_or_team, node_id, arguments, *args, **kwargs),
        )

        team_token = None
        try:
            with trace_api.use_span(span, end_on_exit=False):
                team_token = _setup_team_context(agent_or_team, node_id)
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
                _safe_detach(team_token, "team (run)")
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

        agent_or_team = _get_agent_or_team(instance, args, kwargs)
        node_id = _generate_node_id()
        arguments = _bind_arguments(wrapped, *args, **kwargs)

        span = self._tracer.start_span(
            _resolve_span_name(agent_or_team, "run"),
            context=_get_team_span_context(agent_or_team),
            attributes=_build_span_attributes(wrapped, agent_or_team, node_id, arguments, *args, **kwargs),
        )

        team_token = None
        span_token = None
        try:
            if kwargs.get("yield_run_output") is not True:
                kwargs["yield_run_output"] = True  # type: ignore
                yield_run_output_set = True
            else:
                yield_run_output_set = False

            run_response = None
            responses_to_yield = []

            # Collect all responses with context attached, but do NOT yield inside
            # the attached context — yielding across a contextvars token boundary
            # causes "Token was created in a different Context" errors.
            span_token = context_api.attach(trace_api.set_span_in_context(span))
            try:
                team_token = _setup_team_context(agent_or_team, node_id)
                for response in wrapped(*args, **kwargs):
                    if hasattr(response, "run_id") and response.run_id:
                        span.set_attribute("agno.run.id", response.run_id)
                    if isinstance(response, (RunOutput, TeamRunOutput)):
                        run_response = response
                        if yield_run_output_set:
                            continue
                    responses_to_yield.append(response)
            finally:
                if team_token:
                    _safe_detach(team_token, "team (run_stream)")
                    team_token = None
                _safe_detach(span_token, "span (run_stream)")
                span_token = None

            if run_response is not None:
                output = _extract_run_response_output(run_response)
                if output:
                    span.set_attribute(OUTPUT_VALUE, output)
                    span.set_attribute(OUTPUT_MIME_TYPE, JSON)
            span.set_status(trace_api.StatusCode.OK)

            yield from responses_to_yield

        except Exception as e:
            span.set_status(trace_api.StatusCode.ERROR, str(e))
            span.record_exception(e)
            raise
        finally:
            # team_token / span_token are always None here (cleared in inner finally),
            # but guard anyway in case the inner block was never entered.
            if team_token:
                _safe_detach(team_token, "team (run_stream fallback)")
            if span_token:
                _safe_detach(span_token, "span (run_stream fallback)")
            span.end()

    async def arun(
        self,
        wrapped: Callable[..., Awaitable[Any]],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return await wrapped(*args, **kwargs)

        agent_or_team = _get_agent_or_team(instance, args, kwargs)
        node_id = _generate_node_id()
        arguments = _bind_arguments(wrapped, *args, **kwargs)

        span = self._tracer.start_span(
            _resolve_span_name(agent_or_team, "arun"),
            context=_get_team_span_context(agent_or_team),
            attributes=_build_span_attributes(wrapped, agent_or_team, node_id, arguments, *args, **kwargs),
        )

        team_token = None
        try:
            with trace_api.use_span(span, end_on_exit=False):
                team_token = _setup_team_context(agent_or_team, node_id)
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
                _safe_detach(team_token, "team (arun)")
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
            return

        agent_or_team = _get_agent_or_team(instance, args, kwargs)
        node_id = _generate_node_id()
        arguments = _bind_arguments(wrapped, *args, **kwargs)

        span = self._tracer.start_span(
            _resolve_span_name(agent_or_team, "arun"),
            context=_get_team_span_context(agent_or_team),
            attributes=_build_span_attributes(wrapped, agent_or_team, node_id, arguments, *args, **kwargs),
        )

        team_token = None
        span_token = None
        try:
            if kwargs.get("yield_run_output") is not True:
                kwargs["yield_run_output"] = True  # type: ignore
                yield_run_output_set = True
            else:
                yield_run_output_set = False

            run_response = None
            responses_to_yield = []

            # Collect all responses with context attached, but do NOT yield inside
            # the attached context — each asyncio Task has its own contextvars.Context
            # copy, so yielding across a token boundary raises "Token was created in
            # a different Context".
            span_token = context_api.attach(trace_api.set_span_in_context(span))
            try:
                team_token = _setup_team_context(agent_or_team, node_id)
                async for response in wrapped(*args, **kwargs):  # type: ignore
                    if hasattr(response, "run_id") and response.run_id:
                        span.set_attribute("agno.run.id", response.run_id)
                    if isinstance(response, (RunOutput, TeamRunOutput)):
                        run_response = response
                        if yield_run_output_set:
                            continue
                    responses_to_yield.append(response)
            finally:
                if team_token:
                    _safe_detach(team_token, "team (arun_stream)")
                    team_token = None
                _safe_detach(span_token, "span (arun_stream)")
                span_token = None

            if run_response is not None:
                output = _extract_run_response_output(run_response)
                if output:
                    span.set_attribute(OUTPUT_VALUE, output)
                    span.set_attribute(OUTPUT_MIME_TYPE, JSON)
            span.set_status(trace_api.StatusCode.OK)

            for response in responses_to_yield:
                yield response

        except Exception as e:
            span.set_status(trace_api.StatusCode.ERROR, str(e))
            span.record_exception(e)
            raise
        finally:
            # team_token / span_token are always None here (cleared in inner finally),
            # but guard anyway in case the inner block was never entered.
            if team_token:
                _safe_detach(team_token, "team (arun_stream fallback)")
            if span_token:
                _safe_detach(span_token, "span (arun_stream fallback)")
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
METADATA = SpanAttributes.METADATA

# message attributes
MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON = MessageAttributes.MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON
MESSAGE_FUNCTION_CALL_NAME = MessageAttributes.MESSAGE_FUNCTION_CALL_NAME

# mime types
TEXT = OpenInferenceMimeTypeValues.TEXT.value
JSON = OpenInferenceMimeTypeValues.JSON.value

# span kinds
AGENT = OpenInferenceSpanKindValues.AGENT.value

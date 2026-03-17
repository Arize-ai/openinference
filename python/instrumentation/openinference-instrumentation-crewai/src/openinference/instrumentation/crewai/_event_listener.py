"""CrewAI event-listener instrumentation built on a shared event assembler."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any, Optional

from opentelemetry import context as context_api
from opentelemetry import trace as trace_api

from crewai.events.base_event_listener import BaseEventListener
from crewai.events.event_bus import CrewAIEventsBus
from crewai.events.types.agent_events import (
    AgentExecutionCompletedEvent,
    AgentExecutionErrorEvent,
    AgentExecutionStartedEvent,
    LiteAgentExecutionCompletedEvent,
    LiteAgentExecutionErrorEvent,
    LiteAgentExecutionStartedEvent,
)
from crewai.events.types.crew_events import (
    CrewKickoffCompletedEvent,
    CrewKickoffFailedEvent,
    CrewKickoffStartedEvent,
)
from crewai.events.types.flow_events import (
    FlowFinishedEvent,
    FlowStartedEvent,
    MethodExecutionFailedEvent,
    MethodExecutionFinishedEvent,
    MethodExecutionStartedEvent,
)
from crewai.events.types.llm_events import (
    LLMCallCompletedEvent,
    LLMCallFailedEvent,
    LLMCallStartedEvent,
)
from crewai.events.types.task_events import (
    TaskCompletedEvent,
    TaskFailedEvent,
    TaskStartedEvent,
)
from crewai.events.types.tool_usage_events import (
    ToolUsageErrorEvent,
    ToolUsageFinishedEvent,
    ToolUsageStartedEvent,
)
from openinference.instrumentation import (
    OITracer,
    TraceConfig,
    get_input_attributes,
    safe_json_dumps,
)
from openinference.instrumentation.crewai._event_assembler import (
    CrewAIEventAssembler,
    _SpanEndSpec,
    _SpanStartSpec,
)
from openinference.instrumentation.crewai._wrappers import SafeJSONEncoder
from openinference.semconv.trace import (
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _get_serialized_input_attributes(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        return dict(get_input_attributes(value, mime_type=OpenInferenceMimeTypeValues.TEXT))
    if isinstance(value, (dict, list, tuple)):
        return dict(get_input_attributes(value, mime_type=OpenInferenceMimeTypeValues.JSON))
    return dict(get_input_attributes(value))


def _get_tool_name(tool: Any) -> str:
    if hasattr(tool, "name") and getattr(tool, "name"):
        return str(getattr(tool, "name"))
    if isinstance(tool, Mapping) and tool.get("name"):
        return str(tool["name"])
    return str(tool)


def _serialize_tool_names(tools: Sequence[Any]) -> str:
    return safe_json_dumps([_get_tool_name(tool) for tool in tools], cls=SafeJSONEncoder)


def _get_flow_method_type(source: Any, method_name: str) -> Optional[str]:
    if method_name in set(getattr(source, "_routers", ()) or ()):
        return "router"
    if method_name in list(getattr(source, "_start_methods", ()) or ()):
        return "start"
    if method_name in dict(getattr(source, "_listeners", {}) or {}):
        return "listen"
    return None


def _build_crew_start_spec(source: Any, event: CrewKickoffStartedEvent) -> _SpanStartSpec:
    crew = getattr(event, "crew", None) or source
    crew_name = str(getattr(event, "crew_name", None) or getattr(crew, "name", None) or "Crew")
    attributes: dict[str, Any] = {}

    inputs = getattr(event, "inputs", None)
    if inputs is not None:
        attributes.update(dict(get_input_attributes(inputs)))
        if isinstance(inputs, Mapping) and "id" in inputs:
            attributes["kickoff_id"] = str(inputs["id"])

    if crew is not None:
        crew_key = getattr(crew, "key", None)
        if crew_key is not None:
            attributes["crew_key"] = str(crew_key)

        crew_id = getattr(crew, "id", None)
        if crew_id is not None:
            attributes["crew_id"] = str(crew_id)

        crew_agents = getattr(crew, "agents", None)
        if crew_agents:
            try:
                attributes["crew_agents"] = safe_json_dumps(
                    [
                        {
                            "key": getattr(agent, "key", ""),
                            "id": str(getattr(agent, "id", "")),
                            "role": getattr(agent, "role", ""),
                            "goal": getattr(agent, "goal", ""),
                            "backstory": getattr(agent, "backstory", ""),
                            "tools_names": [
                                _get_tool_name(tool)
                                for tool in (getattr(agent, "tools", None) or [])
                            ],
                        }
                        for agent in crew_agents
                    ],
                    cls=SafeJSONEncoder,
                )
            except Exception:
                logger.debug("Failed to serialize crew agents", exc_info=True)

        crew_tasks = getattr(crew, "tasks", None)
        if crew_tasks:
            try:
                attributes["crew_tasks"] = safe_json_dumps(
                    [
                        {
                            "id": str(getattr(task, "id", "")),
                            "description": getattr(task, "description", ""),
                            "expected_output": getattr(task, "expected_output", ""),
                            "agent_role": getattr(getattr(task, "agent", None), "role", None),
                        }
                        for task in crew_tasks
                    ],
                    cls=SafeJSONEncoder,
                )
            except Exception:
                logger.debug("Failed to serialize crew tasks", exc_info=True)

    return _SpanStartSpec(
        name=f"{crew_name}.kickoff",
        span_kind=OpenInferenceSpanKindValues.CHAIN,
        attributes=attributes,
    )


def _build_agent_start_spec(event: AgentExecutionStartedEvent) -> _SpanStartSpec:
    agent = getattr(event, "agent", None)
    task = getattr(event, "task", None)
    role = str(getattr(agent, "role", "") or "").strip()
    task_name = str(getattr(task, "name", None) or getattr(task, "description", None) or "").strip()

    if role and task_name:
        span_name = f"{role}.{task_name[:50]}.execute"
    elif role:
        span_name = f"{role}.execute"
    else:
        span_name = "Agent.execute"

    attributes: dict[str, Any] = {}
    task_prompt = getattr(event, "task_prompt", None)
    if task_prompt is not None:
        attributes.update(_get_serialized_input_attributes(task_prompt))

    if agent is not None:
        agent_id = getattr(agent, "id", None)
        if agent_id is not None:
            attributes["agent_id"] = str(agent_id)

        agent_key = getattr(agent, "key", None)
        if agent_key is not None:
            attributes["agent_key"] = str(agent_key)

        if role:
            attributes[SpanAttributes.GRAPH_NODE_ID] = role

        goal = getattr(agent, "goal", None)
        if goal:
            attributes["agent.goal"] = str(goal)

        backstory = getattr(agent, "backstory", None)
        if backstory:
            attributes["agent.backstory"] = str(backstory)

    if task is not None:
        task_id = getattr(task, "id", None)
        if task_id is not None:
            attributes["task_id"] = str(task_id)

        task_name_attr = getattr(task, "name", None)
        if task_name_attr:
            attributes["task_name"] = str(task_name_attr)

        task_description = getattr(task, "description", None)
        if task_description:
            attributes["task_description"] = str(task_description)

        expected_output = getattr(task, "expected_output", None)
        if expected_output:
            attributes["task_expected_output"] = str(expected_output)

    tools = getattr(event, "tools", None)
    if tools:
        try:
            attributes["agent.tools"] = _serialize_tool_names(tools)
        except Exception:
            logger.debug("Failed to serialize agent tools", exc_info=True)

    return _SpanStartSpec(
        name=span_name,
        span_kind=OpenInferenceSpanKindValues.AGENT,
        attributes=attributes,
        remember_as_agent=True,
    )


def _build_lite_agent_start_spec(event: LiteAgentExecutionStartedEvent) -> _SpanStartSpec:
    agent_info = dict(getattr(event, "agent_info", None) or {})
    role = str(agent_info.get("role", "") or "").strip()
    span_name = f"{role}.kickoff" if role else "Agent.kickoff"

    attributes: dict[str, Any] = {}
    messages = getattr(event, "messages", None)
    if messages is not None:
        attributes.update(_get_serialized_input_attributes(messages))

    if role:
        attributes[SpanAttributes.GRAPH_NODE_ID] = role

    agent_id = agent_info.get("id")
    if agent_id:
        attributes["agent_id"] = str(agent_id)

    agent_key = agent_info.get("key")
    if agent_key:
        attributes["agent_key"] = str(agent_key)

    goal = agent_info.get("goal")
    if goal:
        attributes["agent.goal"] = str(goal)

    backstory = agent_info.get("backstory")
    if backstory:
        attributes["agent.backstory"] = str(backstory)

    tools = getattr(event, "tools", None) or agent_info.get("tools")
    if tools:
        try:
            attributes["agent.tools"] = _serialize_tool_names(tools)
        except Exception:
            logger.debug("Failed to serialize LiteAgent tools", exc_info=True)

    return _SpanStartSpec(
        name=span_name,
        span_kind=OpenInferenceSpanKindValues.AGENT,
        attributes=attributes,
        remember_as_agent=True,
    )


def _build_tool_start_spec(event: ToolUsageStartedEvent) -> _SpanStartSpec:
    tool_name = str(getattr(event, "tool_name", None) or "Tool")
    attributes: dict[str, Any] = {
        SpanAttributes.TOOL_NAME: tool_name,
    }

    tool_args = getattr(event, "tool_args", None)
    if tool_args is not None:
        attributes.update(_get_serialized_input_attributes(tool_args))

    tool_class = getattr(event, "tool_class", None)
    if tool_class:
        attributes["tool.class"] = str(tool_class)

    agent_role = getattr(event, "agent_role", None)
    if agent_role:
        attributes["tool.agent_role"] = str(agent_role)

    task_name = getattr(event, "task_name", None)
    if task_name:
        attributes["tool.task_name"] = str(task_name)

    run_attempts = getattr(event, "run_attempts", None)
    if run_attempts is not None:
        attributes["tool.run_attempts"] = int(run_attempts)

    delegations = getattr(event, "delegations", None)
    if delegations is not None:
        attributes["tool.delegations"] = int(delegations)

    return _SpanStartSpec(
        name=f"{tool_name}.run",
        span_kind=OpenInferenceSpanKindValues.TOOL,
        attributes=attributes,
    )


def _build_llm_start_spec(event: LLMCallStartedEvent) -> _SpanStartSpec:
    model = str(getattr(event, "model", None) or "unknown")
    attributes: dict[str, Any] = {}

    if model != "unknown":
        attributes[SpanAttributes.LLM_MODEL_NAME] = model

    messages = getattr(event, "messages", None)
    if messages is not None:
        attributes.update(_get_serialized_input_attributes(messages))

    call_id = getattr(event, "call_id", None)
    if call_id:
        attributes["llm.call_id"] = str(call_id)

    tools = getattr(event, "tools", None)
    if tools:
        try:
            attributes["llm.tools"] = safe_json_dumps(tools, cls=SafeJSONEncoder)
        except Exception:
            logger.debug("Failed to serialize LLM tools", exc_info=True)

    available_functions = getattr(event, "available_functions", None)
    if available_functions:
        try:
            attributes["llm.available_functions"] = safe_json_dumps(
                available_functions, cls=SafeJSONEncoder
            )
        except Exception:
            logger.debug("Failed to serialize LLM available functions", exc_info=True)

    return _SpanStartSpec(
        name=f"{model}.llm_call",
        span_kind=OpenInferenceSpanKindValues.LLM,
        attributes=attributes,
    )


def _build_flow_start_spec(source: Any, event: FlowStartedEvent) -> _SpanStartSpec:
    flow_name = str(getattr(event, "flow_name", None) or getattr(source, "name", None) or "Flow")
    attributes: dict[str, Any] = {}

    inputs = getattr(event, "inputs", None)
    if inputs is not None:
        attributes.update(dict(get_input_attributes(inputs)))

    flow_id = getattr(source, "flow_id", None)
    if flow_id is not None:
        attributes["flow_id"] = str(flow_id)

    return _SpanStartSpec(
        name=f"{flow_name}.kickoff",
        span_kind=OpenInferenceSpanKindValues.CHAIN,
        attributes=attributes,
    )


def _build_method_start_spec(source: Any, event: MethodExecutionStartedEvent) -> _SpanStartSpec:
    flow_name = str(getattr(event, "flow_name", None) or "Flow")
    method_name = str(getattr(event, "method_name", None) or "unknown")
    attributes: dict[str, Any] = {
        "flow.node.name": method_name,
    }

    method_type = _get_flow_method_type(source, method_name)
    if method_type:
        attributes["flow.node.type"] = method_type

    params = getattr(event, "params", None)
    if params:
        attributes.update(_get_serialized_input_attributes(params))

    return _SpanStartSpec(
        name=f"{flow_name}.{method_name}",
        span_kind=OpenInferenceSpanKindValues.CHAIN,
        attributes=attributes,
    )


class OpenInferenceEventListener(BaseEventListener):
    """CrewAI event listener that converts official CrewAI events into OI spans."""

    def __init__(
        self,
        tracer_provider: Optional[trace_api.TracerProvider] = None,
        config: Optional[TraceConfig] = None,
        create_llm_spans: bool = True,
    ) -> None:
        if tracer_provider is None:
            tracer_provider = trace_api.get_tracer_provider()
        if config is None:
            config = TraceConfig()

        from openinference.instrumentation.crewai.version import __version__

        self._tracer = OITracer(
            trace_api.get_tracer(__name__, __version__, tracer_provider),
            config=config,
        )
        self._assembler = CrewAIEventAssembler(tracer=self._tracer)
        self._create_llm_spans = create_llm_spans
        self._handlers: list[tuple[type[Any], Any]] = []
        self._event_bus: Optional[CrewAIEventsBus] = None
        super().__init__()

    def setup_listeners(self, crewai_event_bus: CrewAIEventsBus) -> None:
        self._event_bus = crewai_event_bus
        self._register(CrewKickoffStartedEvent, self._on_crew_started)
        self._register(CrewKickoffCompletedEvent, self._on_crew_completed)
        self._register(CrewKickoffFailedEvent, self._on_crew_failed)

        self._register(TaskStartedEvent, self._on_task_started)
        self._register(TaskCompletedEvent, self._on_task_completed)
        self._register(TaskFailedEvent, self._on_task_failed)

        self._register(AgentExecutionStartedEvent, self._on_agent_started)
        self._register(AgentExecutionCompletedEvent, self._on_agent_completed)
        self._register(AgentExecutionErrorEvent, self._on_agent_error)

        self._register(LiteAgentExecutionStartedEvent, self._on_lite_agent_started)
        self._register(LiteAgentExecutionCompletedEvent, self._on_lite_agent_completed)
        self._register(LiteAgentExecutionErrorEvent, self._on_lite_agent_error)

        self._register(ToolUsageStartedEvent, self._on_tool_started)
        self._register(ToolUsageFinishedEvent, self._on_tool_finished)
        self._register(ToolUsageErrorEvent, self._on_tool_error)

        if self._create_llm_spans:
            self._register(LLMCallStartedEvent, self._on_llm_started)
            self._register(LLMCallCompletedEvent, self._on_llm_completed)
            self._register(LLMCallFailedEvent, self._on_llm_failed)

        self._register(FlowStartedEvent, self._on_flow_started)
        self._register(FlowFinishedEvent, self._on_flow_finished)
        self._register(MethodExecutionStartedEvent, self._on_method_started)
        self._register(MethodExecutionFinishedEvent, self._on_method_finished)
        self._register(MethodExecutionFailedEvent, self._on_method_failed)

    def shutdown(self) -> None:
        if self._event_bus is not None:
            for event_cls, handler in self._handlers:
                try:
                    self._event_bus.off(event_cls, handler)
                except Exception:
                    logger.debug("Failed to unregister handler for %s", event_cls, exc_info=True)
        self._handlers.clear()
        self._assembler.shutdown()

    def _register(self, event_cls: type[Any], handler: Any) -> None:
        if self._event_bus is None:
            raise RuntimeError("Event bus is not initialized")
        decorated = self._event_bus.on(event_cls)(handler)
        self._handlers.append((event_cls, decorated))

    @staticmethod
    def _is_suppressed() -> bool:
        return bool(context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY))

    def _on_crew_started(self, source: Any, event: CrewKickoffStartedEvent) -> None:
        if self._is_suppressed():
            return
        self._assembler.start_span(event, _build_crew_start_spec(source, event))

    def _on_crew_completed(self, source: Any, event: CrewKickoffCompletedEvent) -> None:
        if self._is_suppressed():
            return
        total_tokens = getattr(event, "total_tokens", None)
        attributes = {"total_tokens": total_tokens} if total_tokens is not None else {}
        self._assembler.end_span(
            event,
            _SpanEndSpec(
                output=getattr(event, "output", None),
                attributes=attributes,
            ),
        )

    def _on_crew_failed(self, source: Any, event: CrewKickoffFailedEvent) -> None:
        if self._is_suppressed():
            return
        self._assembler.end_span(
            event,
            _SpanEndSpec(error=getattr(event, "error", None) or "Crew kickoff failed"),
        )

    def _on_task_started(self, source: Any, event: TaskStartedEvent) -> None:
        if self._is_suppressed():
            return
        self._assembler.open_scope(event)

    def _on_task_completed(self, source: Any, event: TaskCompletedEvent) -> None:
        if self._is_suppressed():
            return
        self._assembler.close_scope(event)

    def _on_task_failed(self, source: Any, event: TaskFailedEvent) -> None:
        if self._is_suppressed():
            return
        self._assembler.close_scope(event)

    def _on_agent_started(self, source: Any, event: AgentExecutionStartedEvent) -> None:
        if self._is_suppressed():
            return
        self._assembler.start_span(event, _build_agent_start_spec(event))

    def _on_agent_completed(self, source: Any, event: AgentExecutionCompletedEvent) -> None:
        if self._is_suppressed():
            return
        self._assembler.end_span(
            event,
            _SpanEndSpec(output=getattr(event, "output", None)),
        )

    def _on_agent_error(self, source: Any, event: AgentExecutionErrorEvent) -> None:
        if self._is_suppressed():
            return
        self._assembler.end_span(
            event,
            _SpanEndSpec(error=getattr(event, "error", None) or "Agent execution error"),
        )

    def _on_lite_agent_started(self, source: Any, event: LiteAgentExecutionStartedEvent) -> None:
        if self._is_suppressed():
            return
        self._assembler.start_span(event, _build_lite_agent_start_spec(event))

    def _on_lite_agent_completed(
        self, source: Any, event: LiteAgentExecutionCompletedEvent
    ) -> None:
        if self._is_suppressed():
            return
        self._assembler.end_span(
            event,
            _SpanEndSpec(output=getattr(event, "output", None)),
        )

    def _on_lite_agent_error(self, source: Any, event: LiteAgentExecutionErrorEvent) -> None:
        if self._is_suppressed():
            return
        self._assembler.end_span(
            event,
            _SpanEndSpec(
                error=getattr(event, "error", None) or "LiteAgent execution error",
            ),
        )

    def _on_tool_started(self, source: Any, event: ToolUsageStartedEvent) -> None:
        if self._is_suppressed():
            return
        self._assembler.start_span(event, _build_tool_start_spec(event))

    def _on_tool_finished(self, source: Any, event: ToolUsageFinishedEvent) -> None:
        if self._is_suppressed():
            return
        attributes = {"tool.from_cache": True} if getattr(event, "from_cache", False) else {}
        self._assembler.end_span(
            event,
            _SpanEndSpec(
                output=getattr(event, "output", None),
                attributes=attributes,
            ),
        )

    def _on_tool_error(self, source: Any, event: ToolUsageErrorEvent) -> None:
        if self._is_suppressed():
            return
        self._assembler.end_span(
            event,
            _SpanEndSpec(error=str(getattr(event, "error", None) or "Tool usage error")),
        )

    def _on_llm_started(self, source: Any, event: LLMCallStartedEvent) -> None:
        if self._is_suppressed():
            return
        self._assembler.start_span(event, _build_llm_start_spec(event))

    def _on_llm_completed(self, source: Any, event: LLMCallCompletedEvent) -> None:
        if self._is_suppressed():
            return
        call_type = getattr(event, "call_type", None)
        attributes = {"llm.call_type": call_type.value} if call_type is not None else {}
        self._assembler.end_span(
            event,
            _SpanEndSpec(
                output=getattr(event, "response", None),
                attributes=attributes,
            ),
        )

    def _on_llm_failed(self, source: Any, event: LLMCallFailedEvent) -> None:
        if self._is_suppressed():
            return
        self._assembler.end_span(
            event,
            _SpanEndSpec(error=getattr(event, "error", None) or "LLM call failed"),
        )

    def _on_flow_started(self, source: Any, event: FlowStartedEvent) -> None:
        if self._is_suppressed():
            return
        self._assembler.start_span(event, _build_flow_start_spec(source, event))

    def _on_flow_finished(self, source: Any, event: FlowFinishedEvent) -> None:
        if self._is_suppressed():
            return
        self._assembler.end_span(
            event,
            _SpanEndSpec(output=getattr(event, "result", None)),
        )

    def _on_method_started(self, source: Any, event: MethodExecutionStartedEvent) -> None:
        if self._is_suppressed():
            return
        self._assembler.start_span(event, _build_method_start_spec(source, event))

    def _on_method_finished(self, source: Any, event: MethodExecutionFinishedEvent) -> None:
        if self._is_suppressed():
            return
        self._assembler.end_span(
            event,
            _SpanEndSpec(output=getattr(event, "result", None)),
        )

    def _on_method_failed(self, source: Any, event: MethodExecutionFailedEvent) -> None:
        if self._is_suppressed():
            return
        self._assembler.end_span(
            event,
            _SpanEndSpec(error=str(getattr(event, "error", None) or "Method execution failed")),
        )

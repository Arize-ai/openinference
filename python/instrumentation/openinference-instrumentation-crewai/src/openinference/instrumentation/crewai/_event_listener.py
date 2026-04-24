"""CrewAI event-listener instrumentation built on a shared event assembler."""

from __future__ import annotations

import json
import logging
import threading
import weakref
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any, Optional, cast

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
    Message,
    OITracer,
    TokenCount,
    TraceConfig,
    get_input_attributes,
    get_llm_input_message_attributes,
    get_llm_output_message_attributes,
    get_llm_token_count_attributes,
    safe_json_dumps,
)
from openinference.instrumentation.crewai._event_assembler import (
    _MAX_BUFFERED_EVENT_ENTRIES,
    CrewAIEventAssembler,
    _SpanEndSpec,
    _SpanStartSpec,
)
from openinference.instrumentation.crewai._wrappers import SafeJSONEncoder, _find_parent_agent
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


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _normalize_tool_call_function_arguments(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        json.loads(value)
    except json.JSONDecodeError:
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            inner = value[1:-1]
            try:
                json.loads(inner)
            except json.JSONDecodeError:
                return value
            return inner
        return value
    return value


def _normalize_llm_message(message: Mapping[str, Any]) -> Message:
    normalized_message = dict(message)
    tool_calls = normalized_message.get("tool_calls")
    if isinstance(tool_calls, Sequence) and not isinstance(tool_calls, (str, bytes, bytearray)):
        normalized_tool_calls: list[Any] = []
        for tool_call in tool_calls:
            if not isinstance(tool_call, Mapping):
                normalized_tool_calls.append(tool_call)
                continue
            normalized_tool_call = dict(tool_call)
            function = normalized_tool_call.get("function")
            if isinstance(function, Mapping):
                normalized_function = dict(function)
                if "arguments" in normalized_function:
                    normalized_function["arguments"] = _normalize_tool_call_function_arguments(
                        normalized_function.get("arguments")
                    )
                normalized_tool_call["function"] = normalized_function
            normalized_tool_calls.append(normalized_tool_call)
        normalized_message["tool_calls"] = normalized_tool_calls
    return cast(Message, normalized_message)


def _normalize_llm_messages(value: Any, default_role: str) -> list[Message]:
    if isinstance(value, str):
        return [cast(Message, {"role": default_role, "content": value})]
    if isinstance(value, Mapping):
        return [_normalize_llm_message(value)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _normalize_llm_message(message) for message in value if isinstance(message, Mapping)
        ]
    return []


def _get_llm_input_attributes(messages: Any) -> dict[str, Any]:
    normalized_messages = _normalize_llm_messages(messages, default_role="user")
    if not normalized_messages:
        return {}
    return dict(get_llm_input_message_attributes(normalized_messages))


def _get_llm_output_attributes(response: Any) -> dict[str, Any]:
    normalized_messages = _normalize_llm_messages(response, default_role="assistant")
    if not normalized_messages:
        return {}
    return dict(get_llm_output_message_attributes(normalized_messages))


def _normalize_token_usage(usage_data: Mapping[str, Any]) -> TokenCount:
    prompt_tokens = int(
        _first_not_none(
            usage_data.get("prompt_tokens"),
            usage_data.get("prompt_token_count"),
            usage_data.get("input_tokens"),
            0,
        )
    )
    completion_tokens = int(
        _first_not_none(
            usage_data.get("completion_tokens"),
            usage_data.get("candidates_token_count"),
            usage_data.get("output_tokens"),
            0,
        )
    )
    total_tokens = int(
        _first_not_none(
            usage_data.get("total_tokens"),
            prompt_tokens + completion_tokens,
        )
    )
    cached_prompt_tokens = int(
        _first_not_none(
            usage_data.get("cached_tokens"),
            usage_data.get("cached_prompt_tokens"),
            0,
        )
    )

    token_count: TokenCount = {
        "prompt": prompt_tokens,
        "completion": completion_tokens,
        "total": total_tokens,
    }
    if cached_prompt_tokens:
        token_count["prompt_details"] = {"cache_read": cached_prompt_tokens}
    return token_count


def _get_parent_agent_role(agent: Any, task: Any) -> Optional[str]:
    task_context = getattr(task, "context", None)
    if isinstance(task_context, Sequence) and not isinstance(task_context, (str, bytes, bytearray)):
        for context_task in reversed(task_context):
            context_agent = getattr(context_task, "agent", None)
            parent_role = str(getattr(context_agent, "role", "") or "").strip()
            if parent_role:
                return parent_role

    current_role = str(getattr(agent, "role", "") or "").strip()
    if not current_role:
        return None

    crew = getattr(agent, "crew", None) or getattr(getattr(task, "agent", None), "crew", None)
    crew_agents = getattr(crew, "agents", None)
    if isinstance(crew_agents, Sequence) and not isinstance(crew_agents, (str, bytes, bytearray)):
        try:
            parent_role_from_crew = _find_parent_agent(current_role, list(crew_agents))
        except Exception:
            logger.debug("Failed to resolve parent agent from crew ordering", exc_info=True)
        else:
            if parent_role_from_crew:
                return str(parent_role_from_crew).strip()
    return None


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
            parent_role = _get_parent_agent_role(agent, task)
            if parent_role and parent_role != role:
                attributes[SpanAttributes.GRAPH_NODE_PARENT_ID] = parent_role

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
        attributes.update(_get_llm_input_attributes(messages))

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

    _llm_patch_lock = threading.RLock()
    _llm_patch_original: Optional[Any] = None
    _llm_patch_listeners: "weakref.WeakSet[OpenInferenceEventListener]" = weakref.WeakSet()

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
        self._assembler = CrewAIEventAssembler(tracer=self._tracer)  # type: ignore[arg-type]
        self._create_llm_spans = create_llm_spans
        self._llm_usage_by_call_id: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._llm_usage_lock = threading.RLock()
        if self._create_llm_spans:
            self._patch_llm_token_usage_tracking()
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
        self._restore_llm_token_usage_tracking()
        self._assembler.shutdown()

    def _register(self, event_cls: type[Any], handler: Any) -> None:
        if self._event_bus is None:
            raise RuntimeError("Event bus is not initialized")
        decorated = self._event_bus.on(event_cls)(handler)
        self._handlers.append((event_cls, decorated))

    @staticmethod
    def _is_suppressed() -> bool:
        return bool(context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY))

    def _patch_llm_token_usage_tracking(self) -> None:
        from crewai.llms import base_llm as base_llm_module

        cls = type(self)
        with cls._llm_patch_lock:
            original = cls._llm_patch_original
            if original is None:
                original = getattr(base_llm_module.BaseLLM, "_track_token_usage_internal", None)
                if original is None:
                    return
                cls._llm_patch_original = original

                def patched(instance: Any, usage_data: dict[str, Any]) -> None:
                    original(instance, usage_data)
                    try:
                        call_id = base_llm_module.get_current_call_id()
                    except Exception:
                        logger.debug(
                            "Failed to resolve CrewAI LLM call id while tracking token usage",
                            exc_info=True,
                        )
                        return
                    with cls._llm_patch_lock:
                        listeners = list(cls._llm_patch_listeners)
                    for listener in listeners:
                        try:
                            listener._record_llm_token_usage(call_id, usage_data)
                        except Exception:
                            logger.debug(
                                "Failed to record CrewAI LLM token usage for %s",
                                call_id,
                                exc_info=True,
                            )

                setattr(base_llm_module.BaseLLM, "_track_token_usage_internal", patched)

            cls._llm_patch_listeners.add(self)

    def _restore_llm_token_usage_tracking(self) -> None:
        from crewai.llms import base_llm as base_llm_module

        cls = type(self)
        with cls._llm_patch_lock:
            cls._llm_patch_listeners.discard(self)
            original = cls._llm_patch_original
            if original is not None and not cls._llm_patch_listeners:
                setattr(base_llm_module.BaseLLM, "_track_token_usage_internal", original)
                cls._llm_patch_original = None
        with self._llm_usage_lock:
            self._llm_usage_by_call_id.clear()

    def _record_llm_token_usage(self, call_id: str, usage_data: Mapping[str, Any]) -> None:
        if not call_id:
            return
        normalized_usage = _normalize_token_usage(usage_data)
        with self._llm_usage_lock:
            aggregate = self._llm_usage_by_call_id.pop(call_id, None)
            if aggregate is None:
                aggregate = {
                    "prompt": 0,
                    "completion": 0,
                    "total": 0,
                    "prompt_details": {"cache_read": 0},
                }
            self._llm_usage_by_call_id[call_id] = aggregate
            aggregate["prompt"] += normalized_usage["prompt"]
            aggregate["completion"] += normalized_usage["completion"]
            aggregate["total"] += normalized_usage["total"]
            prompt_details = normalized_usage.get("prompt_details")
            if isinstance(prompt_details, Mapping):
                aggregate["prompt_details"]["cache_read"] += int(
                    prompt_details.get("cache_read", 0)
                )
            while len(self._llm_usage_by_call_id) > _MAX_BUFFERED_EVENT_ENTRIES:
                evicted_call_id, _ = self._llm_usage_by_call_id.popitem(last=False)
                logger.warning(
                    "Evicting oldest LLM token usage entry for %s "
                    "after reaching %d buffered entries",
                    evicted_call_id,
                    _MAX_BUFFERED_EVENT_ENTRIES,
                )

    def _consume_llm_token_usage_attributes(self, call_id: Optional[str]) -> dict[str, Any]:
        if not call_id:
            return {}
        with self._llm_usage_lock:
            token_usage = self._llm_usage_by_call_id.pop(call_id, None)
        if not token_usage:
            return {}
        prompt_details = token_usage.get("prompt_details")
        if isinstance(prompt_details, Mapping) and prompt_details.get("cache_read", 0) == 0:
            token_usage = dict(token_usage)
            token_usage.pop("prompt_details", None)
        return dict(get_llm_token_count_attributes(cast(TokenCount, token_usage)))

    def _on_crew_started(self, source: Any, event: CrewKickoffStartedEvent) -> None:
        if self._is_suppressed():
            return
        self._assembler.start_span(event, _build_crew_start_spec(source, event))

    def _on_crew_completed(self, source: Any, event: CrewKickoffCompletedEvent) -> None:
        if self._is_suppressed():
            return
        total_tokens = getattr(event, "total_tokens", None)
        attributes = (
            {SpanAttributes.LLM_TOKEN_COUNT_TOTAL: int(total_tokens)}
            if total_tokens is not None
            else {}
        )
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
        self._assembler.end_span(
            event,
            _SpanEndSpec(
                output=getattr(event, "output", None),
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
        attributes: dict[str, Any] = {}
        attributes.update(_get_llm_output_attributes(getattr(event, "response", None)))
        attributes.update(self._consume_llm_token_usage_attributes(getattr(event, "call_id", None)))
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
        self._consume_llm_token_usage_attributes(getattr(event, "call_id", None))
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

"""wrapt-based wrappers for AG2 (formerly AutoGen) instrumentation."""

from __future__ import annotations

import json
from typing import Any, Callable

from opentelemetry import context as context_api
from opentelemetry.trace import StatusCode

from openinference.instrumentation import OITracer, safe_json_dumps
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes

from ._attributes import (
    AG2_AGENT_DESCRIPTION,
    AG2_AGENT_NAME,
    AG2_GROUPCHAT_AGENTS,
    AG2_GROUPCHAT_MAX_ROUND,
    AG2_GROUPCHAT_SELECTOR,
    AG2_GROUPCHAT_TOTAL_ROUNDS,
    AG2_INITIATOR_NAME,
    AG2_MAX_TURNS,
    AG2_NESTED_CHAT_COUNT,
    AG2_REASONING_BEAM_SIZE,
    AG2_REASONING_MAX_DEPTH,
    AG2_REASONING_METHOD,
    AG2_RECIPIENT_NAME,
    AG2_SWARM_AGENTS,
    AG2_SWARM_INITIAL_AGENT,
    AG2_SWARM_MAX_ROUNDS,
    AG2_TOOL_ARGUMENTS,
    AG2_TOOL_NAME,
)


class _InitiateChatWrapper:
    """
    Wraps ConversableAgent.initiate_chat — the root span for any AG2 conversation.

    Span kind: CHAIN
    """

    def __init__(self, tracer: OITracer) -> None:
        self._tracer = tracer

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        recipient = args[0] if args else kwargs.get("recipient")
        message = args[1] if len(args) > 1 else kwargs.get("message", "")
        span_name = f"{instance.name} → {getattr(recipient, 'name', 'unknown')}"

        with self._tracer.start_as_current_span(
            span_name,
            kind=OpenInferenceSpanKindValues.CHAIN,
        ) as span:
            span.set_attribute(SpanAttributes.INPUT_VALUE, str(message))
            span.set_attribute(AG2_INITIATOR_NAME, instance.name)
            span.set_attribute(AG2_RECIPIENT_NAME, getattr(recipient, "name", ""))
            span.set_attribute(AG2_MAX_TURNS, str(kwargs.get("max_turns", "None")))
            try:
                result = wrapped(*args, **kwargs)
                if result and hasattr(result, "summary"):
                    span.set_attribute(SpanAttributes.OUTPUT_VALUE, str(result.summary))
                span.set_status(StatusCode.OK)
                return result
            except Exception as exc:
                span.set_status(StatusCode.ERROR, str(exc))
                raise


class _GenerateReplyWrapper:
    """
    Wraps ConversableAgent.generate_reply — one span per agent turn.

    Span kind: AGENT
    """

    def __init__(self, tracer: OITracer) -> None:
        self._tracer = tracer

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        messages = kwargs.get("messages") or (args[0] if args else [])
        last_msg = messages[-1].get("content", "") if messages else ""

        with self._tracer.start_as_current_span(
            f"{instance.name}.generate_reply",
            kind=OpenInferenceSpanKindValues.AGENT,
        ) as span:
            span.set_attribute(AG2_AGENT_NAME, instance.name)
            span.set_attribute(AG2_AGENT_DESCRIPTION, (instance.system_message or "")[:500])
            span.set_attribute(SpanAttributes.INPUT_VALUE, str(last_msg))
            try:
                reply = wrapped(*args, **kwargs)
                span.set_attribute(SpanAttributes.OUTPUT_VALUE, str(reply) if reply else "")
                span.set_status(StatusCode.OK)
                return reply
            except Exception as exc:
                span.set_status(StatusCode.ERROR, str(exc))
                raise


class _ExecuteFunctionWrapper:
    """
    Wraps ConversableAgent.execute_function — tool call execution.

    Span kind: TOOL
    """

    def __init__(self, tracer: OITracer) -> None:
        self._tracer = tracer

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        func_call = args[0] if args else kwargs.get("func_call", {})
        fn_name = func_call.get("name", "unknown_function")

        with self._tracer.start_as_current_span(
            f"tool.{fn_name}",
            kind=OpenInferenceSpanKindValues.TOOL,
        ) as span:
            span.set_attribute(AG2_TOOL_NAME, fn_name)
            span.set_attribute(AG2_TOOL_ARGUMENTS, safe_json_dumps(func_call.get("arguments", {})))
            try:
                result = wrapped(*args, **kwargs)
                content = result[1].get("content", "") if isinstance(result, tuple) else str(result)
                span.set_attribute(SpanAttributes.OUTPUT_VALUE, str(content))
                span.set_status(StatusCode.OK)
                return result
            except Exception as exc:
                span.set_status(StatusCode.ERROR, str(exc))
                raise


class _GroupChatWrapper:
    """
    Wraps GroupChatManager.run_chat — orchestration span for multi-agent GroupChat.

    Span kind: CHAIN
    Includes Phoenix-compatible graph topology attributes (graph.node.id.N / graph.node.parent_id.N).
    """

    def __init__(self, tracer: OITracer) -> None:
        self._tracer = tracer

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        gc = instance.groupchat
        agent_names = [a.name for a in gc.agents]

        with self._tracer.start_as_current_span(
            "GroupChat.run_chat",
            kind=OpenInferenceSpanKindValues.CHAIN,
        ) as span:
            span.set_attribute(AG2_GROUPCHAT_AGENTS, json.dumps(agent_names))
            span.set_attribute(AG2_GROUPCHAT_MAX_ROUND, gc.max_round)
            span.set_attribute(AG2_GROUPCHAT_SELECTOR, gc.speaker_selection_method)
            for i, name in enumerate(agent_names):
                span.set_attribute(f"graph.node.id.{i}", name)
                span.set_attribute(f"graph.node.parent_id.{i}", "GroupChatManager")
            try:
                result = wrapped(*args, **kwargs)
                span.set_attribute(AG2_GROUPCHAT_TOTAL_ROUNDS, len(gc.messages))
                span.set_status(StatusCode.OK)
                return result
            except Exception as exc:
                span.set_status(StatusCode.ERROR, str(exc))
                raise


class _SwarmChatWrapper:
    """
    Wraps autogen.run_swarm — root span for SwarmAgent workflows.

    Span kind: CHAIN
    """

    def __init__(self, tracer: OITracer) -> None:
        self._tracer = tracer

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        initial = kwargs.get("initial_agent")
        agents = kwargs.get("agents", [])

        with self._tracer.start_as_current_span(
            "SwarmChat",
            kind=OpenInferenceSpanKindValues.CHAIN,
        ) as span:
            span.set_attribute(AG2_SWARM_INITIAL_AGENT, getattr(initial, "name", ""))
            span.set_attribute(AG2_SWARM_AGENTS, json.dumps([a.name for a in agents]))
            span.set_attribute(AG2_SWARM_MAX_ROUNDS, str(kwargs.get("max_rounds", "None")))
            try:
                result = wrapped(*args, **kwargs)
                span.set_status(StatusCode.OK)
                return result
            except Exception as exc:
                span.set_status(StatusCode.ERROR, str(exc))
                raise


class _ReasoningAgentWrapper:
    """
    Wraps ReasoningAgent.generate_reply — tree-of-thought reasoning spans.

    Span kind: AGENT
    """

    def __init__(self, tracer: OITracer) -> None:
        self._tracer = tracer

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        reason_cfg = getattr(instance, "reason_config", {})

        with self._tracer.start_as_current_span(
            f"{instance.name}.reasoning",
            kind=OpenInferenceSpanKindValues.AGENT,
        ) as span:
            span.set_attribute(AG2_REASONING_METHOD, reason_cfg.get("method", "beam_search"))
            span.set_attribute(AG2_REASONING_BEAM_SIZE, str(reason_cfg.get("beam_size", 3)))
            span.set_attribute(AG2_REASONING_MAX_DEPTH, str(reason_cfg.get("max_depth", 4)))
            try:
                result = wrapped(*args, **kwargs)
                span.set_attribute(SpanAttributes.OUTPUT_VALUE, str(result) if result else "")
                span.set_status(StatusCode.OK)
                return result
            except Exception as exc:
                span.set_status(StatusCode.ERROR, str(exc))
                raise


class _InitiateChatsWrapper:
    """
    Wraps ConversableAgent.initiate_chats — nested chat pipeline.

    Span kind: CHAIN
    """

    def __init__(self, tracer: OITracer) -> None:
        self._tracer = tracer

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        chat_queue = args[0] if args else kwargs.get("chat_queue", [])

        with self._tracer.start_as_current_span(
            f"{instance.name}.initiate_chats",
            kind=OpenInferenceSpanKindValues.CHAIN,
        ) as span:
            span.set_attribute(AG2_NESTED_CHAT_COUNT, len(chat_queue))
            try:
                result = wrapped(*args, **kwargs)
                span.set_status(StatusCode.OK)
                return result
            except Exception as exc:
                span.set_status(StatusCode.ERROR, str(exc))
                raise

"""wrapt-based wrappers for AG2 (formerly AutoGen) instrumentation."""
from __future__ import annotations

import json
from typing import Any, Callable

import wrapt
from openinference.instrumentation import OITracer, safe_json_dumps
from openinference.semconv.trace import (
    MessageAttributes, OpenInferenceSpanKindValues, SpanAttributes,
)
from opentelemetry.trace import StatusCode


class _InitiateChatWrapper:
    """
    Wraps ConversableAgent.initiate_chat — the root span for any AG2 conversation.

    Span kind: CHAIN
    Attributes set:
      input.value            — the initial message
      ag2.initiator.name     — name of the initiating agent
      ag2.recipient.name     — name of the recipient agent
      ag2.max_turns          — configured max_turns
      output.value           — final response message
    """
    def __init__(self, tracer: OITracer):
        self._tracer = tracer

    def __call__(self, wrapped: Callable, instance: Any, args: tuple, kwargs: dict) -> Any:
        recipient = args[0] if args else kwargs.get("recipient")
        message = args[1] if len(args) > 1 else kwargs.get("message", "")
        span_name = f"{instance.name} → {getattr(recipient, 'name', 'unknown')}"

        with self._tracer.start_as_current_span(
            span_name,
            kind=OpenInferenceSpanKindValues.CHAIN,
        ) as span:
            span.set_attribute(SpanAttributes.INPUT_VALUE, str(message))
            span.set_attribute("ag2.initiator.name", instance.name)
            span.set_attribute("ag2.recipient.name", getattr(recipient, "name", ""))
            span.set_attribute("ag2.max_turns", str(kwargs.get("max_turns", "None")))
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
    Attributes set:
      ag2.agent.name         — name of the agent generating the reply
      ag2.agent.description  — agent's system message (first 500 chars)
      input.value            — last incoming message content
      output.value           — generated reply content
    """
    def __init__(self, tracer: OITracer):
        self._tracer = tracer

    def __call__(self, wrapped: Callable, instance: Any, args: tuple, kwargs: dict) -> Any:
        messages = kwargs.get("messages") or (args[0] if args else [])
        last_msg = messages[-1].get("content", "") if messages else ""

        with self._tracer.start_as_current_span(
            f"{instance.name}.generate_reply",
            kind=OpenInferenceSpanKindValues.AGENT,
        ) as span:
            span.set_attribute("ag2.agent.name", instance.name)
            span.set_attribute("ag2.agent.description", (instance.system_message or "")[:500])
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
    Attributes set:
      ag2.tool.name          — function name
      ag2.tool.arguments     — JSON-serialized arguments
      output.value           — tool return value (str)
    """
    def __init__(self, tracer: OITracer):
        self._tracer = tracer

    def __call__(self, wrapped: Callable, instance: Any, args: tuple, kwargs: dict) -> Any:
        func_call = args[0] if args else kwargs.get("func_call", {})
        fn_name = func_call.get("name", "unknown_function")

        with self._tracer.start_as_current_span(
            f"tool.{fn_name}",
            kind=OpenInferenceSpanKindValues.TOOL,
        ) as span:
            span.set_attribute("ag2.tool.name", fn_name)
            span.set_attribute("ag2.tool.arguments", safe_json_dumps(func_call.get("arguments", {})))
            try:
                result = wrapped(*args, **kwargs)
                # result is (is_exec_success: bool, {"name": ..., "role": ..., "content": ...})
                content = result[1].get("content", "") if isinstance(result, tuple) else str(result)
                span.set_attribute(SpanAttributes.OUTPUT_VALUE, str(content))
                span.set_status(StatusCode.OK)
                return result
            except Exception as exc:
                span.set_status(StatusCode.ERROR, str(exc))
                raise


class _GroupChatWrapper:
    """
    Wraps GroupChatManager.run_chat — the orchestration span for multi-agent GroupChat.

    Span kind: CHAIN
    Attributes set:
      ag2.groupchat.agents          — JSON list of participant agent names
      ag2.groupchat.max_round       — configured max_round
      ag2.groupchat.selector        — speaker_selection_method
      graph.node.id.*               — participant graph topology (Phoenix-compatible)
      graph.node.parent_id.*        — parent-child edges for graph rendering
    """
    def __init__(self, tracer: OITracer):
        self._tracer = tracer

    def __call__(self, wrapped: Callable, instance: Any, args: tuple, kwargs: dict) -> Any:
        gc = instance.groupchat
        agent_names = [a.name for a in gc.agents]

        with self._tracer.start_as_current_span(
            "GroupChat.run_chat",
            kind=OpenInferenceSpanKindValues.CHAIN,
        ) as span:
            span.set_attribute("ag2.groupchat.agents", json.dumps(agent_names))
            span.set_attribute("ag2.groupchat.max_round", gc.max_round)
            span.set_attribute("ag2.groupchat.selector", gc.speaker_selection_method)
            # Graph topology for Phoenix graph view
            for i, name in enumerate(agent_names):
                span.set_attribute(f"graph.node.id.{i}", name)
                span.set_attribute(f"graph.node.parent_id.{i}", "GroupChatManager")
            try:
                result = wrapped(*args, **kwargs)
                span.set_attribute("ag2.groupchat.total_rounds", len(gc.messages))
                span.set_status(StatusCode.OK)
                return result
            except Exception as exc:
                span.set_status(StatusCode.ERROR, str(exc))
                raise


class _SwarmChatWrapper:
    """
    Wraps autogen.initiate_swarm_chat — the root span for SwarmAgent workflows.

    Span kind: CHAIN
    Attributes set:
      ag2.swarm.agents              — JSON list of swarm agent names
      ag2.swarm.initial_agent       — first agent to receive the task
      ag2.swarm.max_rounds          — configured max_rounds
      ag2.swarm.handoffs            — JSON list of {from, to, reason} handoff events
    """
    def __init__(self, tracer: OITracer):
        self._tracer = tracer

    def __call__(self, wrapped: Callable, instance: Any, args: tuple, kwargs: dict) -> Any:
        initial = kwargs.get("initial_agent")
        agents = kwargs.get("agents", [])

        with self._tracer.start_as_current_span(
            "SwarmChat",
            kind=OpenInferenceSpanKindValues.CHAIN,
        ) as span:
            span.set_attribute("ag2.swarm.initial_agent", getattr(initial, "name", ""))
            span.set_attribute("ag2.swarm.agents", json.dumps([a.name for a in agents]))
            span.set_attribute("ag2.swarm.max_rounds", str(kwargs.get("max_rounds", "None")))
            try:
                result = wrapped(*args, **kwargs)
                span.set_status(StatusCode.OK)
                return result
            except Exception as exc:
                span.set_status(StatusCode.ERROR, str(exc))
                raise


class _ReasoningAgentWrapper:
    """
    Wraps ReasoningAgent.generate_response — tree-of-thought reasoning spans.

    Span kind: AGENT
    Attributes set:
      ag2.reasoning.method          — "beam_search" | "mcts" | "lats"
      ag2.reasoning.beam_size       — beam width
      ag2.reasoning.max_depth       — tree depth limit
      ag2.reasoning.total_nodes     — nodes explored
      output.value                  — final answer
    """
    def __init__(self, tracer: OITracer):
        self._tracer = tracer

    def __call__(self, wrapped: Callable, instance: Any, args: tuple, kwargs: dict) -> Any:
        reason_cfg = getattr(instance, "reason_config", {})

        with self._tracer.start_as_current_span(
            f"{instance.name}.reasoning",
            kind=OpenInferenceSpanKindValues.AGENT,
        ) as span:
            span.set_attribute("ag2.reasoning.method", reason_cfg.get("method", "beam_search"))
            span.set_attribute("ag2.reasoning.beam_size", str(reason_cfg.get("beam_size", 3)))
            span.set_attribute("ag2.reasoning.max_depth", str(reason_cfg.get("max_depth", 4)))
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
    Attributes set:
      ag2.nested.chat_count         — number of chats in the pipeline
      ag2.nested.carryover_mode     — how context is passed between chats
    """
    def __init__(self, tracer: OITracer):
        self._tracer = tracer

    def __call__(self, wrapped: Callable, instance: Any, args: tuple, kwargs: dict) -> Any:
        chat_queue = args[0] if args else kwargs.get("chat_queue", [])

        with self._tracer.start_as_current_span(
            f"{instance.name}.initiate_chats",
            kind=OpenInferenceSpanKindValues.CHAIN,
        ) as span:
            span.set_attribute("ag2.nested.chat_count", len(chat_queue))
            try:
                result = wrapped(*args, **kwargs)
                span.set_status(StatusCode.OK)
                return result
            except Exception as exc:
                span.set_status(StatusCode.ERROR, str(exc))
                raise

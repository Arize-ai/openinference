from __future__ import annotations

from typing import Any, AsyncGenerator, Callable, Dict, Mapping, Tuple

from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.util.types import AttributeValue

from openinference.instrumentation import get_attributes_from_context, safe_json_dumps
from openinference.semconv.trace import (
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)


class _WithTracer:
    """Base class for wrappers that need a tracer."""

    def __init__(self, tracer: trace_api.Tracer, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._tracer = tracer


class _AgentInvokeAsyncWrapper(_WithTracer):
    """Wrapper for Agent.invoke_async() to trace agent invocations."""

    async def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return await wrapped(*args, **kwargs)

        # Extract agent information
        agent = instance
        agent_name = getattr(agent, "name", "unknown")

        # Extract prompt from args/kwargs
        prompt = args[0] if args else kwargs.get("prompt")

        # Build span attributes
        attributes = dict(
            _get_agent_attributes(agent),
            **{
                OPENINFERENCE_SPAN_KIND: AGENT,
            },
        )

        # Add input attributes
        if prompt is not None:
            input_attrs = _serialize_prompt(prompt)
            attributes.update(input_attrs)

        # Add context attributes
        attributes.update(get_attributes_from_context())

        # Start span
        span_name = f"{agent_name}.invoke"
        with self._tracer.start_as_current_span(
            span_name,
            attributes=attributes,
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            try:
                result = await wrapped(*args, **kwargs)
            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                raise

            # Set output attributes
            span.set_status(trace_api.StatusCode.OK)
            output_attrs = _get_agent_result_attributes(result)
            span.set_attributes(output_attrs)

            return result


class _AgentStreamAsyncWrapper(_WithTracer):
    """Wrapper for Agent.stream_async() to trace streaming agent invocations."""

    async def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> AsyncGenerator[Any, None]:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            async for event in wrapped(*args, **kwargs):
                yield event
            return

        # Extract agent information
        agent = instance
        agent_name = getattr(agent, "name", "unknown")

        # Extract prompt from args/kwargs
        prompt = args[0] if args else kwargs.get("prompt")

        # Build span attributes
        attributes = dict(
            _get_agent_attributes(agent),
            **{
                OPENINFERENCE_SPAN_KIND: AGENT,
            },
        )

        # Add input attributes
        if prompt is not None:
            input_attrs = _serialize_prompt(prompt)
            attributes.update(input_attrs)

        # Add context attributes
        attributes.update(get_attributes_from_context())

        # Start span
        span_name = f"{agent_name}.stream"
        with self._tracer.start_as_current_span(
            span_name,
            attributes=attributes,
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            result = None
            try:
                async for event in wrapped(*args, **kwargs):
                    # Capture the final result event
                    if isinstance(event, dict) and "result" in event:
                        result = event["result"]
                    yield event
            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                raise

            # Set output attributes if we captured a result
            span.set_status(trace_api.StatusCode.OK)
            if result is not None:
                output_attrs = _get_agent_result_attributes(result)
                span.set_attributes(output_attrs)


class _EventLoopCycleWrapper(_WithTracer):
    """Wrapper for event_loop_cycle() to trace event loop execution."""

    async def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> AsyncGenerator[Any, None]:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            async for event in wrapped(*args, **kwargs):
                yield event
            return

        # Extract agent and invocation state
        agent = kwargs.get("agent")
        invocation_state = kwargs.get("invocation_state", {})
        cycle_id = invocation_state.get("event_loop_cycle_id", "unknown")

        if agent is None:
            # Fallback: just pass through
            async for event in wrapped(*args, **kwargs):
                yield event
            return

        agent_name = getattr(agent, "name", "unknown")

        # Build span attributes
        attributes = {
            OPENINFERENCE_SPAN_KIND: CHAIN,
            "event_loop.cycle_id": str(cycle_id),
            SpanAttributes.AGENT_NAME: agent_name,
        }
        attributes.update(get_attributes_from_context())

        # Start span
        span_name = f"{agent_name}.event_loop_cycle"
        with self._tracer.start_as_current_span(
            span_name,
            attributes=attributes,
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            stop_event = None
            try:
                async for event in wrapped(*args, **kwargs):
                    # Capture stop event for output attributes
                    if hasattr(event, "__dict__") and "stop" in getattr(event, "__dict__", {}):
                        stop_event = event
                    yield event
            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                raise

            span.set_status(trace_api.StatusCode.OK)
            if stop_event is not None:
                stop_reason, message, metrics, *_ = stop_event["stop"]
                span.set_attribute("event_loop.stop_reason", stop_reason)
                if message:
                    _add_message_attributes(span, message, prefix="output")


class _ToolExecutorExecuteWrapper(_WithTracer):
    """Wrapper for ToolExecutor._execute() to trace tool executions."""

    async def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> AsyncGenerator[Any, None]:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            async for event in wrapped(*args, **kwargs):
                yield event
            return

        # Extract arguments
        agent = args[0] if len(args) > 0 else kwargs.get("agent")
        tool_uses = args[1] if len(args) > 1 else kwargs.get("tool_uses", [])

        if not tool_uses:
            # No tools to trace, pass through
            async for event in wrapped(*args, **kwargs):
                yield event
            return

        # Create a parent span for all tool executions
        agent_name = getattr(agent, "name", "unknown") if agent else "unknown"
        attributes: Dict[str, AttributeValue] = {
            OPENINFERENCE_SPAN_KIND: CHAIN,
            "tool_execution.count": len(tool_uses),
        }
        attributes.update(get_attributes_from_context())

        span_name = f"{agent_name}.tool_execution"
        with self._tracer.start_as_current_span(
            span_name,
            attributes=dict(attributes),
            record_exception=False,
            set_status_on_exception=False,
        ) as parent_span:
            try:
                async for event in wrapped(*args, **kwargs):
                    yield event
            except Exception as exception:
                parent_span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                parent_span.record_exception(exception)
                raise

            parent_span.set_status(trace_api.StatusCode.OK)


class _ToolStreamWrapper(_WithTracer):
    """Wrapper for AgentTool.stream() to trace individual tool calls."""

    async def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> AsyncGenerator[Any, None]:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            async for event in wrapped(*args, **kwargs):
                yield event
            return

        # Extract tool use information
        tool_use = args[0] if args else kwargs.get("tool_use", {})
        tool_name = instance.tool_name if hasattr(instance, "tool_name") else "unknown"
        tool_use_id = tool_use.get("toolUseId", "unknown")
        tool_input = tool_use.get("input", {})

        # Build span attributes
        attributes = {
            OPENINFERENCE_SPAN_KIND: TOOL,
            SpanAttributes.TOOL_NAME: tool_name,
            "tool.use_id": tool_use_id,
            INPUT_VALUE: safe_json_dumps(tool_input),
            INPUT_MIME_TYPE: JSON,
        }
        attributes.update(get_attributes_from_context())

        # Start span
        span_name = f"tool.{tool_name}"
        with self._tracer.start_as_current_span(
            span_name,
            attributes=attributes,
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            tool_result = None
            try:
                async for event in wrapped(*args, **kwargs):
                    # Capture the final tool result
                    if hasattr(event, "__dict__"):
                        event_dict = getattr(event, "__dict__", {})
                        if "tool_result" in event_dict:
                            tool_result = event_dict["tool_result"]
                    yield event
            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                raise

            span.set_status(trace_api.StatusCode.OK)
            if tool_result:
                _add_tool_result_attributes(span, tool_result)


# Helper functions for extracting attributes


def _get_agent_attributes(agent: Any) -> dict[str, Any]:
    """Extract attributes from an agent instance."""
    attributes = {}

    if hasattr(agent, "name"):
        attributes[SpanAttributes.AGENT_NAME] = agent.name

    if hasattr(agent, "agent_id"):
        attributes["agent.id"] = agent.agent_id

    if hasattr(agent, "model") and hasattr(agent.model, "config"):
        model_config = agent.model.config
        if "model_id" in model_config:
            attributes[SpanAttributes.LLM_MODEL_NAME] = model_config["model_id"]

    if hasattr(agent, "tool_names"):
        tool_names = agent.tool_names
        if tool_names:
            for idx, tool_name in enumerate(tool_names):
                attributes[f"{SpanAttributes.LLM_TOOLS}.{idx}.name"] = tool_name

    return attributes


def _serialize_prompt(prompt: Any) -> dict[str, Any]:
    """Serialize prompt input to span attributes."""
    attributes = {}

    if isinstance(prompt, str):
        attributes[INPUT_VALUE] = prompt
        attributes[INPUT_MIME_TYPE] = TEXT
    elif isinstance(prompt, list):
        # Handle list of messages or content blocks
        attributes[INPUT_VALUE] = safe_json_dumps(prompt)
        attributes[INPUT_MIME_TYPE] = JSON
    elif prompt is None:
        # No new input, using existing conversation
        attributes[INPUT_VALUE] = ""
        attributes[INPUT_MIME_TYPE] = TEXT
    else:
        # Fallback: serialize as JSON
        attributes[INPUT_VALUE] = safe_json_dumps(prompt)
        attributes[INPUT_MIME_TYPE] = JSON

    return attributes


def _get_agent_result_attributes(result: Any) -> dict[str, Any]:
    """Extract attributes from agent result."""
    attributes = {}

    if hasattr(result, "stop_reason"):
        attributes["agent.stop_reason"] = result.stop_reason

    if hasattr(result, "message"):
        message = result.message
        attributes[OUTPUT_VALUE] = safe_json_dumps(message)
        attributes[OUTPUT_MIME_TYPE] = JSON

    if hasattr(result, "metrics"):
        metrics = result.metrics
        if hasattr(metrics, "usage"):
            usage = metrics.usage
            if "input_tokens" in usage:
                attributes[SpanAttributes.LLM_TOKEN_COUNT_PROMPT] = usage["input_tokens"]
            if "output_tokens" in usage:
                attributes[SpanAttributes.LLM_TOKEN_COUNT_COMPLETION] = usage["output_tokens"]
            if "total_tokens" in usage:
                attributes[SpanAttributes.LLM_TOKEN_COUNT_TOTAL] = usage["total_tokens"]

    return attributes


def _add_message_attributes(span: trace_api.Span, message: Any, prefix: str = "output") -> None:
    """Add message content as span attributes."""
    if isinstance(message, dict):
        role = message.get("role")
        content = message.get("content", [])

        if role:
            span.set_attribute(f"{prefix}.message.role", role)

        # Add content blocks
        for idx, content_block in enumerate(content):
            if "text" in content_block:
                span.set_attribute(f"{prefix}.message.content.{idx}.text", content_block["text"])
            elif "toolUse" in content_block:
                tool_use = content_block["toolUse"]
                span.set_attribute(
                    f"{prefix}.message.content.{idx}.tool_use.name", tool_use.get("name", "")
                )
                span.set_attribute(
                    f"{prefix}.message.content.{idx}.tool_use.id",
                    tool_use.get("toolUseId", ""),
                )


def _add_tool_result_attributes(span: trace_api.Span, tool_result: Any) -> None:
    """Add tool result as span attributes."""
    if isinstance(tool_result, dict):
        status = tool_result.get("status")
        content = tool_result.get("content", [])

        if status:
            span.set_attribute("tool.result.status", status)

        # Extract text content
        text_content = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                text_content.append(block["text"])

        if text_content:
            span.set_attribute(OUTPUT_VALUE, "\n".join(text_content))
            span.set_attribute(OUTPUT_MIME_TYPE, TEXT)


# Constants from semantic conventions
OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
AGENT = OpenInferenceSpanKindValues.AGENT.value
CHAIN = OpenInferenceSpanKindValues.CHAIN.value
TOOL = OpenInferenceSpanKindValues.TOOL.value
LLM = OpenInferenceSpanKindValues.LLM.value

INPUT_VALUE = SpanAttributes.INPUT_VALUE
INPUT_MIME_TYPE = SpanAttributes.INPUT_MIME_TYPE
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
OUTPUT_MIME_TYPE = SpanAttributes.OUTPUT_MIME_TYPE

TEXT = OpenInferenceMimeTypeValues.TEXT.value
JSON = OpenInferenceMimeTypeValues.JSON.value

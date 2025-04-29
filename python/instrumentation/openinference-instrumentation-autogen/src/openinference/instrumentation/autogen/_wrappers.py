from typing import Any, Dict, Tuple
from opentelemetry import trace as trace_api
from opentelemetry.trace import Span, Status, StatusCode
from openinference.instrumentation import OITracer

class _AssistantAgentWrapper:
    def __init__(self, tracer: OITracer) -> None:
        self._tracer = tracer

    def on_messages_wrapper(self, wrapped: Any, instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        with self._tracer.start_as_current_span("AssistantAgent.on_messages") as span:
            try:
                # Extract messages from args
                messages = args[0] if args else kwargs.get("messages")
                if messages:
                    span.set_attribute("messages.count", len(messages))
                    # Add other relevant attributes about messages
                
                result = wrapped(*args, **kwargs)
                return result
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

    def on_messages_stream_wrapper(self, wrapped: Any, instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        with self._tracer.start_as_current_span("AssistantAgent.on_messages_stream") as span:
            try:
                messages = args[0] if args else kwargs.get("messages")
                if messages:
                    span.set_attribute("messages.count", len(messages))
                
                result = wrapped(*args, **kwargs)
                return result
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

    def call_llm_wrapper(self, wrapped: Any, instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        with self._tracer.start_as_current_span("AssistantAgent.call_llm") as span:
            try:
                # Extract model client info
                model_client = args[0] if args else kwargs.get("model_client")
                if model_client:
                    span.set_attribute("model.name", getattr(model_client, "model", "unknown"))
                
                result = wrapped(*args, **kwargs)
                return result
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

    def execute_tool_wrapper(self, wrapped: Any, instance: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        with self._tracer.start_as_current_span("AssistantAgent.execute_tool") as span:
            try:
                # Extract tool call info
                tool_call = args[0] if args else kwargs.get("tool_call")
                if tool_call:
                    span.set_attribute("tool.name", getattr(tool_call, "name", "unknown"))
                    span.set_attribute("tool.arguments", str(getattr(tool_call, "arguments", "")))
                
                result = wrapped(*args, **kwargs)
                return result
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise
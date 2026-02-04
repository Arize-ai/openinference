"""Microsoft Agent Framework to OpenInference Span Processor.

This module provides a span processor that converts Microsoft Agent Framework's native
OpenTelemetry spans (using GenAI semantic conventions) to OpenInference format for
compatibility with OpenInference-compliant backends like Arize Phoenix.

The processor transforms:
- GenAI attributes (gen_ai.*) to OpenInference attributes (llm.*, tool.*, etc.)
- Span names to OpenInference span kinds (AGENT, CHAIN, TOOL, LLM)
- Message structures to OpenInference flattened format
- Token usage attributes to OpenInference format
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.trace import Status, StatusCode

from openinference.instrumentation.agent_framework import __version__
from openinference.instrumentation.agent_framework.semantic_conventions import (
    AGENT_DESCRIPTION,
    AGENT_ID,
    AGENT_INVOKE_OPERATION,
    AGENT_NAME,
    CHAT_COMPLETION_OPERATION,
    CONVERSATION_ID,
    EDGE_GROUP_ID,
    EDGE_GROUP_TYPE,
    EXECUTOR_ID,
    EXECUTOR_PROCESS_SPAN,
    EXECUTOR_TYPE,
    FINISH_REASONS,
    INPUT_MESSAGES,
    INPUT_TOKENS,
    LLM_REQUEST_MAX_TOKENS,
    LLM_REQUEST_MODEL,
    LLM_REQUEST_TEMPERATURE,
    LLM_REQUEST_TOP_P,
    LLM_RESPONSE_MODEL,
    OPERATION,
    OUTPUT_MESSAGES,
    OUTPUT_TOKENS,
    PROVIDER_NAME,
    RESPONSE_ID,
    SYSTEM_INSTRUCTIONS,
    TOOL_ARGUMENTS,
    TOOL_CALL_ID,
    TOOL_DEFINITIONS,
    TOOL_DESCRIPTION,
    TOOL_EXECUTION_OPERATION,
    TOOL_NAME,
    TOOL_RESULT,
    WORKFLOW_ID,
    WORKFLOW_NAME,
    WORKFLOW_RUN_SPAN,
    safe_json_dumps,
)

logger = logging.getLogger(__name__)


class AgentFrameworkToOpenInferenceProcessor(SpanProcessor):
    """
    SpanProcessor that converts Microsoft Agent Framework telemetry attributes
    to OpenInference format for compatibility with OpenInference-compliant backends.

    This processor intercepts spans on completion and transforms their attributes
    from the GenAI semantic conventions used by Microsoft Agent Framework to the
    OpenInference semantic conventions.

    Usage:
        ```python
        from opentelemetry.sdk.trace import TracerProvider
        from openinference.instrumentation.agent_framework import (
            AgentFrameworkToOpenInferenceProcessor
        )

        provider = TracerProvider()
        provider.add_span_processor(
            AgentFrameworkToOpenInferenceProcessor(debug=False)
        )
        ```
    """

    def __init__(self, debug: bool = False) -> None:
        """
        Initialize the processor.

        Args:
            debug: Whether to log debug information about transformations
        """
        super().__init__()
        self.debug = debug

    def on_start(self, span: ReadableSpan, parent_context: Any = None) -> None:
        """Called when a span is started. No-op for this processor."""
        pass

    def on_end(self, span: ReadableSpan) -> None:
        """
        Called when a span ends. Transform the span attributes from Microsoft Agent
        Framework GenAI format to OpenInference format.
        """
        if not hasattr(span, "_attributes") or not span._attributes:
            return

        original_attrs = dict(span._attributes)

        try:
            events: List[Any] = []
            if hasattr(span, "_events"):
                events = list(span._events)
            elif hasattr(span, "events"):
                events = list(span.events)

            transformed_attrs = self._transform_attributes(original_attrs, span, events)
            # Merge with original attributes to preserve GenAI attributes that weren't transformed
            span._attributes = {**span.attributes, **transformed_attrs}  # type: ignore[dict-item]

            # MS Agent Framework only sets ERROR status, not OK - set OK for successful spans
            if not span.status.status_code == StatusCode.ERROR:
                span._status = Status(status_code=StatusCode.OK)

            if self.debug:
                logger.info(
                    "span_name=<%s>, orig_attrs=<%d>, trans_attrs=<%d> | transformed span",
                    span.name,
                    len(original_attrs),
                    len(transformed_attrs),
                )
                logger.info("events=<%d> | processed events", len(events))

        except Exception as e:
            logger.error(f"Failed to transform span '{span.name}': {e}", exc_info=True)
            span._attributes = original_attrs

    def _transform_attributes(
        self, attrs: Dict[str, Any], span: ReadableSpan, events: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """
        Transform Microsoft Agent Framework attributes to OpenInference format.

        Args:
            attrs: Original span attributes
            span: The ReadableSpan being processed
            events: Span events (reserved for future use - MS Agent Framework
                   currently stores messages in attributes, not events)

        Returns:
            Transformed attributes dictionary
        """
        # Note: events parameter reserved for future use if MS Agent Framework
        # migrates to event-based message storage like other frameworks
        _ = events

        result: Dict[str, Any] = {}

        span_kind = self._determine_span_kind(span, attrs)
        result["openinference.span.kind"] = span_kind

        result.update(self._set_graph_node_attributes(span, attrs, span_kind))
        self._map_model_info(attrs, result)

        input_messages, output_messages = self._extract_messages(attrs)

        self._map_token_usage(attrs, result)

        if span_kind in ["LLM", "AGENT"]:
            self._handle_llm_agent_span(attrs, result, input_messages, output_messages)
        elif span_kind == "TOOL":
            self._handle_tool_span(attrs, result)
        elif span_kind == "CHAIN":
            self._handle_chain_span(attrs, result, input_messages, output_messages)
        self._map_session_info(attrs, result)
        self._map_invocation_parameters(attrs, result)
        self._add_metadata(attrs, result)

        return result

    def _determine_span_kind(self, span: ReadableSpan, attrs: Dict[str, Any]) -> str:
        """
        Determine the OpenInference span kind based on MS Agent Framework operation.

        Args:
            span: The span being processed
            attrs: Span attributes

        Returns:
            OpenInference span kind string (LLM, AGENT, TOOL, CHAIN)
        """
        operation = attrs.get(OPERATION, "")
        span_name = span.name

        if operation == CHAT_COMPLETION_OPERATION:
            return "LLM"
        elif operation == TOOL_EXECUTION_OPERATION:
            return "TOOL"
        elif operation == AGENT_INVOKE_OPERATION:
            return "AGENT"

        if span_name.startswith("chat "):
            return "LLM"
        elif span_name.startswith("execute_tool "):
            return "TOOL"
        elif span_name.startswith("invoke_agent "):
            return "AGENT"
        elif span_name.startswith(WORKFLOW_RUN_SPAN):
            return "CHAIN"
        elif span_name.startswith(EXECUTOR_PROCESS_SPAN):
            return "CHAIN"

        if attrs.get(AGENT_NAME) or attrs.get(AGENT_ID):
            return "AGENT"
        if attrs.get(WORKFLOW_ID) or attrs.get(WORKFLOW_NAME):
            return "CHAIN"
        if attrs.get(EXECUTOR_ID) or attrs.get(EXECUTOR_TYPE):
            return "CHAIN"

        return "CHAIN"

    def _set_graph_node_attributes(
        self, span: ReadableSpan, attrs: Dict[str, Any], span_kind: str
    ) -> Dict[str, Any]:
        """
        Set graph node attributes for visualization hierarchy.

        Args:
            span: The span being processed
            attrs: Span attributes
            span_kind: Determined OpenInference span kind

        Returns:
            Dictionary of graph node attributes
        """
        graph_attrs: Dict[str, Any] = {}
        span_id = span.get_span_context().span_id  # type: ignore[no-untyped-call]

        if span_kind == "AGENT":
            agent_id = attrs.get(AGENT_ID, span_id)
            graph_attrs["graph.node.id"] = f"agent_{agent_id}"
            agent_name = attrs.get(AGENT_NAME)
            if agent_name:
                graph_attrs["graph.node.name"] = agent_name

        elif span_kind == "LLM":
            graph_attrs["graph.node.id"] = f"llm_{span_id}"
            # Parent is agent if we're in an agent context
            if attrs.get(AGENT_ID):
                graph_attrs["graph.node.parent_id"] = f"agent_{attrs.get(AGENT_ID)}"

        elif span_kind == "TOOL":
            tool_name = attrs.get(TOOL_NAME, "unknown_tool")
            graph_attrs["graph.node.id"] = f"tool_{tool_name}_{span_id}"
            graph_attrs["graph.node.name"] = tool_name

        elif span_kind == "CHAIN":
            # Handle workflow and executor spans
            if workflow_id := attrs.get(WORKFLOW_ID):
                graph_attrs["graph.node.id"] = f"workflow_{workflow_id}"
                if workflow_name := attrs.get(WORKFLOW_NAME):
                    graph_attrs["graph.node.name"] = workflow_name
            elif executor_id := attrs.get(EXECUTOR_ID):
                graph_attrs["graph.node.id"] = f"executor_{executor_id}"
                if executor_type := attrs.get(EXECUTOR_TYPE):
                    graph_attrs["graph.node.name"] = executor_type
            elif edge_group_id := attrs.get(EDGE_GROUP_ID):
                graph_attrs["graph.node.id"] = f"edge_group_{edge_group_id}"
                if edge_group_type := attrs.get(EDGE_GROUP_TYPE):
                    graph_attrs["graph.node.name"] = edge_group_type
            else:
                graph_attrs["graph.node.id"] = f"chain_{span_id}"

        return graph_attrs

    def _map_model_info(self, attrs: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Map model and provider information to OpenInference format."""
        model_name = attrs.get(LLM_REQUEST_MODEL) or attrs.get(LLM_RESPONSE_MODEL)
        if model_name:
            result["llm.model_name"] = model_name

        provider = attrs.get(PROVIDER_NAME)
        if provider:
            result["llm.provider"] = provider

    def _extract_messages(
        self, attrs: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Extract input and output messages from attributes.

        MS Agent Framework stores messages as JSON strings in gen_ai.input.messages
        and gen_ai.output.messages attributes.

        Returns:
            Tuple of (input_messages, output_messages)
        """
        input_messages: List[Dict[str, Any]] = []
        output_messages: List[Dict[str, Any]] = []

        if input_msgs_raw := attrs.get(INPUT_MESSAGES):
            input_messages = self._parse_messages(input_msgs_raw)
        if output_msgs_raw := attrs.get(OUTPUT_MESSAGES):
            output_messages = self._parse_messages(output_msgs_raw)

        return input_messages, output_messages

    def _parse_messages(self, messages_raw: Any) -> List[Dict[str, Any]]:
        """
        Parse messages from MS Agent Framework format.

        MS Agent Framework uses a format like:
        [{"role": "user", "parts": [{"type": "text", "content": "..."}]}]

        Args:
            messages_raw: Raw messages (JSON string or list)

        Returns:
            List of parsed message dictionaries in OpenInference format
        """
        messages: List[Dict[str, Any]] = []

        try:
            if isinstance(messages_raw, str):
                messages_data = json.loads(messages_raw)
            else:
                messages_data = messages_raw

            if not isinstance(messages_data, list):
                return messages

            for msg in messages_data:
                if not isinstance(msg, dict):
                    continue

                parsed_msg = self._parse_single_message(msg)
                if parsed_msg:
                    messages.append(parsed_msg)

        except (json.JSONDecodeError, TypeError) as e:
            logger.debug(f"Failed to parse messages: {e}")

        return messages

    def _parse_single_message(self, msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Parse a single message from MS Agent Framework format to OpenInference format.

        Args:
            msg: Message dictionary with 'role' and 'parts' keys

        Returns:
            Parsed message in OpenInference format or None
        """
        role = msg.get("role", "user")
        parts = msg.get("parts") or []

        result: Dict[str, Any] = {"message.role": role}

        text_content: List[str] = []
        tool_calls: List[Dict[str, Any]] = []

        for part in parts:
            if not isinstance(part, dict):
                continue

            part_type = part.get("type", "")

            if part_type == "text":
                if content := part.get("content"):
                    text_content.append(str(content))

            elif part_type == "reasoning":
                if content := part.get("content"):
                    text_content.append(str(content))

            elif part_type == "tool_call":
                tool_call = {
                    "tool_call.id": part.get("id", ""),
                    "tool_call.function.name": part.get("name", ""),
                    "tool_call.function.arguments": safe_json_dumps(part.get("arguments", {})),
                }
                tool_calls.append(tool_call)

            elif part_type == "tool_call_response":
                result["message.role"] = "tool"
                result["message.tool_call_id"] = part.get("id", "")
                response = part.get("response", {})
                if isinstance(response, dict):
                    text_content.append(safe_json_dumps(response))
                else:
                    text_content.append(str(response))

        if text_content:
            result["message.content"] = " ".join(text_content)
        if tool_calls:
            result["message.tool_calls"] = tool_calls

        if "message.content" not in result and "message.tool_calls" not in result:
            return None

        return result

    def _handle_llm_agent_span(
        self,
        attrs: Dict[str, Any],
        result: Dict[str, Any],
        input_messages: List[Dict[str, Any]],
        output_messages: List[Dict[str, Any]],
    ) -> None:
        """Handle LLM and AGENT span attributes."""
        if input_messages:
            result["llm.input_messages"] = safe_json_dumps(input_messages)
            self._flatten_messages(input_messages, "llm.input_messages", result)

        if output_messages:
            result["llm.output_messages"] = safe_json_dumps(output_messages)
            self._flatten_messages(output_messages, "llm.output_messages", result)

        self._create_input_output_values(attrs, result, input_messages, output_messages)

        if attrs.get(SYSTEM_INSTRUCTIONS) or attrs.get(AGENT_NAME):
            result["llm.system"] = "microsoft.agent_framework"

        if tool_defs := attrs.get(TOOL_DEFINITIONS):
            self._map_tools(tool_defs, result)

    def _handle_tool_span(self, attrs: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Handle TOOL span attributes."""
        if tool_name := attrs.get(TOOL_NAME):
            result["tool.name"] = tool_name
        if tool_call_id := attrs.get(TOOL_CALL_ID):
            result["tool.call_id"] = tool_call_id
        if tool_desc := attrs.get(TOOL_DESCRIPTION):
            result["tool.description"] = tool_desc

        if tool_args := attrs.get(TOOL_ARGUMENTS):
            if isinstance(tool_args, str):
                result["tool.parameters"] = tool_args
                result["input.value"] = tool_args
            else:
                result["tool.parameters"] = safe_json_dumps(tool_args)
                result["input.value"] = safe_json_dumps(tool_args)
            result["input.mime_type"] = "application/json"

        if tool_result := attrs.get(TOOL_RESULT):
            if isinstance(tool_result, str):
                result["output.value"] = tool_result
            else:
                result["output.value"] = safe_json_dumps(tool_result)
            result["output.mime_type"] = "text/plain"

    def _handle_chain_span(
        self,
        attrs: Dict[str, Any],
        result: Dict[str, Any],
        input_messages: List[Dict[str, Any]],
        output_messages: List[Dict[str, Any]],
    ) -> None:
        """Handle CHAIN span attributes (workflows, executors)."""
        if input_messages:
            for msg in input_messages:
                if msg.get("message.role") == "user":
                    if content := msg.get("message.content"):
                        result["input.value"] = content
                        result["input.mime_type"] = "text/plain"
                        break

        if output_messages:
            for msg in reversed(output_messages):
                if msg.get("message.role") == "assistant":
                    if content := msg.get("message.content"):
                        result["output.value"] = content
                        result["output.mime_type"] = "text/plain"
                        break

    def _flatten_messages(
        self, messages: List[Dict[str, Any]], key_prefix: str, result: Dict[str, Any]
    ) -> None:
        """Flatten messages to dotted attribute notation for OpenInference."""
        for idx, msg in enumerate(messages):
            for key, value in msg.items():
                clean_key = key.replace("message.", "") if key.startswith("message.") else key
                dotted_key = f"{key_prefix}.{idx}.message.{clean_key}"

                if clean_key == "tool_calls" and isinstance(value, list):
                    for tool_idx, tool_call in enumerate(value):
                        if isinstance(tool_call, dict):
                            for tool_key, tool_val in tool_call.items():
                                tool_dotted_key = (
                                    f"{key_prefix}.{idx}.message.tool_calls.{tool_idx}.{tool_key}"
                                )
                                result[tool_dotted_key] = self._serialize_value(tool_val)
                else:
                    result[dotted_key] = self._serialize_value(value)

    def _create_input_output_values(
        self,
        attrs: Dict[str, Any],
        result: Dict[str, Any],
        input_messages: List[Dict[str, Any]],
        output_messages: List[Dict[str, Any]],
    ) -> None:
        """Create input.value and output.value attributes."""
        span_kind = result.get("openinference.span.kind")
        model_name = result.get("llm.model_name", attrs.get(LLM_REQUEST_MODEL, "unknown"))

        if span_kind in ["LLM", "AGENT"]:
            if input_messages:
                if len(input_messages) == 1 and input_messages[0].get("message.role") == "user":
                    result["input.value"] = input_messages[0].get("message.content", "")
                    result["input.mime_type"] = "text/plain"
                else:
                    input_structure = {"messages": input_messages, "model": model_name}
                    result["input.value"] = safe_json_dumps(input_structure)
                    result["input.mime_type"] = "application/json"

            if output_messages:
                last_message = output_messages[-1]
                content = last_message.get("message.content", "")

                if span_kind == "LLM":
                    finish_reasons = attrs.get(FINISH_REASONS)
                    finish_reason = "stop"
                    if finish_reasons:
                        try:
                            reasons = (
                                json.loads(finish_reasons)
                                if isinstance(finish_reasons, str)
                                else finish_reasons
                            )
                            if isinstance(reasons, list) and reasons:
                                finish_reason = reasons[0]
                        except (json.JSONDecodeError, TypeError):
                            pass

                    output_structure = {
                        "choices": [
                            {
                                "finish_reason": finish_reason,
                                "index": 0,
                                "message": {
                                    "content": content,
                                    "role": last_message.get("message.role", "assistant"),
                                },
                            }
                        ],
                        "model": model_name,
                        "usage": {
                            "completion_tokens": result.get("llm.token_count.completion"),
                            "prompt_tokens": result.get("llm.token_count.prompt"),
                            "total_tokens": result.get("llm.token_count.total"),
                        },
                    }
                    result["output.value"] = safe_json_dumps(output_structure)
                    result["output.mime_type"] = "application/json"
                else:
                    result["output.value"] = content
                    result["output.mime_type"] = "text/plain"

    def _map_token_usage(self, attrs: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Map token usage from GenAI format to OpenInference format."""
        input_tokens = attrs.get(INPUT_TOKENS)
        if input_tokens is not None:
            result["llm.token_count.prompt"] = input_tokens
        output_tokens = attrs.get(OUTPUT_TOKENS)
        if output_tokens is not None:
            result["llm.token_count.completion"] = output_tokens
        if "llm.token_count.prompt" in result and "llm.token_count.completion" in result:
            result["llm.token_count.total"] = (
                result["llm.token_count.prompt"] + result["llm.token_count.completion"]
            )

    def _map_session_info(self, attrs: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Map session and conversation info."""
        if conversation_id := attrs.get(CONVERSATION_ID):
            result["session.id"] = conversation_id

    def _map_invocation_parameters(self, attrs: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Map invocation parameters to OpenInference format."""
        params: Dict[str, Any] = {}

        param_mappings = {
            LLM_REQUEST_MAX_TOKENS: "max_tokens",
            LLM_REQUEST_TEMPERATURE: "temperature",
            LLM_REQUEST_TOP_P: "top_p",
        }

        for attr_key, param_key in param_mappings.items():
            if attr_key in attrs:
                params[param_key] = attrs[attr_key]

        if params:
            result["llm.invocation_parameters"] = safe_json_dumps(params)

    def _map_tools(self, tools_data: Any, result: Dict[str, Any]) -> None:
        """Map tool definitions to OpenInference format."""
        try:
            if isinstance(tools_data, str):
                tools_data = json.loads(tools_data)

            if not isinstance(tools_data, list):
                return

            for idx, tool in enumerate(tools_data):
                if isinstance(tool, dict):
                    if name := tool.get("name"):
                        result[f"llm.tools.{idx}.tool.name"] = name
                    if desc := tool.get("description"):
                        result[f"llm.tools.{idx}.tool.description"] = desc
                    # Handle function schema
                    if "function" in tool:
                        func = tool["function"]
                        if isinstance(func, dict):
                            if fname := func.get("name"):
                                result[f"llm.tools.{idx}.tool.name"] = fname
                            if fdesc := func.get("description"):
                                result[f"llm.tools.{idx}.tool.description"] = fdesc
                            if params := func.get("parameters"):
                                result[f"llm.tools.{idx}.tool.json_schema"] = safe_json_dumps(
                                    params
                                )
                    elif params := tool.get("parameters"):
                        result[f"llm.tools.{idx}.tool.json_schema"] = safe_json_dumps(params)
                    elif input_schema := tool.get("input_schema"):
                        result[f"llm.tools.{idx}.tool.json_schema"] = safe_json_dumps(input_schema)

        except (json.JSONDecodeError, TypeError) as e:
            logger.debug(f"Failed to parse tools: {e}")

    def _add_metadata(self, attrs: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Add remaining attributes as metadata."""
        metadata: Dict[str, Any] = {}
        skip_keys = {
            OPERATION,
            PROVIDER_NAME,
            LLM_REQUEST_MODEL,
            LLM_RESPONSE_MODEL,
            INPUT_TOKENS,
            OUTPUT_TOKENS,
            INPUT_MESSAGES,
            OUTPUT_MESSAGES,
            TOOL_NAME,
            TOOL_CALL_ID,
            TOOL_ARGUMENTS,
            TOOL_RESULT,
            TOOL_DESCRIPTION,
            TOOL_DEFINITIONS,
            AGENT_ID,
            AGENT_NAME,
            AGENT_DESCRIPTION,
            CONVERSATION_ID,
            FINISH_REASONS,
            RESPONSE_ID,
            SYSTEM_INSTRUCTIONS,
            LLM_REQUEST_MAX_TOKENS,
            LLM_REQUEST_TEMPERATURE,
            LLM_REQUEST_TOP_P,
            WORKFLOW_ID,
            WORKFLOW_NAME,
            EXECUTOR_ID,
            EXECUTOR_TYPE,
            EDGE_GROUP_ID,
            EDGE_GROUP_TYPE,
        }

        for key, value in attrs.items():
            if key not in skip_keys and key not in result:
                metadata[key] = self._serialize_value(value)

        if metadata:
            result["metadata"] = safe_json_dumps(metadata)

    def _serialize_value(self, value: Any) -> Any:
        """Serialize a value for span attributes."""
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return safe_json_dumps(value)

    def shutdown(self) -> None:
        """Shutdown the processor."""
        pass

    def force_flush(self, timeout_millis: Optional[int] = None) -> bool:
        """Force flush any pending data."""
        return True

    def get_processor_info(self) -> Dict[str, Any]:
        """Get information about this processor's capabilities."""
        return {
            "processor_name": "AgentFrameworkToOpenInferenceProcessor",
            "version": __version__,
            "debug_enabled": self.debug,
            "supported_span_kinds": ["LLM", "AGENT", "CHAIN", "TOOL"],
            "supported_operations": [
                "chat",
                "execute_tool",
                "invoke_agent",
                "create_agent",
                "workflow.run",
                "executor.process",
            ],
            "features": [
                "Message extraction and transformation",
                "Token usage mapping",
                "Tool call processing",
                "Graph node hierarchy mapping",
                "Workflow/executor span support",
                "Invocation parameters mapping",
            ],
        }

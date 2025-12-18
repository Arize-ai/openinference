"""Strands to OpenInference Span Processor.

This module provides a span processor that converts Strands' native OpenTelemetry spans
(using GenAI semantic conventions) to OpenInference format for compatibility with
OpenInference-compliant backends.

The processor transforms:
- GenAI attributes (gen_ai.*) to OpenInference attributes (llm.*, tool.*, agent.*)
- Span names to OpenInference span kinds (AGENT, CHAIN, TOOL, LLM)
- GenAI events to OpenInference message structures
- Token usage attributes to OpenInference format
"""

import json
import logging
from typing import Any, Dict, List, Optional

from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.sdk.trace.export import SpanExportResult

from openinference.instrumentation import (
    GEN_AI_REQUEST_MODEL,
    GEN_AI_TOOL_NAME,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    GenAIAttributes,
    GenAIEventNames,
    safe_json_dumps,
)

logger = logging.getLogger(__name__)


class StrandsToOpenInferenceProcessor(SpanProcessor):
    """
    SpanProcessor that converts Strands telemetry attributes to OpenInference format
    for compatibility with OpenInference-compliant backends.
    """

    def __init__(self, debug: bool = False) -> None:
        """
        Initialize the processor.

        Args:
            debug: Whether to log debug information
        """
        super().__init__()
        self.debug = debug

    def on_start(self, span: ReadableSpan, parent_context: Any = None) -> None:
        """Called when a span is started."""
        pass

    def on_end(self, span: ReadableSpan) -> None:
        """
        Called when a span ends. Transform the span attributes from Strands format
        to OpenInference format.
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
            span._attributes = transformed_attrs

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
        Transform Strands attributes to OpenInference format, including event processing.
        """
        result: Dict[str, Any] = {}
        span_kind = self._determine_span_kind(span, attrs)
        result["openinference.span.kind"] = span_kind
        result.update(self._set_graph_node_attributes(span, attrs, span_kind))

        # Extract messages from events if available, otherwise fall back to attributes
        if events and len(events) > 0:
            input_messages, output_messages = self._extract_messages_from_events(events)
        else:
            prompt = attrs.get(GenAIAttributes.PROMPT)
            completion = attrs.get(GenAIAttributes.COMPLETION)
            if prompt or completion:
                input_messages, output_messages = self._extract_messages_from_attributes(
                    prompt, completion
                )
            else:
                input_messages, output_messages = [], []

        model_id = attrs.get(GEN_AI_REQUEST_MODEL)
        agent_name = attrs.get("agent.name") or attrs.get(GenAIAttributes.AGENT_NAME)

        if model_id:
            result["llm.model_name"] = model_id
            result[GEN_AI_REQUEST_MODEL] = model_id

        if agent_name:
            result["llm.system"] = "strands-agents"
            result["llm.provider"] = "strands-agents"

        self._handle_tags(attrs, result)

        if span_kind in ["LLM", "AGENT", "CHAIN"]:
            self._handle_llm_span(attrs, result, input_messages, output_messages)
        elif span_kind == "TOOL":
            self._handle_tool_span(attrs, result, events)

        self._map_token_usage(attrs, result)

        passthrough_attrs = [
            "session.id",
            "user.id",
            "llm.prompt_template.template",
            "llm.prompt_template.version",
            "llm.prompt_template.variables",
            "gen_ai.event.start_time",
            "gen_ai.event.end_time",
        ]

        for key in passthrough_attrs:
            if key in attrs:
                result[key] = attrs[key]

        self._add_metadata(attrs, result)
        return result

    def _extract_messages_from_events(
        self, events: List[Any]
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Extract input and output messages from Strands events with updated format handling."""
        input_messages = []
        output_messages = []

        for event in events:
            event_name = (
                getattr(event, "name", "") if hasattr(event, "name") else event.get("name", "")
            )
            event_attrs = (
                getattr(event, "attributes", {})
                if hasattr(event, "attributes")
                else event.get("attributes", {})
            )

            if event_name == GenAIEventNames.USER_MESSAGE:
                content = event_attrs.get("content", "")
                message = self._parse_message_content(content, "user")
                if message:
                    input_messages.append(message)

            elif event_name == GenAIEventNames.ASSISTANT_MESSAGE:
                content = event_attrs.get("content", "")
                message = self._parse_message_content(content, "assistant")
                if message:
                    output_messages.append(message)

            elif event_name == GenAIEventNames.CHOICE:
                message_content = event_attrs.get("message", "")
                if message_content:
                    message = self._parse_message_content(message_content, "assistant")
                    if message:
                        if "finish_reason" in event_attrs:
                            message["message.finish_reason"] = event_attrs["finish_reason"]
                        output_messages.append(message)

            elif event_name == GenAIEventNames.TOOL_MESSAGE:
                content = event_attrs.get("content", "")
                tool_id = event_attrs.get("id", "")
                if content:
                    message = self._parse_message_content(content, "tool")
                    if message and tool_id:
                        message["message.tool_call_id"] = tool_id
                        input_messages.append(message)

        return input_messages, output_messages

    def _extract_messages_from_attributes(
        self, prompt: Any, completion: Any
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Fallback method to extract messages from attributes."""
        input_messages = []
        output_messages = []

        if prompt:
            if isinstance(prompt, str):
                try:
                    prompt_data = json.loads(prompt)
                    if isinstance(prompt_data, list):
                        for msg in prompt_data:
                            normalized = self._normalize_message(msg)
                            if normalized.get("message.role") == "user":
                                input_messages.append(normalized)
                    elif isinstance(prompt_data, dict):
                        # Handle single dict prompt (e.g., {"role": "user", "content": "Hello"})
                        normalized = self._normalize_message(prompt_data)
                        input_messages.append(normalized)
                    else:
                        # Handle other JSON types (string, number, etc.)
                        input_messages.append(
                            {"message.role": "user", "message.content": str(prompt_data)}
                        )
                except json.JSONDecodeError:
                    input_messages.append({"message.role": "user", "message.content": str(prompt)})

        if completion:
            if isinstance(completion, str):
                try:
                    completion_data = json.loads(completion)
                    if isinstance(completion_data, list):
                        message = self._parse_strands_completion(completion_data)
                        if message:
                            output_messages.append(message)
                    elif isinstance(completion_data, dict):
                        # Handle dict completions (e.g., {"text": "hello"})
                        if "text" in completion_data:
                            output_messages.append(
                                {
                                    "message.role": "assistant",
                                    "message.content": str(completion_data["text"]),
                                }
                            )
                        else:
                            output_messages.append(
                                {
                                    "message.role": "assistant",
                                    "message.content": safe_json_dumps(completion_data),
                                }
                            )
                    else:
                        # Handle other JSON types (string, number, etc.)
                        output_messages.append(
                            {"message.role": "assistant", "message.content": str(completion_data)}
                        )
                except json.JSONDecodeError:
                    output_messages.append(
                        {"message.role": "assistant", "message.content": str(completion)}
                    )

        return input_messages, output_messages

    def _parse_message_content(self, content: str, role: str) -> Optional[Dict[str, Any]]:
        """Parse message content from Strands event format with enhanced JSON parsing."""
        if not content:
            return None

        try:
            content_data = json.loads(content) if isinstance(content, str) else content

            if isinstance(content_data, list):
                message: Dict[str, Any] = {"message.role": role}

                text_parts: List[str] = []
                tool_calls: List[Dict[str, Any]] = []
                for item in content_data:
                    if isinstance(item, dict):
                        if "text" in item:
                            text_parts.append(str(item["text"]))
                        elif "toolUse" in item:
                            tool_use = item["toolUse"]
                            tool_call = {
                                "tool_call.id": tool_use.get("toolUseId", ""),
                                "tool_call.function.name": tool_use.get("name", ""),
                                "tool_call.function.arguments": safe_json_dumps(
                                    tool_use.get("input", {})
                                ),
                            }
                            tool_calls.append(tool_call)
                        elif "toolResult" in item:
                            tool_result = item["toolResult"]
                            if "content" in tool_result:
                                if isinstance(tool_result["content"], list):
                                    for tr_content in tool_result["content"]:
                                        if isinstance(tr_content, dict) and "text" in tr_content:
                                            text_parts.append(str(tr_content["text"]))
                                elif isinstance(tool_result["content"], str):
                                    text_parts.append(tool_result["content"])
                            message["message.role"] = "tool"
                            if "toolUseId" in tool_result:
                                message["message.tool_call_id"] = tool_result["toolUseId"]

                if text_parts:
                    message["message.content"] = " ".join(text_parts)

                if tool_calls:
                    message["message.tool_calls"] = tool_calls

                return message
            elif isinstance(content_data, dict):
                if "text" in content_data:
                    return {"message.role": role, "message.content": str(content_data["text"])}
                else:
                    return {"message.role": role, "message.content": str(content_data)}
            else:
                return {"message.role": role, "message.content": str(content_data)}

        except (json.JSONDecodeError, TypeError):
            return {"message.role": role, "message.content": str(content)}

    def _parse_strands_completion(self, completion_data: List[Any]) -> Optional[Dict[str, Any]]:
        message: Dict[str, Any] = {"message.role": "assistant"}

        text_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        for item in completion_data:
            if isinstance(item, dict):
                if "text" in item:
                    text_parts.append(str(item["text"]))
                elif "toolUse" in item:
                    tool_use = item["toolUse"]
                    tool_call = {
                        "tool_call.id": tool_use.get("toolUseId", ""),
                        "tool_call.function.name": tool_use.get("name", ""),
                        "tool_call.function.arguments": safe_json_dumps(tool_use.get("input", {})),
                    }
                    tool_calls.append(tool_call)

        if text_parts:
            message["message.content"] = " ".join(text_parts)

        if tool_calls:
            message["message.tool_calls"] = tool_calls

        if "message.content" not in message and "message.tool_calls" not in message:
            return None

        return message

    def _handle_llm_span(
        self,
        attrs: Dict[str, Any],
        result: Dict[str, Any],
        input_messages: List[Dict[str, Any]],
        output_messages: List[Dict[str, Any]],
    ) -> None:
        """Handle LLM/Agent span with extracted messages."""

        if input_messages:
            result["llm.input_messages"] = safe_json_dumps(input_messages)
            self._flatten_messages(input_messages, "llm.input_messages", result)

        if output_messages:
            result["llm.output_messages"] = safe_json_dumps(output_messages)
            self._flatten_messages(output_messages, "llm.output_messages", result)

        if tools := (attrs.get(GenAIAttributes.AGENT_TOOLS) or attrs.get("agent.tools")):
            self._map_tools(tools, result)

        self._create_input_output_values(attrs, result, input_messages, output_messages)

        self._map_invocation_parameters(attrs, result)

    def _flatten_messages(
        self, messages: List[Dict[str, Any]], key_prefix: str, result: Dict[str, Any]
    ) -> None:
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
        span_kind = result.get("openinference.span.kind")
        model_name = result.get("llm.model_name") or attrs.get(GEN_AI_REQUEST_MODEL) or "unknown"

        if span_kind in ["LLM", "AGENT", "CHAIN"]:
            if input_messages:
                if len(input_messages) == 1 and input_messages[0].get("message.role") == "user":
                    # Simple user message
                    result["input.value"] = input_messages[0].get("message.content", "")
                    result["input.mime_type"] = "text/plain"
                else:
                    # Complex conversation
                    input_structure = {"messages": input_messages, "model": model_name}
                    result["input.value"] = safe_json_dumps(input_structure)
                    result["input.mime_type"] = "application/json"

            if output_messages:
                last_message = output_messages[-1]
                content = last_message.get("message.content", "")

                if span_kind == "LLM":
                    output_structure = {
                        "choices": [
                            {
                                "finish_reason": last_message.get("message.finish_reason", "stop"),
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

    def _handle_tags(self, attrs: Dict[str, Any], result: Dict[str, Any]) -> None:
        tags = None

        if "arize.tags" in attrs:
            tags = attrs["arize.tags"]
        elif "tag.tags" in attrs:
            tags = attrs["tag.tags"]

        if tags:
            if isinstance(tags, list):
                result["tag.tags"] = tags
            elif isinstance(tags, str):
                result["tag.tags"] = [tags]

    def _determine_span_kind(self, span: ReadableSpan, attrs: Dict[str, Any]) -> str:
        span_name = span.name

        if span_name == "chat":
            return "LLM"
        elif span_name.startswith("execute_tool "):
            return "TOOL"
        elif span_name == "execute_event_loop_cycle":
            return "CHAIN"
        elif span_name.startswith("invoke_agent"):
            return "AGENT"
        elif "Model invoke" in span_name:
            return "LLM"
        elif span_name.startswith("Tool:"):
            return "TOOL"
        elif "Cycle" in span_name:
            return "CHAIN"
        elif attrs.get(GenAIAttributes.AGENT_NAME) or attrs.get("agent.name"):
            return "AGENT"

        return "CHAIN"

    def _set_graph_node_attributes(
        self, span: ReadableSpan, attrs: Dict[str, Any], span_kind: str
    ) -> Dict[str, Any]:
        """
        Set graph node attributes for visualization.

        Returns a dict of graph node attributes to be merged into the result.
        Parent IDs are only set when reliably determinable without state tracking.
        """
        graph_attrs: Dict[str, Any] = {}
        span_name = span.name
        span_id = span.get_span_context().span_id  # type: ignore[no-untyped-call]

        if span_kind == "AGENT":
            graph_attrs["graph.node.id"] = "strands_agent"
        elif span_kind == "CHAIN":
            # execute_event_loop_cycle: Strands' agentic loop iteration where the LLM
            # reasons, plans, and optionally selects tools. Each cycle is a child of
            # the agent span.
            if span_name == "execute_event_loop_cycle":
                cycle_id = attrs.get("event_loop.cycle_id", f"cycle_{span_id}")
                graph_attrs["graph.node.id"] = f"cycle_{cycle_id}"
                graph_attrs["graph.node.parent_id"] = "strands_agent"
            elif "Cycle " in span_name:  # Legacy support
                cycle_id = span_name.replace("Cycle ", "").strip()
                graph_attrs["graph.node.id"] = f"cycle_{cycle_id}"
                graph_attrs["graph.node.parent_id"] = "strands_agent"
        elif span_kind == "LLM":
            graph_attrs["graph.node.id"] = f"llm_{span_id}"
        elif span_kind == "TOOL":
            tool_name = (
                span_name.replace("execute_tool ", "")
                if span_name.startswith("execute_tool ")
                else "unknown_tool"
            )
            graph_attrs["graph.node.id"] = f"tool_{tool_name}_{span_id}"

        return graph_attrs

    def _handle_tool_span(
        self, attrs: Dict[str, Any], result: Dict[str, Any], events: Optional[List[Any]] = None
    ) -> None:
        """Handle tool-specific attributes with enhanced event processing."""
        tool_name = attrs.get(GEN_AI_TOOL_NAME)
        tool_call_id = attrs.get(GenAIAttributes.TOOL_CALL_ID)
        tool_status = attrs.get("tool.status")

        if tool_name:
            result["tool.name"] = tool_name

        if tool_call_id:
            result["tool.call_id"] = tool_call_id

        if tool_status:
            result["tool.status"] = tool_status

        if events:
            tool_parameters = None
            tool_output = None

            for event in events:
                event_name = (
                    getattr(event, "name", "") if hasattr(event, "name") else event.get("name", "")
                )
                event_attrs = (
                    getattr(event, "attributes", {})
                    if hasattr(event, "attributes")
                    else event.get("attributes", {})
                )

                if event_name == GenAIEventNames.TOOL_MESSAGE:
                    content = event_attrs.get("content", "")
                    if content:
                        try:
                            content_data = (
                                json.loads(content) if isinstance(content, str) else content
                            )
                            if isinstance(content_data, dict):
                                tool_parameters = content_data
                            else:
                                tool_parameters = {"input": str(content_data)}
                        except (json.JSONDecodeError, TypeError):
                            tool_parameters = {"input": str(content)}

                elif event_name == GenAIEventNames.CHOICE:
                    message = event_attrs.get("message", "")
                    if message:
                        try:
                            message_data = (
                                json.loads(message) if isinstance(message, str) else message
                            )
                            if isinstance(message_data, list):
                                text_parts = []
                                for item in message_data:
                                    if isinstance(item, dict) and "text" in item:
                                        text_parts.append(item["text"])
                                tool_output = (
                                    " ".join(text_parts) if text_parts else str(message_data)
                                )
                            else:
                                tool_output = str(message_data)
                        except (json.JSONDecodeError, TypeError):
                            tool_output = str(message)

            if tool_parameters:
                result["tool.parameters"] = safe_json_dumps(tool_parameters)

                if tool_name and tool_call_id:
                    input_messages = [
                        {
                            "message.role": "assistant",
                            "message.content": "",
                            "message.tool_calls": [
                                {
                                    "tool_call.id": tool_call_id,
                                    "tool_call.function.name": tool_name,
                                    "tool_call.function.arguments": safe_json_dumps(
                                        tool_parameters
                                    ),
                                }
                            ],
                        }
                    ]

                    result["llm.input_messages"] = safe_json_dumps(input_messages)
                    self._flatten_messages(input_messages, "llm.input_messages", result)

                if isinstance(tool_parameters, dict):
                    if "text" in tool_parameters:
                        result["input.value"] = tool_parameters["text"]
                        result["input.mime_type"] = "text/plain"
                    else:
                        result["input.value"] = safe_json_dumps(tool_parameters)
                        result["input.mime_type"] = "application/json"

            if tool_output:
                result["output.value"] = tool_output
                result["output.mime_type"] = "text/plain"

    def _map_tools(self, tools_data: Any, result: Dict[str, Any]) -> None:
        """Map tools from Strands to OpenInference format."""
        if isinstance(tools_data, str):
            try:
                tools_data = json.loads(tools_data)
            except json.JSONDecodeError:
                return

        if not isinstance(tools_data, list):
            return

        for idx, tool in enumerate(tools_data):
            if isinstance(tool, str):
                result[f"llm.tools.{idx}.tool.name"] = tool
                result[f"llm.tools.{idx}.tool.description"] = f"Tool: {tool}"
            elif isinstance(tool, dict):
                result[f"llm.tools.{idx}.tool.name"] = tool.get("name", "")
                result[f"llm.tools.{idx}.tool.description"] = tool.get("description", "")
                if "parameters" in tool or "input_schema" in tool:
                    schema = tool.get("parameters") or tool.get("input_schema")
                    result[f"llm.tools.{idx}.tool.json_schema"] = safe_json_dumps(schema)

    def _map_token_usage(self, attrs: Dict[str, Any], result: Dict[str, Any]) -> None:
        token_mappings = [
            (GenAIAttributes.USAGE_PROMPT_TOKENS, "llm.token_count.prompt"),
            (GEN_AI_USAGE_INPUT_TOKENS, "llm.token_count.prompt"),
            (GenAIAttributes.USAGE_COMPLETION_TOKENS, "llm.token_count.completion"),
            (GEN_AI_USAGE_OUTPUT_TOKENS, "llm.token_count.completion"),
            (GenAIAttributes.USAGE_TOTAL_TOKENS, "llm.token_count.total"),
        ]

        for strands_key, openinf_key in token_mappings:
            value = attrs.get(strands_key)
            if value is not None:
                result[openinf_key] = value

    def _map_invocation_parameters(self, attrs: Dict[str, Any], result: Dict[str, Any]) -> None:
        params = {}
        param_mappings = {
            "max_tokens": "max_tokens",
            "temperature": "temperature",
            "top_p": "top_p",
        }

        for key, param_key in param_mappings.items():
            if key in attrs:
                params[param_key] = attrs[key]

        if params:
            result["llm.invocation_parameters"] = safe_json_dumps(params)

    def _normalize_message(self, msg: Any) -> Dict[str, Any]:
        if not isinstance(msg, dict):
            return {"message.role": "user", "message.content": str(msg)}

        result = {}
        if "role" in msg:
            result["message.role"] = msg["role"]

        if "content" in msg:
            content = msg["content"]
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        text_parts.append(str(item["text"]))
                result["message.content"] = " ".join(text_parts) if text_parts else ""
            else:
                result["message.content"] = str(content)

        return result

    def _add_metadata(self, attrs: Dict[str, Any], result: Dict[str, Any]) -> None:
        metadata = {}
        skip_keys = {
            GenAIAttributes.PROMPT,
            GenAIAttributes.COMPLETION,
            GenAIAttributes.AGENT_TOOLS,
            "agent.tools",
        }

        for key, value in attrs.items():
            if key not in skip_keys and key not in result:
                metadata[key] = self._serialize_value(value)

        if metadata:
            result["metadata"] = safe_json_dumps(metadata)

    def _serialize_value(self, value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value

        return safe_json_dumps(value)

    def shutdown(self) -> SpanExportResult:  # type: ignore[override]
        return SpanExportResult.SUCCESS

    def force_flush(self, timeout_millis: Optional[int] = None) -> bool:
        return True

    @staticmethod
    def get_migration_guide() -> Dict[str, str]:
        return {
            # Deprecated attributes with replacements
            GenAIAttributes.USAGE_PROMPT_TOKENS: GEN_AI_USAGE_INPUT_TOKENS,
            GenAIAttributes.USAGE_COMPLETION_TOKENS: GEN_AI_USAGE_OUTPUT_TOKENS,
            "gen_ai.openai.request.seed": "gen_ai.request.seed",
            "gen_ai.openai.request.response_format": "gen_ai.output.type",
            # Deprecated attributes without direct replacements
            GenAIAttributes.PROMPT: (
                f"Migrate to event-based messaging using {GenAIEventNames.USER_MESSAGE} events"
            ),
            GenAIAttributes.COMPLETION: (
                f"Migrate to event-based messaging using {GenAIEventNames.ASSISTANT_MESSAGE} "
                f"and {GenAIEventNames.CHOICE} events"
            ),
            # Span naming changes
            "Model invoke": "chat",
            "Cycle [UUID]": "execute_event_loop_cycle",
            "Tool: [name]": "execute_tool [name]",
        }

    def get_processor_info(self) -> Dict[str, Any]:
        return {
            "processor_name": "StrandsToOpenInferenceProcessor",
            "version": "0.1.0",
            "supports_events": True,
            "supports_deprecated_attributes": True,
            "supports_new_semantic_conventions": True,
            "debug_enabled": self.debug,
            "migration_guide": self.get_migration_guide(),
            "supported_span_kinds": ["LLM", "AGENT", "CHAIN", "TOOL"],
            "supported_span_names": [
                "chat",
                "execute_event_loop_cycle",
                "execute_tool [name]",
                "invoke_agent [name]",
                "Model invoke",
                "Cycle [UUID]",
                "Tool: [name]",
            ],
            "supported_event_types": [
                GenAIEventNames.USER_MESSAGE,
                GenAIEventNames.ASSISTANT_MESSAGE,
                GenAIEventNames.CHOICE,
                GenAIEventNames.TOOL_MESSAGE,
            ],
            "features": [
                "Event-based message extraction",
                "Enhanced JSON content parsing",
                "Tool result processing",
                "Updated span naming conventions",
                "Deprecated attribute handling with warnings",
                "OpenInference semantic convention compliance",
                "Strands-specific format parsing",
                "Graph node hierarchy mapping",
                "Token usage tracking",
                "Tool call processing",
                "Multi-format content support",
            ],
        }

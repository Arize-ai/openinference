"""Strands Agents to OpenInference Span Processor.

This module provides a span processor that converts Strands Agents' native OpenTelemetry spans
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

from openinference.instrumentation.strands_agents.semantic_conventions import (
    GEN_AI_REQUEST_MAX_TOKENS,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_REQUEST_TEMPERATURE,
    GEN_AI_REQUEST_TOP_P,
    GEN_AI_TOOL_NAME,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    GenAIAttributes,
    GenAIEventNames,
    safe_json_dumps,
)
from openinference.semconv.trace import (
    OpenInferenceMimeTypeValues,
    SpanAttributes,
)

logger = logging.getLogger(__name__)


class StrandsAgentsToOpenInferenceProcessor(SpanProcessor):
    """
    SpanProcessor that converts Strands Agents telemetry attributes to OpenInference format
    for compatibility with OpenInference-compliant backends.

    Important: This processor mutates spans in-place. Add it BEFORE any span
    processors/exporters that should receive the transformed OpenInference spans.

    Example:
        # Correct order: processor first, then exporter
        tracer_provider.add_span_processor(StrandsAgentsToOpenInferenceProcessor())
        tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
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

            # Strip gen_ai.* events after transformation since their content has been
            # extracted into OpenInference attributes. Keeping them would show empty
            # events in the UI (Arize/Phoenix only display exception events).
            self._strip_genai_events(span)

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

    def _strip_genai_events(self, span: ReadableSpan) -> None:
        """Remove gen_ai.* prefixed events from the span after transformation.

        These events have already been processed into OpenInference attributes,
        so keeping them would be redundant and clutter the UI.
        """
        if hasattr(span, "_events") and span._events:
            filtered_events = [
                event
                for event in span._events
                if not (hasattr(event, "name") and event.name.startswith("gen_ai."))
            ]
            span._events = filtered_events

    def _transform_attributes(
        self, attrs: Dict[str, Any], span: ReadableSpan, events: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """
        Transform Strands attributes to OpenInference format, including event processing.
        """
        result: Dict[str, Any] = {}
        span_kind = self._determine_span_kind(span, attrs)
        result[SpanAttributes.OPENINFERENCE_SPAN_KIND] = span_kind
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
        # Check gen_ai.agent.name first (standard GenAI convention), then fall back to agent.name
        agent_name = attrs.get(GenAIAttributes.AGENT_NAME) or attrs.get("agent.name")

        if model_id:
            result[SpanAttributes.LLM_MODEL_NAME] = model_id
            result[GEN_AI_REQUEST_MODEL] = model_id

        if agent_name:
            result[SpanAttributes.LLM_SYSTEM] = "strands-agents"
            result[SpanAttributes.LLM_PROVIDER] = "strands-agents"

        self._handle_tags(attrs, result)

        if span_kind in ["LLM", "AGENT", "CHAIN"]:
            self._handle_llm_span(attrs, result, input_messages, output_messages)
        elif span_kind == "TOOL":
            self._handle_tool_span(attrs, result, events)

        self._map_token_usage(attrs, result)

        passthrough_attrs = [
            SpanAttributes.SESSION_ID,
            SpanAttributes.USER_ID,
            SpanAttributes.LLM_PROMPT_TEMPLATE,
            SpanAttributes.LLM_PROMPT_TEMPLATE_VERSION,
            SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES,
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
                        normalized = self._normalize_message(prompt_data)
                        if normalized.get("message.role") == "user":
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
                        elif "toolUse" in item and isinstance(item["toolUse"], dict):
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

                if "message.content" not in message and "message.tool_calls" not in message:
                    return None

                return message
            elif isinstance(content_data, dict):
                if "text" in content_data:
                    return {"message.role": role, "message.content": str(content_data["text"])}
                else:
                    return {"message.role": role, "message.content": safe_json_dumps(content_data)}
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
                elif "toolUse" in item and isinstance(item["toolUse"], dict):
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
            result[SpanAttributes.LLM_INPUT_MESSAGES] = safe_json_dumps(input_messages)
            self._flatten_messages(input_messages, SpanAttributes.LLM_INPUT_MESSAGES, result)

        if output_messages:
            result[SpanAttributes.LLM_OUTPUT_MESSAGES] = safe_json_dumps(output_messages)
            self._flatten_messages(output_messages, SpanAttributes.LLM_OUTPUT_MESSAGES, result)

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
        span_kind = result.get(SpanAttributes.OPENINFERENCE_SPAN_KIND)
        model_name = (
            result.get(SpanAttributes.LLM_MODEL_NAME)
            or attrs.get(GEN_AI_REQUEST_MODEL)
            or "unknown"
        )

        if span_kind in ["LLM", "AGENT", "CHAIN"]:
            if input_messages:
                if len(input_messages) == 1 and input_messages[0].get("message.role") == "user":
                    # Simple user message
                    input_content = input_messages[0].get("message.content", "")
                    result[SpanAttributes.INPUT_VALUE] = input_content
                    result[SpanAttributes.INPUT_MIME_TYPE] = OpenInferenceMimeTypeValues.TEXT.value
                else:
                    # Complex conversation
                    input_structure = {"messages": input_messages, "model": model_name}
                    result[SpanAttributes.INPUT_VALUE] = safe_json_dumps(input_structure)
                    result[SpanAttributes.INPUT_MIME_TYPE] = OpenInferenceMimeTypeValues.JSON.value

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
                            "completion_tokens": result.get(
                                SpanAttributes.LLM_TOKEN_COUNT_COMPLETION
                            ),
                            "prompt_tokens": result.get(SpanAttributes.LLM_TOKEN_COUNT_PROMPT),
                            "total_tokens": result.get(SpanAttributes.LLM_TOKEN_COUNT_TOTAL),
                        },
                    }
                    result[SpanAttributes.OUTPUT_VALUE] = safe_json_dumps(output_structure)
                    result[SpanAttributes.OUTPUT_MIME_TYPE] = OpenInferenceMimeTypeValues.JSON.value
                else:
                    result[SpanAttributes.OUTPUT_VALUE] = content
                    result[SpanAttributes.OUTPUT_MIME_TYPE] = OpenInferenceMimeTypeValues.TEXT.value

    def _handle_tags(self, attrs: Dict[str, Any], result: Dict[str, Any]) -> None:
        tags = attrs.get(SpanAttributes.TAG_TAGS)

        if tags:
            if isinstance(tags, list):
                result[SpanAttributes.TAG_TAGS] = tags
            elif isinstance(tags, str):
                result[SpanAttributes.TAG_TAGS] = [tags]

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
                cycle_id = attrs.get("event_loop.cycle_id", span_id)
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
            result[SpanAttributes.TOOL_NAME] = tool_name

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
                                        text_parts.append(str(item["text"]))
                                tool_output = (
                                    " ".join(text_parts) if text_parts else str(message_data)
                                )
                            else:
                                tool_output = str(message_data)
                        except (json.JSONDecodeError, TypeError):
                            tool_output = str(message)

            if tool_parameters:
                result[SpanAttributes.TOOL_PARAMETERS] = safe_json_dumps(tool_parameters)

                if tool_name and tool_call_id:
                    # For tool spans, the assistant message contains only tool_calls
                    # (no text content). The empty content is intentional as the
                    # tool call itself IS the message payload.
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

                    result[SpanAttributes.LLM_INPUT_MESSAGES] = safe_json_dumps(input_messages)
                    self._flatten_messages(
                        input_messages, SpanAttributes.LLM_INPUT_MESSAGES, result
                    )

                if isinstance(tool_parameters, dict):
                    if "text" in tool_parameters:
                        result[SpanAttributes.INPUT_VALUE] = tool_parameters["text"]
                        result[SpanAttributes.INPUT_MIME_TYPE] = (
                            OpenInferenceMimeTypeValues.TEXT.value
                        )
                    else:
                        result[SpanAttributes.INPUT_VALUE] = safe_json_dumps(tool_parameters)
                        result[SpanAttributes.INPUT_MIME_TYPE] = (
                            OpenInferenceMimeTypeValues.JSON.value
                        )

            if tool_output:
                result[SpanAttributes.OUTPUT_VALUE] = tool_output
                result[SpanAttributes.OUTPUT_MIME_TYPE] = OpenInferenceMimeTypeValues.TEXT.value

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
            (GenAIAttributes.USAGE_PROMPT_TOKENS, SpanAttributes.LLM_TOKEN_COUNT_PROMPT),
            (GEN_AI_USAGE_INPUT_TOKENS, SpanAttributes.LLM_TOKEN_COUNT_PROMPT),
            (GenAIAttributes.USAGE_COMPLETION_TOKENS, SpanAttributes.LLM_TOKEN_COUNT_COMPLETION),
            (GEN_AI_USAGE_OUTPUT_TOKENS, SpanAttributes.LLM_TOKEN_COUNT_COMPLETION),
            (GenAIAttributes.USAGE_TOTAL_TOKENS, SpanAttributes.LLM_TOKEN_COUNT_TOTAL),
        ]

        for strands_key, openinf_key in token_mappings:
            value = attrs.get(strands_key)
            if value is not None:
                result[openinf_key] = value

    def _map_invocation_parameters(self, attrs: Dict[str, Any], result: Dict[str, Any]) -> None:
        params = {}
        param_mappings = {
            GEN_AI_REQUEST_MAX_TOKENS: "max_tokens",
            GEN_AI_REQUEST_TEMPERATURE: "temperature",
            GEN_AI_REQUEST_TOP_P: "top_p",
        }

        for key, param_key in param_mappings.items():
            if key in attrs:
                params[param_key] = attrs[key]

        if params:
            result[SpanAttributes.LLM_INVOCATION_PARAMETERS] = safe_json_dumps(params)

    def _normalize_message(self, msg: Any) -> Dict[str, Any]:
        if not isinstance(msg, dict):
            return {"message.role": "user", "message.content": str(msg)}

        result: Dict[str, Any] = {}
        result["message.role"] = msg.get("role", "user")

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

    def _add_metadata(self, attrs: Dict[str, Any], final_attrs: Dict[str, Any]) -> None:
        """Add remaining attributes as metadata.

        Args:
            attrs: Original span attributes from Strands
            final_attrs: Transformed OpenInference attributes being built
        """
        metadata = {}
        # Skip keys that have already been processed into OpenInference format:
        # - gen_ai.prompt/completion → llm.input_messages/llm.output_messages
        # - gen_ai.agent.tools/agent.tools → llm.tools.{idx}.*
        # Including these in metadata would be redundant and bloat span data.
        skip_keys = {
            GenAIAttributes.PROMPT,
            GenAIAttributes.COMPLETION,
            GenAIAttributes.AGENT_TOOLS,
            "agent.tools",
        }

        for key, value in attrs.items():
            if key not in skip_keys and key not in final_attrs:
                metadata[key] = self._serialize_value(value)

        if metadata:
            final_attrs[SpanAttributes.METADATA] = safe_json_dumps(metadata)

    def _serialize_value(self, value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value

        return safe_json_dumps(value)

    def shutdown(self) -> None:
        """Shutdown the processor. No-op for this processor."""
        pass

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

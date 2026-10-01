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

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.trace import Span, Status, StatusCode

from openinference.instrumentation import TraceConfig
from openinference.instrumentation.strands_agents.semantic_conventions import (
    GEN_AI_INPUT_MESSAGES,
    GEN_AI_OUTPUT_MESSAGES,
    GEN_AI_PROVIDER_NAME,
    GEN_AI_REQUEST_MAX_TOKENS,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_REQUEST_TEMPERATURE,
    GEN_AI_REQUEST_TOP_P,
    GEN_AI_SYSTEM,
    GEN_AI_SYSTEM_INSTRUCTIONS,
    GEN_AI_TOOL_CALL_ARGUMENTS,
    GEN_AI_TOOL_CALL_RESULT,
    GEN_AI_TOOL_NAME,
    GEN_AI_USAGE_CACHE_CREATION_TOKENS,
    GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
    GEN_AI_USAGE_CACHE_READ_TOKENS,
    GEN_AI_USAGE_CACHE_WRITE_INPUT_TOKENS,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    GenAIAttributes,
    GenAIEventNames,
    safe_json_dumps,
)
from openinference.semconv.trace import (
    AudioAttributes,
    ImageAttributes,
    MessageContentAttributes,
    OpenInferenceMimeTypeValues,
    SpanAttributes,
    VideoAttributes,
)

logger = logging.getLogger(__name__)

# Rules for identifying spans emitted by the Strands Agents SDK.
_STRANDS_SDK_NAME = "strands-agents"
_EVENT_LOOP_CYCLE_ID = "event_loop.cycle_id"

# Raw Strands attributes that carry prompt, message or tool content. The processor turns them
# into OpenInference attributes, so they are kept out of `metadata` and are dropped when the
# TraceConfig hides that side of the span.
_RAW_INPUT_CONTENT = frozenset(
    {
        GEN_AI_SYSTEM_INSTRUCTIONS,
        GEN_AI_INPUT_MESSAGES,
        GEN_AI_TOOL_CALL_ARGUMENTS,
        GenAIAttributes.SYSTEM_PROMPT,
        GenAIAttributes.PROMPT,
    }
)
_RAW_OUTPUT_CONTENT = frozenset(
    {GEN_AI_OUTPUT_MESSAGES, GEN_AI_TOOL_CALL_RESULT, GenAIAttributes.COMPLETION}
)

# Legacy Strands events and the side of the span whose content they carry.
_INPUT_EVENTS = frozenset(
    {
        GenAIEventNames.SYSTEM_MESSAGE,
        GenAIEventNames.USER_MESSAGE,
        GenAIEventNames.ASSISTANT_MESSAGE,
        GenAIEventNames.TOOL_MESSAGE,
    }
)
_OUTPUT_EVENTS = frozenset({GenAIEventNames.CHOICE})

# Content blocks that carry no content of their own.
_EMPTY_BLOCKS = frozenset({"cachePoint"})

# Media blocks whose source can be a stored location, and the URL attribute for each.
_MEDIA_URL_KEYS = {
    "image": f"{MessageContentAttributes.MESSAGE_CONTENT_IMAGE}.{ImageAttributes.IMAGE_URL}",
    "video": f"{MessageContentAttributes.MESSAGE_CONTENT_VIDEO}.{VideoAttributes.VIDEO_URL}",
    "audio": f"{MessageContentAttributes.MESSAGE_CONTENT_AUDIO}.{AudioAttributes.AUDIO_URL}",
}


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

    def __init__(self, debug: bool = False, config: Optional[TraceConfig] = None) -> None:
        """
        Initialize the processor.

        Args:
            debug: Whether to log debug information
            config: Controls which inputs and outputs are kept on the exported spans. Defaults
                to a TraceConfig built from the OPENINFERENCE_HIDE_* environment variables.
        """
        super().__init__()
        self.debug = debug
        self._config = config or TraceConfig()

    def on_start(self, span: Span, parent_context: Optional[Context] = None) -> None:
        """Called when a span is started."""
        pass

    def on_end(self, span: ReadableSpan) -> None:
        """
        Called when a span ends. Transform the span attributes from Strands format
        to OpenInference format.
        """
        if not self._is_strands_span(span):
            return

        original_attrs = dict(span._attributes)  # type: ignore[arg-type]

        try:
            events: List[Any] = []
            if hasattr(span, "_events"):
                events = list(span._events)
            elif hasattr(span, "events"):
                events = list(span.events)

            # Get the OpenInference attributes from the span.
            transformed_attrs = self._transform_attributes(original_attrs, span, events)

            # Combine the original attributes with the OpenInference attributes.
            span._attributes = {
                **self._without_hidden_content(original_attrs),
                **self._mask_attributes(transformed_attrs),
            }
            if not span.status.status_code == StatusCode.ERROR:
                span._status = Status(status_code=StatusCode.OK)

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
            span._attributes = self._without_hidden_content(original_attrs)
            self._strip_hidden_events(span)

    def _mask_attributes(self, attrs: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the TraceConfig to OpenInference attributes; hidden ones are dropped."""
        masked: Dict[str, Any] = {}
        for key, value in attrs.items():
            if key == SpanAttributes.TOOL_PARAMETERS and self._config.hide_inputs:
                # This processor puts the call's argument values in tool.parameters.
                continue
            masked_value = self._config.mask(key, value)
            if masked_value is not None:
                masked[key] = masked_value
        return masked

    def _hidden_sides(self) -> tuple[bool, bool]:
        """Whether the TraceConfig hides any of the input content, and any of the output."""
        config = self._config
        return (
            bool(config.hide_inputs or config.hide_input_messages or config.hide_input_text),
            bool(config.hide_outputs or config.hide_output_messages or config.hide_output_text),
        )

    def _without_hidden_content(self, attrs: Dict[str, Any]) -> Dict[str, Any]:
        """Drop the raw Strands content attributes that the TraceConfig hides."""
        hide_input, hide_output = self._hidden_sides()
        hidden: frozenset[str] = frozenset()
        if hide_input:
            hidden |= _RAW_INPUT_CONTENT
        if hide_output:
            hidden |= _RAW_OUTPUT_CONTENT
        return {key: value for key, value in attrs.items() if key not in hidden}

    def _strip_hidden_events(self, span: ReadableSpan) -> None:
        """Drop the gen_ai.* events whose content the TraceConfig hides.

        Used when the conversion fails, because the events are then exported as they are.
        """
        hide_input, hide_output = self._hidden_sides()
        if not (hide_input or hide_output) or not getattr(span, "_events", None):
            return

        def is_hidden(event: Any) -> bool:
            name = getattr(event, "name", "")
            keys = set(getattr(event, "attributes", None) or {})
            return bool(
                (hide_input and (name in _INPUT_EVENTS or keys & _RAW_INPUT_CONTENT))
                or (hide_output and (name in _OUTPUT_EVENTS or keys & _RAW_OUTPUT_CONTENT))
            )

        span._events = [event for event in span._events if not is_hidden(event)]

    def _is_strands_span(self, span: ReadableSpan) -> bool:
        """Return True if the span was emitted by Strands Agents SDK."""
        attrs = getattr(span, "_attributes", None) or {}

        system = attrs.get(GEN_AI_SYSTEM)
        provider = attrs.get(GEN_AI_PROVIDER_NAME)

        # If the span identifies its SDK, trust that identifier.
        if system is not None or provider is not None:
            return system == _STRANDS_SDK_NAME or provider == _STRANDS_SDK_NAME

        # Current Strands event loop spans do not emit SDK identifiers.
        return span.name == "execute_event_loop_cycle" and _EVENT_LOOP_CYCLE_ID in attrs

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

        # Strands can also record the messages directly on the span instead of as events.
        if not input_messages:
            input_messages = self._parse_genai_messages(attrs.get(GEN_AI_INPUT_MESSAGES))
        if not output_messages:
            output_messages = self._parse_genai_messages(attrs.get(GEN_AI_OUTPUT_MESSAGES))

        self._add_tool_message_names(input_messages, output_messages)

        if not any(m.get("message.role") == "system" for m in input_messages):
            system_message = self._parse_system_instructions(attrs.get(GEN_AI_SYSTEM_INSTRUCTIONS))
            legacy_prompt = attrs.get(GenAIAttributes.SYSTEM_PROMPT)
            if not system_message and legacy_prompt and isinstance(legacy_prompt, str):
                # Strands 1.19-1.33 put the prompt on the agent span as plain text.
                system_message = {"message.role": "system", "message.content": legacy_prompt}
            if system_message:
                input_messages.insert(0, system_message)

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

            if event_name == GenAIEventNames.SYSTEM_MESSAGE:
                content = event_attrs.get("content", "")
                message = self._parse_message_content(content, "system")
                if message:
                    input_messages.append(message)

            elif event_name == GenAIEventNames.USER_MESSAGE:
                content = event_attrs.get("content", "")
                message = self._parse_message_content(content, "user")
                if message:
                    input_messages.append(message)

            elif event_name == GenAIEventNames.ASSISTANT_MESSAGE:
                content = event_attrs.get("content", "")
                message = self._parse_message_content(content, "assistant")
                if message:
                    # Earlier assistant turns are conversation history, so they belong to the input.
                    input_messages.append(message)

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
                if content:
                    message = self._parse_message_content(content, "tool")
                    # Tool spans carry the id on the event; LLM spans carry it in the toolResult.
                    tool_id = event_attrs.get("id") or (message or {}).get("message.tool_call_id")
                    if message and tool_id:
                        message["message.tool_call_id"] = tool_id
                        input_messages.append(message)

            else:
                # Latest GenAI conventions carry everything on the operation details event.
                if GEN_AI_SYSTEM_INSTRUCTIONS in event_attrs:
                    if not any(m.get("message.role") == "system" for m in input_messages):
                        message = self._parse_system_instructions(
                            event_attrs.get(GEN_AI_SYSTEM_INSTRUCTIONS)
                        )
                        if message:
                            input_messages.insert(0, message)
                if GEN_AI_INPUT_MESSAGES in event_attrs:
                    input_messages.extend(
                        self._parse_genai_messages(event_attrs.get(GEN_AI_INPUT_MESSAGES))
                    )
                if GEN_AI_OUTPUT_MESSAGES in event_attrs:
                    output_messages.extend(
                        self._parse_genai_messages(event_attrs.get(GEN_AI_OUTPUT_MESSAGES))
                    )

        return input_messages, output_messages

    def _parse_genai_messages(self, value: Any) -> List[Dict[str, Any]]:
        """Convert gen_ai.input.messages / gen_ai.output.messages ({role, parts}) to messages."""
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                return []
        if isinstance(value, dict):
            value = [value]
        if not isinstance(value, list):
            return []

        messages: List[Dict[str, Any]] = []
        for item in value:
            if not isinstance(item, dict) or not isinstance(item.get("parts"), list):
                continue
            text_parts: List[str] = []
            contents: List[Dict[str, Any]] = []
            tool_calls: List[Dict[str, Any]] = []
            tool_results: List[Dict[str, Any]] = []
            for part in item["parts"]:
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type")
                if part_type == "text" and part.get("content") is not None:
                    text_parts.append(str(part["content"]))
                    contents.append(self._text_item(str(part["content"])))
                elif part_type == "tool_call":
                    tool_calls.append(
                        {
                            "tool_call.id": part.get("id", ""),
                            "tool_call.function.name": part.get("name", ""),
                            "tool_call.function.arguments": safe_json_dumps(
                                part.get("arguments", {})
                            ),
                        }
                    )
                elif part_type == "tool_call_response":
                    tool_results.append(part)
                elif isinstance(part_type, str):
                    # Strands files every other content block as {"type": <block>, "content": ...}.
                    body = (
                        part["content"]
                        if "content" in part
                        else {k: v for k, v in part.items() if k != "type"}
                    )
                    if converted := self._unconverted_item({part_type: body}):
                        contents.append(converted)

            role = item.get("role") or "user"
            message: Dict[str, Any] = {"message.role": role}
            self._set_content(message, role, text_parts, contents)
            if tool_calls:
                message["message.tool_calls"] = tool_calls
            if finish_reason := item.get("finish_reason"):
                message["message.finish_reason"] = finish_reason
            if "message.content" in message or "message.contents" in message or tool_calls:
                messages.append(message)

            # Each tool result is its own tool message, whatever role Strands filed it under.
            for result in tool_results:
                messages.append(
                    {
                        "message.role": "tool",
                        "message.tool_call_id": result.get("id", ""),
                        "message.content": self._tool_response_text(result.get("response")),
                    }
                )
        return messages

    @staticmethod
    def _text_item(text: str) -> Dict[str, Any]:
        return {
            MessageContentAttributes.MESSAGE_CONTENT_TYPE: "text",
            MessageContentAttributes.MESSAGE_CONTENT_TEXT: text,
        }

    def _unconverted_item(self, block: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Content item for a Strands block that has no OpenInference attribute of its own.

        Reasoning keeps its own type, and so do images, video and audio stored in S3, whose
        URI Strands keeps. Strands replaces raw image, audio and video bytes with a placeholder
        before they reach the span, so every other block is kept as its JSON text instead of
        being dropped.
        """
        if _EMPTY_BLOCKS.intersection(block):
            return None
        for kind, url_key in _MEDIA_URL_KEYS.items():
            media = block.get(kind)
            source = media.get("source") if isinstance(media, dict) else None
            location = source.get("location") if isinstance(source, dict) else None
            if isinstance(location, dict) and isinstance(location.get("uri"), str):
                return {
                    MessageContentAttributes.MESSAGE_CONTENT_TYPE: kind,
                    url_key: location["uri"],
                }
        reasoning = block.get("reasoningContent")
        if isinstance(reasoning, dict):
            reasoning_text = reasoning.get("reasoningText")
            if isinstance(reasoning_text, dict) and reasoning_text.get("text"):
                return {
                    MessageContentAttributes.MESSAGE_CONTENT_TYPE: "reasoning",
                    MessageContentAttributes.MESSAGE_CONTENT_TEXT: str(reasoning_text["text"]),
                }
        return self._text_item(safe_json_dumps(block))

    @staticmethod
    def _set_content(
        message: Dict[str, Any], role: str, text_parts: List[str], contents: List[Dict[str, Any]]
    ) -> None:
        """Plain `message.content` for text-only messages, `message.contents` otherwise."""
        if len(contents) > len(text_parts):
            message["message.contents"] = contents
        elif text_parts:
            # Strands joins the text blocks of a system prompt with line breaks.
            message["message.content"] = ("\n" if role == "system" else " ").join(text_parts)

    @staticmethod
    def _message_text(message: Dict[str, Any]) -> str:
        """The text of a message, whether it is plain content or ordered content items."""
        content = message.get("message.content")
        if isinstance(content, str):
            return content
        texts = [
            str(item.get(MessageContentAttributes.MESSAGE_CONTENT_TEXT, ""))
            for item in message.get("message.contents") or []
            if item.get(MessageContentAttributes.MESSAGE_CONTENT_TYPE) == "text"
        ]
        return " ".join(texts)

    def _tool_response_text(self, response: Any) -> str:
        if isinstance(response, str):
            return response
        if isinstance(response, list):
            texts = [b["text"] for b in response if isinstance(b, dict) and "text" in b]
            if texts and len(texts) == len(response):
                return " ".join(str(t) for t in texts)
        return safe_json_dumps(response)

    def _add_tool_message_names(
        self, input_messages: List[Dict[str, Any]], output_messages: List[Dict[str, Any]]
    ) -> None:
        """Name each tool result message after the function whose call it answers."""
        names: Dict[str, str] = {}
        for message in input_messages + output_messages:
            for call in message.get("message.tool_calls") or []:
                call_id = call.get("tool_call.id")
                if call_id:
                    names[call_id] = call.get("tool_call.function.name", "")
        for message in input_messages:
            if message.get("message.role") == "tool" and "message.name" not in message:
                name = names.get(message.get("message.tool_call_id", ""))
                if name:
                    message["message.name"] = name

    def _parse_system_instructions(self, value: Any) -> Optional[Dict[str, Any]]:
        """Build a system message from gen_ai.system_instructions (a JSON list of parts or text)."""
        if not value:
            return None
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                return {"message.role": "system", "message.content": value}
        if isinstance(value, dict):
            value = [value]
        if not isinstance(value, list):
            return {"message.role": "system", "message.content": str(value)}

        text_parts: List[str] = []
        for part in value:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict) and part.get("type", "text") == "text":
                text = part.get("content", part.get("text"))
                if text:
                    text_parts.append(str(text))
        if not text_parts:
            return None
        return {"message.role": "system", "message.content": "\n".join(text_parts)}

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
                contents: List[Dict[str, Any]] = []
                tool_calls: List[Dict[str, Any]] = []
                for item in content_data:
                    if isinstance(item, dict):
                        if "text" in item:
                            text_parts.append(str(item["text"]))
                            contents.append(self._text_item(str(item["text"])))
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
                            if tool_result.get("content"):
                                result_text = self._tool_response_text(tool_result["content"])
                                text_parts.append(result_text)
                                contents.append(self._text_item(result_text))
                            message["message.role"] = "tool"
                            if "toolUseId" in tool_result:
                                message["message.tool_call_id"] = tool_result["toolUseId"]
                        elif converted := self._unconverted_item(item):
                            contents.append(converted)

                self._set_content(message, str(message["message.role"]), text_parts, contents)

                if tool_calls:
                    message["message.tool_calls"] = tool_calls

                if (
                    "message.content" not in message
                    and "message.contents" not in message
                    and "message.tool_calls" not in message
                ):
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
            self._flatten_messages(input_messages, SpanAttributes.LLM_INPUT_MESSAGES, result)

        if output_messages:
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

                if clean_key in ("tool_calls", "contents") and isinstance(value, list):
                    for item_idx, item in enumerate(value):
                        if isinstance(item, dict):
                            for item_key, item_val in item.items():
                                item_dotted_key = (
                                    f"{key_prefix}.{idx}.message.{clean_key}.{item_idx}.{item_key}"
                                )
                                result[item_dotted_key] = self._serialize_value(item_val)
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
                # System prompts stay in llm.input_messages only, so a single user
                # message still yields plain-text input.value.
                non_system = [m for m in input_messages if m.get("message.role") != "system"]
                if (
                    len(non_system) == 1
                    and non_system[0].get("message.role") == "user"
                    and "message.contents" not in non_system[0]
                ):
                    # Simple user message
                    input_content = non_system[0].get("message.content", "")
                    result[SpanAttributes.INPUT_VALUE] = input_content
                    result[SpanAttributes.INPUT_MIME_TYPE] = OpenInferenceMimeTypeValues.TEXT.value
                else:
                    # Complex conversation
                    input_structure = {"messages": input_messages, "model": model_name}
                    result[SpanAttributes.INPUT_VALUE] = safe_json_dumps(input_structure)
                    result[SpanAttributes.INPUT_MIME_TYPE] = OpenInferenceMimeTypeValues.JSON.value

            if output_messages:
                last_message = output_messages[-1]
                content = self._message_text(last_message)

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
        span_context = span.get_span_context()
        span_id = span_context.span_id if span_context is not None else 0

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

        tool_parameters: Optional[Dict[str, Any]] = None
        tool_output: Optional[str] = None
        if events:
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

                # Latest GenAI conventions: the call and its response are tool messages.
                if tool_parameters is None and GEN_AI_INPUT_MESSAGES in event_attrs:
                    tool_parameters = self._latest_tool_arguments(
                        event_attrs.get(GEN_AI_INPUT_MESSAGES)
                    )
                if tool_output is None and GEN_AI_OUTPUT_MESSAGES in event_attrs:
                    tool_output = self._latest_tool_output(event_attrs.get(GEN_AI_OUTPUT_MESSAGES))

        # Latest GenAI conventions also record them as span attributes.
        if tool_parameters is None:
            tool_parameters = self._tool_arguments(
                attrs.get(GEN_AI_TOOL_CALL_ARGUMENTS)
            ) or self._latest_tool_arguments(attrs.get(GEN_AI_INPUT_MESSAGES))
        if tool_output is None:
            tool_output = self._tool_result_text(
                attrs.get(GEN_AI_TOOL_CALL_RESULT)
            ) or self._latest_tool_output(attrs.get(GEN_AI_OUTPUT_MESSAGES))

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
                                "tool_call.function.arguments": safe_json_dumps(tool_parameters),
                            }
                        ],
                    }
                ]

                self._flatten_messages(input_messages, SpanAttributes.LLM_INPUT_MESSAGES, result)

            if isinstance(tool_parameters, dict):
                if "text" in tool_parameters:
                    result[SpanAttributes.INPUT_VALUE] = tool_parameters["text"]
                    result[SpanAttributes.INPUT_MIME_TYPE] = OpenInferenceMimeTypeValues.TEXT.value
                else:
                    result[SpanAttributes.INPUT_VALUE] = safe_json_dumps(tool_parameters)
                    result[SpanAttributes.INPUT_MIME_TYPE] = OpenInferenceMimeTypeValues.JSON.value

        if tool_output:
            result[SpanAttributes.OUTPUT_VALUE] = tool_output
            result[SpanAttributes.OUTPUT_MIME_TYPE] = OpenInferenceMimeTypeValues.TEXT.value

    @staticmethod
    def _tool_arguments(value: Any) -> Optional[Dict[str, Any]]:
        """Tool call arguments as a dict, the way legacy tool messages are read."""
        if value is None or value == "":
            return None
        try:
            data = json.loads(value) if isinstance(value, str) else value
        except (json.JSONDecodeError, TypeError):
            return {"input": str(value)}
        return data if isinstance(data, dict) else {"input": str(data)}

    def _tool_result_text(self, value: Any) -> Optional[str]:
        if value is None or value == "":
            return None
        try:
            data = json.loads(value) if isinstance(value, str) else value
        except (json.JSONDecodeError, TypeError):
            return str(value)
        return self._tool_response_text(data)

    @staticmethod
    def _latest_tool_part(value: Any, part_type: str) -> Optional[Dict[str, Any]]:
        """First part of the given type in a gen_ai.input/output.messages value."""
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                return None
        for message in value if isinstance(value, list) else []:
            parts = message.get("parts") if isinstance(message, dict) else None
            for part in parts if isinstance(parts, list) else []:
                if isinstance(part, dict) and part.get("type") == part_type:
                    return part
        return None

    def _latest_tool_arguments(self, value: Any) -> Optional[Dict[str, Any]]:
        part = self._latest_tool_part(value, "tool_call")
        return self._tool_arguments(part.get("arguments")) if part else None

    def _latest_tool_output(self, value: Any) -> Optional[str]:
        part = self._latest_tool_part(value, "tool_call_response")
        if part is None or part.get("response") is None:
            return None
        return self._tool_response_text(part["response"])

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

        cache_read = (
            attrs.get(GEN_AI_USAGE_CACHE_READ_TOKENS)
            or attrs.get(GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS)
            or 0
        )
        cache_write = (
            attrs.get(GEN_AI_USAGE_CACHE_CREATION_TOKENS)
            or attrs.get(GEN_AI_USAGE_CACHE_WRITE_INPUT_TOKENS)
            or 0
        )
        if cache_read:
            result[SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ] = cache_read
        if cache_write:
            result[SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE] = cache_write
        if cache_read or cache_write:
            # The OpenInference prompt count includes its cached tokens. Strands 1.34+ and some
            # providers already count them in the input tokens; the totals tell which, the same
            # way Strands decides it: prompt + completion == total means they are included.
            prompt = result.get(SpanAttributes.LLM_TOKEN_COUNT_PROMPT) or 0
            completion = result.get(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION) or 0
            total = result.get(SpanAttributes.LLM_TOKEN_COUNT_TOTAL)
            if total is None or prompt + completion != total:
                result[SpanAttributes.LLM_TOKEN_COUNT_PROMPT] = prompt + cache_read + cache_write

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
        # Prompt and message content is skipped as well so it is not exported a third time.
        skip_keys = {
            GenAIAttributes.AGENT_TOOLS,
            "agent.tools",
            *_RAW_INPUT_CONTENT,
            *_RAW_OUTPUT_CONTENT,
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

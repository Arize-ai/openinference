import json
import logging
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple, cast

from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_OPERATION_NAME,
    GEN_AI_REQUEST_FREQUENCY_PENALTY,
    GEN_AI_REQUEST_MAX_TOKENS,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_REQUEST_PRESENCE_PENALTY,
    GEN_AI_REQUEST_SEED,
    GEN_AI_REQUEST_STOP_SEQUENCES,
    GEN_AI_REQUEST_TEMPERATURE,
    GEN_AI_REQUEST_TOP_K,
    GEN_AI_REQUEST_TOP_P,
    GEN_AI_SYSTEM,
    GEN_AI_TOOL_CALL_ID,
    GEN_AI_TOOL_DESCRIPTION,
    GEN_AI_TOOL_NAME,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    GenAiOperationNameValues,
)

from openinference.instrumentation import safe_json_dumps
from openinference.semconv.trace import (
    MessageAttributes,
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolAttributes,
    ToolCallAttributes,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# Many event related conventions are not in the opentelemetry-python package yet
class OTELConventions:
    EVENTS = "events"
    EVENT_NAME = "event.name"


class GenAIConversationID:
    CONVERSATION_ID = "gen_ai.conversation.id"


class GenAIEventNames:
    SYSTEM_MESSAGE = "gen_ai.system.message"
    USER_MESSAGE = "gen_ai.user.message"
    ASSISTANT_MESSAGE = "gen_ai.assistant.message"
    TOOL_MESSAGE = "gen_ai.tool.message"
    CHOICE = "gen_ai.choice"


class GenAICommonBodyFields:
    ROLE = "role"
    CONTENT = "content"


class GenAISystemMessageBodyFields(GenAICommonBodyFields):
    pass


class GenAIUserMessageBodyFields(GenAICommonBodyFields):
    pass


class GenAIAssistantMessageBodyFields(GenAICommonBodyFields):
    TOOL_CALLS = "tool_calls"


class GenAIToolCallFields:
    FUNCTION = "function"
    ID = "id"
    TYPE = "type"


class GenAIFunctionFields:
    NAME = "name"
    ARGUMENTS = "arguments"


class GenAIToolMessageBodyFields(GenAICommonBodyFields):
    ID = "id"


class GenAIChoiceBodyFields:
    INDEX = "index"
    FINISH_REASON = "finish_reason"
    MESSAGE = "message"
    TOOL_CALLS = "tool_calls"


# Pydantic-specific constants as they're not part of OTEL
class PydanticAgentName:
    AGENT = "agent_name"


class PydanticTools:
    TOOLS = "tools"


class PydanticGenAIAttribute:
    GEN_AI = "gen_ai"


class PydanticGenAITool:
    TOOL = "tool"


class PydanticCustomAttributes:
    MODEL_REQUEST_PARAMETERS = "model_request_parameters"


class PydanticModelRequestParameters:
    TOOLS = "output_tools"
    NAME = "name"
    DESCRIPTION = "description"
    PARAMETERS = "parameters"


class PydanticModelRequestParametersTool:
    NAME = "name"
    DESCRIPTION = "description"
    PARAMETERS = "parameters"


class PydanticMessageToolCallFunctionFinalResult:
    FINAL_RESULT = "final_result"


class PydanticMessageRoleUser:
    USER = "user"


class PydanticFinalResult:
    FINAL_RESULT = "final_result"


class PydanticAllMessagesEvents:
    ALL_MESSAGES_EVENTS = "all_messages_events"


def get_attributes(gen_ai_attrs: Mapping[str, Any]) -> Iterator[Tuple[str, Any]]:
    """
    Main function to extract OpenInference attributes from GenAI attributes.

    Args:
        gen_ai_attrs: Dictionary with OTEL GenAI semantic convention attributes

    Returns:
        Iterator of (key, value) pairs for OpenInference attributes
    """
    yield from _extract_agent_attributes(gen_ai_attrs)
    yield from _extract_common_attributes(gen_ai_attrs)
    yield from _extract_llm_attributes(gen_ai_attrs)
    yield from _extract_tool_attributes(gen_ai_attrs)


def _extract_common_attributes(gen_ai_attrs: Mapping[str, Any]) -> Iterator[Tuple[str, Any]]:
    """Extract attributes common to all operation types."""

    # We want to ignore token counts on non LLM spans. Pydantic adds token counts to agents
    ignore_token_counts = PydanticAgentName.AGENT in gen_ai_attrs
    if GEN_AI_OPERATION_NAME in gen_ai_attrs:
        try:
            operation = gen_ai_attrs[GEN_AI_OPERATION_NAME]
            span_kind = _map_operation_to_span_kind(operation)
            yield SpanAttributes.OPENINFERENCE_SPAN_KIND, span_kind.value
        except ValueError:
            yield (
                SpanAttributes.OPENINFERENCE_SPAN_KIND,
                OpenInferenceSpanKindValues.UNKNOWN.value,
            )
    elif GEN_AI_TOOL_NAME in gen_ai_attrs:
        yield SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.TOOL.value
    elif PydanticAgentName.AGENT in gen_ai_attrs:
        yield SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.AGENT.value
    elif PydanticTools.TOOLS in gen_ai_attrs:
        yield SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.CHAIN.value

    if GEN_AI_SYSTEM in gen_ai_attrs:
        yield SpanAttributes.LLM_SYSTEM, gen_ai_attrs[GEN_AI_SYSTEM]

    if GEN_AI_REQUEST_MODEL in gen_ai_attrs:
        yield SpanAttributes.LLM_MODEL_NAME, gen_ai_attrs[GEN_AI_REQUEST_MODEL]

    if GEN_AI_USAGE_INPUT_TOKENS in gen_ai_attrs and not ignore_token_counts:
        yield (
            SpanAttributes.LLM_TOKEN_COUNT_PROMPT,
            gen_ai_attrs[GEN_AI_USAGE_INPUT_TOKENS],
        )

    if GEN_AI_USAGE_OUTPUT_TOKENS in gen_ai_attrs and not ignore_token_counts:
        yield (
            SpanAttributes.LLM_TOKEN_COUNT_COMPLETION,
            gen_ai_attrs[GEN_AI_USAGE_OUTPUT_TOKENS],
        )

    if (
        GEN_AI_USAGE_INPUT_TOKENS in gen_ai_attrs
        and GEN_AI_USAGE_OUTPUT_TOKENS in gen_ai_attrs
        and not ignore_token_counts
    ):
        total_tokens = (
            gen_ai_attrs[GEN_AI_USAGE_INPUT_TOKENS] + gen_ai_attrs[GEN_AI_USAGE_OUTPUT_TOKENS]
        )
        yield SpanAttributes.LLM_TOKEN_COUNT_TOTAL, total_tokens

    if GenAIConversationID.CONVERSATION_ID in gen_ai_attrs:
        yield SpanAttributes.SESSION_ID, gen_ai_attrs[GenAIConversationID.CONVERSATION_ID]


def _extract_agent_attributes(gen_ai_attrs: Mapping[str, Any]) -> Iterator[Tuple[str, Any]]:
    """Extract attributes specific to agent operations."""
    if PydanticFinalResult.FINAL_RESULT in gen_ai_attrs:
        yield SpanAttributes.OUTPUT_VALUE, gen_ai_attrs[PydanticFinalResult.FINAL_RESULT]

    # Extract input value from pydantic_ai.all_messages (v2 AGENT spans)
    if "pydantic_ai.all_messages" in gen_ai_attrs:
        all_messages_str = gen_ai_attrs["pydantic_ai.all_messages"]
        if isinstance(all_messages_str, str):
            try:
                all_messages = json.loads(all_messages_str)
                if isinstance(all_messages, list) and all_messages:
                    # Find last user message for input value
                    for msg in reversed(all_messages):
                        if msg.get("role") == "user" and "parts" in msg:
                            for part in msg["parts"]:
                                if isinstance(part, dict) and part.get("type") == "text":
                                    yield SpanAttributes.INPUT_VALUE, part.get("content", "")
                                    return
            except json.JSONDecodeError:
                pass
    # Extract input value from all_messages_events (v1 AGENT spans)
    elif PydanticAllMessagesEvents.ALL_MESSAGES_EVENTS in gen_ai_attrs:
        events = _parse_events(gen_ai_attrs[PydanticAllMessagesEvents.ALL_MESSAGES_EVENTS])
        if events:
            input_messages = _extract_llm_input_messages(events)
            input_value = _find_llm_input_value(input_messages)
            if input_value is not None:
                yield SpanAttributes.INPUT_VALUE, input_value


def _extract_llm_attributes(gen_ai_attrs: Mapping[str, Any]) -> Iterator[Tuple[str, Any]]:
    """Extract attributes specific to LLM operations (chat, text completion)."""

    invocation_params = _extract_invocation_parameters(gen_ai_attrs)
    if invocation_params:
        yield SpanAttributes.LLM_INVOCATION_PARAMETERS, safe_json_dumps(invocation_params)

    yield from _extract_tools_attributes(gen_ai_attrs)

    # Extract messages from gen_ai.input.messages/output.messages (v2 LLM spans)
    # v1 with event_mode="attributes" will have both gen_ai messages AND events
    # v2 will only have gen_ai messages
    if "gen_ai.input.messages" in gen_ai_attrs and OTELConventions.EVENTS not in gen_ai_attrs:
        yield from _extract_from_gen_ai_messages(gen_ai_attrs)
        return

    # Extract messages from OTEL events (v1 LLM spans with event_mode="events")
    if OTELConventions.EVENTS in gen_ai_attrs:
        events = _parse_events(gen_ai_attrs[OTELConventions.EVENTS])
        if events:
            input_messages = _extract_llm_input_messages(events)
            for index, message in enumerate(input_messages):
                for key, value in message.items():
                    yield f"{SpanAttributes.LLM_INPUT_MESSAGES}.{index}.{key}", value

            input_value = _find_llm_input_value(input_messages)
            if input_value is not None:
                yield SpanAttributes.INPUT_VALUE, input_value

            output_messages = _extract_llm_output_messages(events)
            for index, message in enumerate(output_messages):
                for key, value in _flatten_message(message).items():
                    yield f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.{index}.{key}", value

            output_value = _find_llm_output_value(output_messages)
            if output_value is not None:
                yield SpanAttributes.OUTPUT_VALUE, output_value


def _flatten_message(message: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a message dictionary for output, handling nested structures like tool_calls."""
    flattened = {}

    for key, value in message.items():
        if key != MessageAttributes.MESSAGE_TOOL_CALLS:
            flattened[key] = value

    if MessageAttributes.MESSAGE_TOOL_CALLS in message:
        tool_calls = message[MessageAttributes.MESSAGE_TOOL_CALLS]
        for idx, tool_call in enumerate(tool_calls):
            for tc_key, tc_value in tool_call.items():
                flattened[f"{MessageAttributes.MESSAGE_TOOL_CALLS}.{idx}.{tc_key}"] = tc_value

    return flattened


def _extract_tool_attributes(gen_ai_attrs: Mapping[str, Any]) -> Iterator[Tuple[str, Any]]:
    """Extract attributes specific to tool operations."""

    if GEN_AI_TOOL_NAME in gen_ai_attrs:
        yield SpanAttributes.TOOL_NAME, gen_ai_attrs[GEN_AI_TOOL_NAME]

    if GEN_AI_TOOL_DESCRIPTION in gen_ai_attrs:
        yield SpanAttributes.TOOL_DESCRIPTION, gen_ai_attrs[GEN_AI_TOOL_DESCRIPTION]

    if GEN_AI_TOOL_CALL_ID in gen_ai_attrs:
        yield ToolCallAttributes.TOOL_CALL_ID, gen_ai_attrs[GEN_AI_TOOL_CALL_ID]

    if OTELConventions.EVENTS in gen_ai_attrs:
        events = _parse_events(gen_ai_attrs[OTELConventions.EVENTS])

        for event in events:
            if event.get(OTELConventions.EVENT_NAME) == GenAIEventNames.TOOL_MESSAGE:
                if GenAIToolMessageBodyFields.ID in event:
                    yield ToolCallAttributes.TOOL_CALL_ID, event[GenAIToolMessageBodyFields.ID]

            if event.get(OTELConventions.EVENT_NAME) == GenAIEventNames.ASSISTANT_MESSAGE:
                if GenAIAssistantMessageBodyFields.TOOL_CALLS in event and isinstance(
                    event[GenAIAssistantMessageBodyFields.TOOL_CALLS], list
                ):
                    for tool_call in event[GenAIAssistantMessageBodyFields.TOOL_CALLS]:
                        if GenAIToolCallFields.FUNCTION in tool_call and isinstance(
                            tool_call[GenAIToolCallFields.FUNCTION], dict
                        ):
                            if GenAIFunctionFields.NAME in tool_call[GenAIToolCallFields.FUNCTION]:
                                yield (
                                    ToolCallAttributes.TOOL_CALL_FUNCTION_NAME,
                                    tool_call[GenAIToolCallFields.FUNCTION][
                                        GenAIFunctionFields.NAME
                                    ],
                                )
                            if (
                                GenAIFunctionFields.ARGUMENTS
                                in tool_call[GenAIToolCallFields.FUNCTION]
                            ):
                                yield (
                                    ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON,
                                    tool_call[GenAIToolCallFields.FUNCTION][
                                        GenAIFunctionFields.ARGUMENTS
                                    ],
                                )


def _extract_tools_attributes(gen_ai_attrs: Mapping[str, Any]) -> Iterator[Tuple[str, Any]]:
    """Extract tool definitions from model request parameters."""
    if PydanticCustomAttributes.MODEL_REQUEST_PARAMETERS not in gen_ai_attrs:
        return

    try:
        params = _parse_json_value(gen_ai_attrs[PydanticCustomAttributes.MODEL_REQUEST_PARAMETERS])
        if not params or not isinstance(params, dict):
            return

        tools = []
        if PydanticModelRequestParameters.TOOLS in params and isinstance(
            params[PydanticModelRequestParameters.TOOLS], list
        ):
            for tool in params[PydanticModelRequestParameters.TOOLS]:
                if not isinstance(tool, dict):
                    continue

                tool_info: Dict[str, Any] = {}
                if PydanticModelRequestParametersTool.NAME in tool:
                    tool_info[SpanAttributes.TOOL_NAME] = tool[
                        PydanticModelRequestParametersTool.NAME
                    ]
                if PydanticModelRequestParametersTool.DESCRIPTION in tool:
                    tool_info[SpanAttributes.TOOL_DESCRIPTION] = tool[
                        PydanticModelRequestParametersTool.DESCRIPTION
                    ]
                if PydanticModelRequestParametersTool.PARAMETERS in tool and isinstance(
                    tool[PydanticModelRequestParametersTool.PARAMETERS], dict
                ):
                    tool_info[ToolAttributes.TOOL_JSON_SCHEMA] = safe_json_dumps(
                        tool[PydanticModelRequestParametersTool.PARAMETERS]
                    )

                if tool_info:
                    tools.append(tool_info)

        for idx, tool in enumerate(tools):
            for key, value in tool.items():
                yield f"{SpanAttributes.LLM_TOOLS}.{idx}.{key}", value
    except Exception as e:
        logger.debug(f"Error parsing model request parameters: {e}")


def _extract_invocation_parameters(gen_ai_attrs: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract invocation parameters that map to OpenInference."""
    param_keys = [
        (GEN_AI_REQUEST_TEMPERATURE, "temperature"),
        (GEN_AI_REQUEST_TOP_P, "top_p"),
        (GEN_AI_REQUEST_TOP_K, "top_k"),
        (GEN_AI_REQUEST_MAX_TOKENS, "max_tokens"),
        (GEN_AI_REQUEST_FREQUENCY_PENALTY, "frequency_penalty"),
        (GEN_AI_REQUEST_PRESENCE_PENALTY, "presence_penalty"),
        (GEN_AI_REQUEST_STOP_SEQUENCES, "stop"),
        (GEN_AI_REQUEST_SEED, "seed"),
    ]

    invocation_params = {}
    for gen_ai_key, param_name in param_keys:
        if gen_ai_key in gen_ai_attrs:
            invocation_params[param_name] = gen_ai_attrs[gen_ai_key]

    return invocation_params


def _extract_llm_input_messages(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract input messages from events for LLM operations."""
    input_messages: List[Dict[str, Any]] = []

    for event in events:
        if event.get(OTELConventions.EVENT_NAME) == GenAIEventNames.USER_MESSAGE:
            message: Dict[str, Any] = {}
            if GenAIUserMessageBodyFields.ROLE in event:
                message[MessageAttributes.MESSAGE_ROLE] = event[GenAIUserMessageBodyFields.ROLE]

            if GenAIUserMessageBodyFields.CONTENT in event:
                message[MessageAttributes.MESSAGE_CONTENT] = event[
                    GenAIUserMessageBodyFields.CONTENT
                ]

            if message:
                input_messages.append(message)

        if event.get(OTELConventions.EVENT_NAME) == GenAIEventNames.SYSTEM_MESSAGE:
            message = {}
            if GenAISystemMessageBodyFields.ROLE in event:
                message[MessageAttributes.MESSAGE_ROLE] = event[GenAISystemMessageBodyFields.ROLE]

            if GenAISystemMessageBodyFields.CONTENT in event:
                message[MessageAttributes.MESSAGE_CONTENT] = event[
                    GenAISystemMessageBodyFields.CONTENT
                ]

            if message:
                input_messages.append(message)

        if event.get(OTELConventions.EVENT_NAME) == GenAIEventNames.ASSISTANT_MESSAGE:
            message = {}
            if GenAIAssistantMessageBodyFields.ROLE in event:
                message[MessageAttributes.MESSAGE_ROLE] = event[
                    GenAIAssistantMessageBodyFields.ROLE
                ]

            if GenAIAssistantMessageBodyFields.CONTENT in event:
                message[MessageAttributes.MESSAGE_CONTENT] = event[
                    GenAIAssistantMessageBodyFields.CONTENT
                ]

            if GenAIAssistantMessageBodyFields.TOOL_CALLS in event and isinstance(
                event[GenAIAssistantMessageBodyFields.TOOL_CALLS], list
            ):
                tool_calls = []
                for tool_call in event[GenAIAssistantMessageBodyFields.TOOL_CALLS]:
                    tc = _process_tool_call(tool_call)
                    if tc:
                        tool_calls.append(tc)

                if tool_calls:
                    message[MessageAttributes.MESSAGE_TOOL_CALLS] = tool_calls

            if message:
                input_messages.append(message)

        if event.get(OTELConventions.EVENT_NAME) == GenAIEventNames.TOOL_MESSAGE:
            message = {}
            if GenAIToolMessageBodyFields.ROLE in event:
                message[MessageAttributes.MESSAGE_ROLE] = event[GenAIToolMessageBodyFields.ROLE]

            if GenAIToolMessageBodyFields.CONTENT in event:
                message[MessageAttributes.MESSAGE_CONTENT] = event[
                    GenAIToolMessageBodyFields.CONTENT
                ]

            if GenAIToolMessageBodyFields.ID in event:
                message[ToolCallAttributes.TOOL_CALL_ID] = event[GenAIToolMessageBodyFields.ID]

            if message:
                input_messages.append(message)

    return input_messages


def _process_tool_call(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    """Process a tool call and extract relevant information."""
    if not isinstance(tool_call, dict):
        return {}

    tc: Dict[str, Any] = {}
    if GenAIToolCallFields.ID in tool_call:
        tc[ToolCallAttributes.TOOL_CALL_ID] = tool_call[GenAIToolCallFields.ID]

    if GenAIToolCallFields.FUNCTION in tool_call and isinstance(
        tool_call[GenAIToolCallFields.FUNCTION], dict
    ):
        function = tool_call[GenAIToolCallFields.FUNCTION]
        if GenAIFunctionFields.NAME in function:
            tc[MessageAttributes.MESSAGE_FUNCTION_CALL_NAME] = function[GenAIFunctionFields.NAME]

        if GenAIFunctionFields.ARGUMENTS in function:
            tc[MessageAttributes.MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON] = function[
                GenAIFunctionFields.ARGUMENTS
            ]

    return tc


def _extract_llm_output_messages(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract output messages from events for LLM operations."""
    output_messages: List[Dict[str, Any]] = []

    for event in events:
        if event.get(OTELConventions.EVENT_NAME) == GenAIEventNames.CHOICE:
            choice: Dict[str, Any] = {}

            if GenAIChoiceBodyFields.MESSAGE in event and isinstance(
                event[GenAIChoiceBodyFields.MESSAGE], dict
            ):
                message: Dict[str, Any] = {}
                message_data = event[GenAIChoiceBodyFields.MESSAGE]

                if GenAICommonBodyFields.ROLE in message_data:
                    message[MessageAttributes.MESSAGE_ROLE] = message_data[
                        GenAICommonBodyFields.ROLE
                    ]

                if GenAICommonBodyFields.CONTENT in message_data:
                    message[MessageAttributes.MESSAGE_CONTENT] = message_data[
                        GenAICommonBodyFields.CONTENT
                    ]

                if GenAIChoiceBodyFields.TOOL_CALLS in message_data and isinstance(
                    message_data[GenAIChoiceBodyFields.TOOL_CALLS], list
                ):
                    tool_calls = []
                    for tool_call in message_data[GenAIChoiceBodyFields.TOOL_CALLS]:
                        tc = _process_tool_call(tool_call)
                        if tc:
                            tool_calls.append(tc)

                    if tool_calls:
                        message[MessageAttributes.MESSAGE_TOOL_CALLS] = tool_calls

                message.update(choice)

                if message:
                    output_messages.append(message)

    return output_messages


def _find_llm_input_value(input_messages: List[Dict[str, Any]]) -> Optional[str]:
    """Find input value from last user message for LLM operations."""
    for message in reversed(input_messages):
        if (
            message.get(MessageAttributes.MESSAGE_ROLE) == PydanticMessageRoleUser.USER
            and MessageAttributes.MESSAGE_CONTENT in message
        ):
            content = message[MessageAttributes.MESSAGE_CONTENT]
            if isinstance(content, str):
                return content
            return None

    return None


def _find_llm_output_value(output_messages: List[Dict[str, Any]]) -> Optional[str]:
    """Find output value from message in choice events."""
    for message in output_messages:
        if MessageAttributes.MESSAGE_CONTENT in message:
            content = message[MessageAttributes.MESSAGE_CONTENT]
            if isinstance(content, str):
                return content
            return None

        if MessageAttributes.MESSAGE_TOOL_CALLS in message and isinstance(
            message[MessageAttributes.MESSAGE_TOOL_CALLS], list
        ):
            for tool_call in message[MessageAttributes.MESSAGE_TOOL_CALLS]:
                if (
                    MessageAttributes.MESSAGE_FUNCTION_CALL_NAME in tool_call
                    and tool_call[MessageAttributes.MESSAGE_FUNCTION_CALL_NAME]
                    == PydanticMessageToolCallFunctionFinalResult.FINAL_RESULT
                ):
                    if MessageAttributes.MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON in tool_call:
                        args = tool_call[MessageAttributes.MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON]
                        if isinstance(args, str):
                            return args
                        return None
    return None


def _map_operation_to_span_kind(operation: str) -> OpenInferenceSpanKindValues:
    """Map GenAIOperationName to OpenInferenceSpanKindValues."""
    mapping = {
        GenAiOperationNameValues.CHAT.value: OpenInferenceSpanKindValues.LLM,
        GenAiOperationNameValues.TEXT_COMPLETION.value: OpenInferenceSpanKindValues.LLM,
        GenAiOperationNameValues.CREATE_AGENT.value: OpenInferenceSpanKindValues.AGENT,
        GenAiOperationNameValues.EMBEDDINGS.value: OpenInferenceSpanKindValues.EMBEDDING,
        GenAiOperationNameValues.EXECUTE_TOOL.value: OpenInferenceSpanKindValues.TOOL,
        GenAiOperationNameValues.INVOKE_AGENT.value: OpenInferenceSpanKindValues.AGENT,
        GenAiOperationNameValues.GENERATE_CONTENT.value: OpenInferenceSpanKindValues.LLM,
    }
    return mapping.get(operation, OpenInferenceSpanKindValues.UNKNOWN)


def _parse_events(events_value: Any) -> List[Dict[str, Any]]:
    """Parse events from string or list."""
    if isinstance(events_value, str):
        try:
            parsed = json.loads(events_value)
            if isinstance(parsed, list):
                return parsed
            return []
        except json.JSONDecodeError:
            return []
    elif isinstance(events_value, list):
        return cast(List[Dict[str, Any]], events_value)
    return []


def _parse_json_value(json_value: Any) -> Any:
    """Parse JSON from string or return as is if already parsed."""
    if isinstance(json_value, str):
        try:
            return json.loads(json_value)
        except json.JSONDecodeError:
            return None
    return json_value


def _extract_from_gen_ai_messages(gen_ai_attrs: Mapping[str, Any]) -> Iterator[Tuple[str, Any]]:
    """Extract OpenInference attributes from pydantic_ai v2 gen_ai messages format."""

    # Extract input messages and input value
    input_value = None
    if "gen_ai.input.messages" in gen_ai_attrs:
        input_messages_str = gen_ai_attrs["gen_ai.input.messages"]
        if isinstance(input_messages_str, str):
            try:
                input_messages = json.loads(input_messages_str)
                if isinstance(input_messages, list):
                    for index, msg in enumerate(input_messages):
                        if "role" in msg:
                            yield (
                                f"{SpanAttributes.LLM_INPUT_MESSAGES}.{index}.{MessageAttributes.MESSAGE_ROLE}",
                                msg["role"],
                            )

                        # Extract content from parts
                        if "parts" in msg and isinstance(msg["parts"], list):
                            for part in msg["parts"]:
                                if (
                                    isinstance(part, dict)
                                    and part.get("type") == "text"
                                    and "content" in part
                                ):
                                    yield (
                                        f"{SpanAttributes.LLM_INPUT_MESSAGES}.{index}.{MessageAttributes.MESSAGE_CONTENT}",
                                        part["content"],
                                    )

                                    # Set INPUT_VALUE for the last user message found
                                    if msg.get("role") == "user":
                                        input_value = part["content"]
                                    break
                    if input_value is not None:
                        yield SpanAttributes.INPUT_VALUE, input_value
            except json.JSONDecodeError:
                pass

    # Extract output messages
    if "gen_ai.output.messages" in gen_ai_attrs:
        output_messages_str = gen_ai_attrs["gen_ai.output.messages"]
        if isinstance(output_messages_str, str):
            try:
                output_messages = json.loads(output_messages_str)
                if isinstance(output_messages, list):
                    for index, msg in enumerate(output_messages):
                        if "role" in msg:
                            yield (
                                f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.{index}.{MessageAttributes.MESSAGE_ROLE}",
                                msg["role"],
                            )

                        # Extract content or tool calls from parts
                        if "parts" in msg and isinstance(msg["parts"], list):
                            for part in msg["parts"]:
                                if isinstance(part, dict):
                                    if part.get("type") == "text" and "content" in part:
                                        yield (
                                            f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.{index}.{MessageAttributes.MESSAGE_CONTENT}",
                                            part["content"],
                                        )
                                        break
                                    elif part.get("type") == "tool_call":
                                        # Extract tool call information
                                        if "name" in part:
                                            yield (
                                                f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.{index}.{MessageAttributes.MESSAGE_TOOL_CALLS}.0.{MessageAttributes.MESSAGE_FUNCTION_CALL_NAME}",
                                                part["name"],
                                            )
                                        if "arguments" in part:
                                            yield (
                                                f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.{index}.{MessageAttributes.MESSAGE_TOOL_CALLS}.0.{MessageAttributes.MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON}",
                                                part["arguments"],
                                            )

                                            # Also set OUTPUT_VALUE for final_result
                                            if part.get("name") == "final_result":
                                                yield SpanAttributes.OUTPUT_VALUE, part["arguments"]
                                        break
            except json.JSONDecodeError:
                pass

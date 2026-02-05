"""Semantic conventions for Microsoft Agent Framework telemetry.

This module defines attribute constants used by Microsoft Agent Framework's
OpenTelemetry instrumentation and provides transformation functions to convert
GenAI attributes to OpenInference format.

Reference: agent_framework/observability.py OtelAttr enum
"""

import json
import logging
from typing import Any, Dict, Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Operation names
OPERATION = "gen_ai.operation.name"
CHAT_COMPLETION_OPERATION = "chat"
TOOL_EXECUTION_OPERATION = "execute_tool"
AGENT_INVOKE_OPERATION = "invoke_agent"

# Provider and system
PROVIDER_NAME = "gen_ai.provider.name"

# Request attributes
LLM_REQUEST_MODEL = "gen_ai.request.model"
LLM_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
LLM_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
LLM_REQUEST_TOP_P = "gen_ai.request.top_p"

# Response attributes
LLM_RESPONSE_MODEL = "gen_ai.response.model"
FINISH_REASONS = "gen_ai.response.finish_reasons"
RESPONSE_ID = "gen_ai.response.id"

# Usage attributes
INPUT_TOKENS = "gen_ai.usage.input_tokens"
OUTPUT_TOKENS = "gen_ai.usage.output_tokens"

# Tool attributes
TOOL_CALL_ID = "gen_ai.tool.call.id"
TOOL_DESCRIPTION = "gen_ai.tool.description"
TOOL_NAME = "gen_ai.tool.name"
TOOL_DEFINITIONS = "gen_ai.tool.definitions"
TOOL_ARGUMENTS = "gen_ai.tool.call.arguments"
TOOL_RESULT = "gen_ai.tool.call.result"

# Agent attributes
AGENT_ID = "gen_ai.agent.id"
AGENT_NAME = "gen_ai.agent.name"
AGENT_DESCRIPTION = "gen_ai.agent.description"
CONVERSATION_ID = "gen_ai.conversation.id"

# Message attributes
INPUT_MESSAGES = "gen_ai.input.messages"
OUTPUT_MESSAGES = "gen_ai.output.messages"
SYSTEM_INSTRUCTIONS = "gen_ai.system_instructions"

# Workflow attributes
WORKFLOW_ID = "workflow.id"
WORKFLOW_NAME = "workflow.name"
WORKFLOW_RUN_SPAN = "workflow.run"

# Executor attributes
EXECUTOR_ID = "executor.id"
EXECUTOR_TYPE = "executor.type"
EXECUTOR_PROCESS_SPAN = "executor.process"

# Edge group attributes
EDGE_GROUP_TYPE = "edge_group.type"
EDGE_GROUP_ID = "edge_group.id"


def safe_json_dumps(obj: Any) -> str:
    """Safely serialize an object to JSON string.

    Args:
        obj: Object to serialize

    Returns:
        JSON string representation
    """
    try:
        return json.dumps(obj, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(obj)


def get_attributes(
    attrs: Dict[str, Any], span_name: str, span_id: int
) -> Iterator[Tuple[str, Any]]:
    """
    Extract OpenInference attributes from Microsoft Agent Framework GenAI attributes.

    Args:
        attrs: Original span attributes with GenAI semantic conventions
        span_name: The span name
        span_id: The span context ID

    Returns:
        Iterator of (key, value) pairs for OpenInference attributes
    """
    # Determine span kind first as it affects other attribute extraction
    span_kind = _determine_span_kind(span_name, attrs)
    yield "openinference.span.kind", span_kind

    # Extract graph node attributes for visualization hierarchy
    yield from _extract_graph_node_attributes(span_id, attrs, span_kind)

    # Extract model and provider info
    yield from _extract_model_info(attrs)

    # Extract messages (input and output)
    input_messages, output_messages = _extract_messages(attrs)

    # Extract token usage
    yield from _extract_token_usage(attrs)

    # Extract span-kind specific attributes
    if span_kind in ["LLM", "AGENT"]:
        yield from _extract_llm_agent_attributes(attrs, input_messages, output_messages, span_kind)
    elif span_kind == "TOOL":
        yield from _extract_tool_attributes(attrs)
    elif span_kind == "CHAIN":
        yield from _extract_chain_attributes(attrs, input_messages, output_messages)

    # Extract session and invocation parameters
    yield from _extract_session_info(attrs)
    yield from _extract_invocation_parameters(attrs)

    # Add remaining attributes as metadata
    yield from _extract_metadata(attrs)


def _determine_span_kind(span_name: str, attrs: Dict[str, Any]) -> str:
    """Determine the OpenInference span kind based on MS Agent Framework operation."""
    operation = attrs.get(OPERATION, "")

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


def _extract_graph_node_attributes(
    span_id: int, attrs: Dict[str, Any], span_kind: str
) -> Iterator[Tuple[str, Any]]:
    """Set graph node attributes for visualization hierarchy."""
    if span_kind == "AGENT":
        agent_id = attrs.get(AGENT_ID, span_id)
        yield "graph.node.id", f"agent_{agent_id}"
        if agent_name := attrs.get(AGENT_NAME):
            yield "graph.node.name", agent_name

    elif span_kind == "LLM":
        yield "graph.node.id", f"llm_{span_id}"
        if agent_id := attrs.get(AGENT_ID):
            yield "graph.node.parent_id", f"agent_{agent_id}"

    elif span_kind == "TOOL":
        tool_name = attrs.get(TOOL_NAME, "unknown_tool")
        yield "graph.node.id", f"tool_{tool_name}_{span_id}"
        yield "graph.node.name", tool_name

    elif span_kind == "CHAIN":
        if workflow_id := attrs.get(WORKFLOW_ID):
            yield "graph.node.id", f"workflow_{workflow_id}"
            if workflow_name := attrs.get(WORKFLOW_NAME):
                yield "graph.node.name", workflow_name
        elif executor_id := attrs.get(EXECUTOR_ID):
            yield "graph.node.id", f"executor_{executor_id}"
            if executor_type := attrs.get(EXECUTOR_TYPE):
                yield "graph.node.name", executor_type
        elif edge_group_id := attrs.get(EDGE_GROUP_ID):
            yield "graph.node.id", f"edge_group_{edge_group_id}"
            if edge_group_type := attrs.get(EDGE_GROUP_TYPE):
                yield "graph.node.name", edge_group_type
        else:
            yield "graph.node.id", f"chain_{span_id}"


def _extract_model_info(attrs: Dict[str, Any]) -> Iterator[Tuple[str, Any]]:
    """Map model and provider information to OpenInference format."""
    model_name = attrs.get(LLM_REQUEST_MODEL) or attrs.get(LLM_RESPONSE_MODEL)
    if model_name:
        yield "llm.model_name", model_name

    if provider := attrs.get(PROVIDER_NAME):
        yield "llm.provider", provider


def _extract_messages(attrs: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Extract input and output messages from attributes."""
    input_messages: List[Dict[str, Any]] = []
    output_messages: List[Dict[str, Any]] = []

    if input_msgs_raw := attrs.get(INPUT_MESSAGES):
        input_messages = _parse_messages(input_msgs_raw)
    if output_msgs_raw := attrs.get(OUTPUT_MESSAGES):
        output_messages = _parse_messages(output_msgs_raw)

    return input_messages, output_messages


def _parse_messages(messages_raw: Any) -> List[Dict[str, Any]]:
    """Parse messages from MS Agent Framework format."""
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

            parsed_msg = _parse_single_message(msg)
            if parsed_msg:
                messages.append(parsed_msg)

    except (json.JSONDecodeError, TypeError) as e:
        logger.debug(f"Failed to parse messages: {e}")

    return messages


def _parse_single_message(msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Parse a single message from MS Agent Framework format to OpenInference format."""
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


def _extract_token_usage(attrs: Dict[str, Any]) -> Iterator[Tuple[str, Any]]:
    """Map token usage from GenAI format to OpenInference format."""
    input_tokens = attrs.get(INPUT_TOKENS)
    if input_tokens is not None:
        yield "llm.token_count.prompt", input_tokens
    output_tokens = attrs.get(OUTPUT_TOKENS)
    if output_tokens is not None:
        yield "llm.token_count.completion", output_tokens
    if input_tokens is not None and output_tokens is not None:
        yield "llm.token_count.total", input_tokens + output_tokens


def _extract_llm_agent_attributes(
    attrs: Dict[str, Any],
    input_messages: List[Dict[str, Any]],
    output_messages: List[Dict[str, Any]],
    span_kind: str,
) -> Iterator[Tuple[str, Any]]:
    """Handle LLM and AGENT span attributes."""
    if input_messages:
        yield "llm.input_messages", safe_json_dumps(input_messages)
        yield from _flatten_messages(input_messages, "llm.input_messages")

    if output_messages:
        yield "llm.output_messages", safe_json_dumps(output_messages)
        yield from _flatten_messages(output_messages, "llm.output_messages")

    yield from _create_input_output_values(attrs, input_messages, output_messages, span_kind)

    if attrs.get(SYSTEM_INSTRUCTIONS) or attrs.get(AGENT_NAME):
        yield "llm.system", "microsoft.agent_framework"

    if tool_defs := attrs.get(TOOL_DEFINITIONS):
        yield from _map_tools(tool_defs)


def _flatten_messages(messages: List[Dict[str, Any]], key_prefix: str) -> Iterator[Tuple[str, Any]]:
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
                            yield tool_dotted_key, _serialize_value(tool_val)
            else:
                yield dotted_key, _serialize_value(value)


def _create_input_output_values(
    attrs: Dict[str, Any],
    input_messages: List[Dict[str, Any]],
    output_messages: List[Dict[str, Any]],
    span_kind: str,
) -> Iterator[Tuple[str, Any]]:
    """Create input.value and output.value attributes."""
    model_name = attrs.get(LLM_REQUEST_MODEL, "unknown")

    if span_kind in ["LLM", "AGENT"]:
        if input_messages:
            if len(input_messages) == 1 and input_messages[0].get("message.role") == "user":
                yield "input.value", input_messages[0].get("message.content", "")
                yield "input.mime_type", "text/plain"
            else:
                input_structure = {"messages": input_messages, "model": model_name}
                yield "input.value", safe_json_dumps(input_structure)
                yield "input.mime_type", "application/json"

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

                # Get token counts if available
                completion_tokens = attrs.get(OUTPUT_TOKENS)
                prompt_tokens = attrs.get(INPUT_TOKENS)
                total_tokens = None
                if completion_tokens is not None and prompt_tokens is not None:
                    total_tokens = completion_tokens + prompt_tokens

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
                        "completion_tokens": completion_tokens,
                        "prompt_tokens": prompt_tokens,
                        "total_tokens": total_tokens,
                    },
                }
                yield "output.value", safe_json_dumps(output_structure)
                yield "output.mime_type", "application/json"
            else:
                yield "output.value", content
                yield "output.mime_type", "text/plain"


def _extract_tool_attributes(attrs: Dict[str, Any]) -> Iterator[Tuple[str, Any]]:
    """Handle TOOL span attributes."""
    if tool_name := attrs.get(TOOL_NAME):
        yield "tool.name", tool_name
    if tool_call_id := attrs.get(TOOL_CALL_ID):
        yield "tool.call_id", tool_call_id
    if tool_desc := attrs.get(TOOL_DESCRIPTION):
        yield "tool.description", tool_desc

    if tool_args := attrs.get(TOOL_ARGUMENTS):
        if isinstance(tool_args, str):
            yield "tool.parameters", tool_args
            yield "input.value", tool_args
        else:
            yield "tool.parameters", safe_json_dumps(tool_args)
            yield "input.value", safe_json_dumps(tool_args)
        yield "input.mime_type", "application/json"

    if tool_result := attrs.get(TOOL_RESULT):
        if isinstance(tool_result, str):
            yield "output.value", tool_result
        else:
            yield "output.value", safe_json_dumps(tool_result)
        yield "output.mime_type", "text/plain"


def _extract_chain_attributes(
    attrs: Dict[str, Any],
    input_messages: List[Dict[str, Any]],
    output_messages: List[Dict[str, Any]],
) -> Iterator[Tuple[str, Any]]:
    """Handle CHAIN span attributes (workflows, executors)."""
    if input_messages:
        for msg in input_messages:
            if msg.get("message.role") == "user":
                if content := msg.get("message.content"):
                    yield "input.value", content
                    yield "input.mime_type", "text/plain"
                    break

    if output_messages:
        for msg in reversed(output_messages):
            if msg.get("message.role") == "assistant":
                if content := msg.get("message.content"):
                    yield "output.value", content
                    yield "output.mime_type", "text/plain"
                    break


def _extract_session_info(attrs: Dict[str, Any]) -> Iterator[Tuple[str, Any]]:
    """Map session and conversation info."""
    if conversation_id := attrs.get(CONVERSATION_ID):
        yield "session.id", conversation_id


def _extract_invocation_parameters(attrs: Dict[str, Any]) -> Iterator[Tuple[str, Any]]:
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
        yield "llm.invocation_parameters", safe_json_dumps(params)


def _map_tools(tools_data: Any) -> Iterator[Tuple[str, Any]]:
    """Map tool definitions to OpenInference format."""
    try:
        if isinstance(tools_data, str):
            tools_data = json.loads(tools_data)

        if not isinstance(tools_data, list):
            return

        for idx, tool in enumerate(tools_data):
            if isinstance(tool, dict):
                if name := tool.get("name"):
                    yield f"llm.tools.{idx}.tool.name", name
                if desc := tool.get("description"):
                    yield f"llm.tools.{idx}.tool.description", desc
                # Handle function schema
                if "function" in tool:
                    func = tool["function"]
                    if isinstance(func, dict):
                        if fname := func.get("name"):
                            yield f"llm.tools.{idx}.tool.name", fname
                        if fdesc := func.get("description"):
                            yield f"llm.tools.{idx}.tool.description", fdesc
                        if params := func.get("parameters"):
                            yield f"llm.tools.{idx}.tool.json_schema", safe_json_dumps(params)
                elif params := tool.get("parameters"):
                    yield f"llm.tools.{idx}.tool.json_schema", safe_json_dumps(params)
                elif input_schema := tool.get("input_schema"):
                    yield f"llm.tools.{idx}.tool.json_schema", safe_json_dumps(input_schema)

    except (json.JSONDecodeError, TypeError) as e:
        logger.debug(f"Failed to parse tools: {e}")


def _extract_metadata(attrs: Dict[str, Any]) -> Iterator[Tuple[str, Any]]:
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
        if key not in skip_keys:
            metadata[key] = _serialize_value(value)

    if metadata:
        yield "metadata", safe_json_dumps(metadata)


def _serialize_value(value: Any) -> Any:
    """Serialize a value for span attributes."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return safe_json_dumps(value)

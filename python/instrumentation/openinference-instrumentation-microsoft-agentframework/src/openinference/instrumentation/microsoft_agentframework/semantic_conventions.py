"""Semantic conventions for Microsoft Agent Framework telemetry.

This module defines attribute constants used by Microsoft Agent Framework's
OpenTelemetry instrumentation. These constants are based on the OtelAttr enum
defined in agent_framework.observability.

Reference: agent_framework/observability.py OtelAttr enum
"""

import json
from typing import Any

# Operation names
OPERATION = "gen_ai.operation.name"
CHAT_COMPLETION_OPERATION = "chat"
TOOL_EXECUTION_OPERATION = "execute_tool"
AGENT_CREATE_OPERATION = "create_agent"
AGENT_INVOKE_OPERATION = "invoke_agent"

# Provider and system
PROVIDER_NAME = "gen_ai.provider.name"
AGENT_FRAMEWORK_GEN_AI_SYSTEM = "microsoft.agent_framework"

# Request attributes
LLM_REQUEST_MODEL = "gen_ai.request.model"
LLM_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
LLM_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
LLM_REQUEST_TOP_P = "gen_ai.request.top_p"
SEED = "gen_ai.request.seed"
FREQUENCY_PENALTY = "gen_ai.request.frequency_penalty"
PRESENCE_PENALTY = "gen_ai.request.presence_penalty"
STOP_SEQUENCES = "gen_ai.request.stop_sequences"
TOP_K = "gen_ai.request.top_k"
CHOICE_COUNT = "gen_ai.request.choice.count"
ENCODING_FORMATS = "gen_ai.request.encoding_formats"

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
TOOL_TYPE = "gen_ai.tool.type"
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
WORKFLOW_DESCRIPTION = "workflow.description"
WORKFLOW_DEFINITION = "workflow.definition"
WORKFLOW_BUILD_SPAN = "workflow.build"
WORKFLOW_RUN_SPAN = "workflow.run"

# Executor attributes
EXECUTOR_ID = "executor.id"
EXECUTOR_TYPE = "executor.type"
EXECUTOR_PROCESS_SPAN = "executor.process"

# Edge group attributes
EDGE_GROUP_TYPE = "edge_group.type"
EDGE_GROUP_ID = "edge_group.id"

# Server attributes
ADDRESS = "server.address"
PORT = "server.port"

# Error attributes
ERROR_TYPE = "error.type"

# Event names (used in span events)
SYSTEM_MESSAGE_EVENT = "gen_ai.system.message"
USER_MESSAGE_EVENT = "gen_ai.user.message"
ASSISTANT_MESSAGE_EVENT = "gen_ai.assistant.message"
TOOL_MESSAGE_EVENT = "gen_ai.tool.message"
CHOICE_EVENT = "gen_ai.choice"

# LLM system attribute from semconv_ai
LLM_SYSTEM = "gen_ai.system"


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

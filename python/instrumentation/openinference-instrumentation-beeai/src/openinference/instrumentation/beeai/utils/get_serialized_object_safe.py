# Copyright 2025 IBM Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from typing import Any, Dict, List, cast

from beeai_framework.agents.base import BaseAgent
from beeai_framework.agents.react.events import (
    ReActAgentErrorEvent,
    ReActAgentRetryEvent,
    ReActAgentStartEvent,
    ReActAgentSuccessEvent,
    ReActAgentUpdateEvent,
)
from beeai_framework.backend.chat import (
    ChatModel,
)
from beeai_framework.backend.events import (
    ChatModelErrorEvent,
    ChatModelStartEvent,
    ChatModelSuccessEvent,
)
from beeai_framework.backend.types import ChatModelUsage
from beeai_framework.emitter import EventMeta
from beeai_framework.errors import FrameworkError
from beeai_framework.memory.base_memory import BaseMemory
from beeai_framework.tools.events import (
    ToolErrorEvent,
    ToolRetryEvent,
    ToolStartEvent,
    ToolSuccessEvent,
)
from beeai_framework.tools.tool import AnyTool, Tool
from pydantic import BaseModel, InstanceOf

from openinference.semconv.trace import (
    MessageAttributes,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)


def parse_llm_input_messages(messages: List[Any]) -> Dict[str, str]:
    result = {}
    for idx, message in enumerate(messages):
        result[f"{SpanAttributes.LLM_INPUT_MESSAGES}.{idx}.{MessageAttributes.MESSAGE_ROLE}"] = (
            message.role.value
        )
        text_content = "".join(
            part.text
            for part in getattr(message, "content", [])
            if getattr(part, "type", "") == "text"
        )
        result[f"{SpanAttributes.LLM_INPUT_MESSAGES}.{idx}.{MessageAttributes.MESSAGE_CONTENT}"] = (
            text_content
        )
    return result


def parse_llm_output_messages(messages: List[Any]) -> Dict[str, str]:
    result = {}
    for idx, message in enumerate(messages):
        result[f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.{idx}.{MessageAttributes.MESSAGE_ROLE}"] = (
            message.role.value
        )
        text_content = "".join(
            part.text
            for part in getattr(message, "content", [])
            if getattr(part, "type", "") == "text"
        )
        result[
            f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.{idx}.{MessageAttributes.MESSAGE_CONTENT}"
        ] = text_content
    return result


def get_serialized_object_safe(data_object: Any, meta: EventMeta) -> Any:
    try:
        # agent events
        if (
            meta.name in {"start", "success", "error", "retry"}
            and hasattr(meta, "creator")
            and isinstance(meta.creator, BaseAgent)
        ):
            agent_event_typed_data = cast(
                ReActAgentStartEvent
                | ReActAgentRetryEvent
                | ReActAgentErrorEvent
                | ReActAgentSuccessEvent,
                data_object,
            )

            output = {
                SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.AGENT.value,
                "iteration": agent_event_typed_data.meta.iteration
                if hasattr(agent_event_typed_data, "meta")
                else None,
            }

            if hasattr(agent_event_typed_data, "tools"):
                typed_tools = cast(list[InstanceOf[AnyTool]], agent_event_typed_data.tools)
                output[SpanAttributes.LLM_TOOLS] = json.dumps(
                    [
                        {
                            SpanAttributes.TOOL_NAME: tool.name,
                            SpanAttributes.TOOL_DESCRIPTION: tool.description,
                            "tool.options": json.dumps(tool.options),
                        }
                        for tool in typed_tools
                    ]
                )

            if hasattr(agent_event_typed_data, "memory") and hasattr(
                agent_event_typed_data.memory, "messages"
            ):
                typed_memory = cast(InstanceOf[BaseMemory], agent_event_typed_data.memory)
                output[SpanAttributes.INPUT_MIME_TYPE] = OpenInferenceMimeTypeValues.JSON.value
                output[SpanAttributes.INPUT_VALUE] = json.dumps(
                    [
                        {
                            "text": m.text,
                            "role": m.role.value if hasattr(m.role, "value") else m.role,
                        }
                        for m in typed_memory.messages
                    ]
                )

            if hasattr(agent_event_typed_data, "error"):
                typed_error = cast(InstanceOf[FrameworkError], agent_event_typed_data.error)

                output["exception.message"] = typed_error.message
                output["exception.stacktrace"] = getattr(typed_error, "stack", "")
                output["exception.type"] = typed_error.__class__.__name__

            if hasattr(agent_event_typed_data, "data") and agent_event_typed_data.data is not None:
                output[SpanAttributes.OUTPUT_MIME_TYPE] = OpenInferenceMimeTypeValues.JSON.value
                output[SpanAttributes.OUTPUT_VALUE] = json.dumps(
                    agent_event_typed_data.data.to_plain()
                )

            return output

        ## update events
        if meta.name in {"partial_update", "update"}:
            update_event_typed_data = cast(ReActAgentUpdateEvent, data_object)

            if isinstance(data_object.data, dict) and not data_object.data:
                return

            output = {
                SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.AGENT.value,
            }

            if hasattr(update_event_typed_data, "data") and isinstance(
                update_event_typed_data.data, dict
            ):
                output[SpanAttributes.OUTPUT_VALUE] = update_event_typed_data.data.get(
                    "final_answer"
                ) or update_event_typed_data.data.get("tool_output")
                output[SpanAttributes.OUTPUT_MIME_TYPE] = OpenInferenceMimeTypeValues.JSON.value
                output["thought"] = update_event_typed_data.data.get("thought")

                if update_event_typed_data.data.get("tool_name"):
                    output[SpanAttributes.TOOL_NAME] = update_event_typed_data.data["tool_name"]

                if update_event_typed_data.data.get("tool_input"):
                    output[SpanAttributes.TOOL_PARAMETERS] = json.dumps(
                        update_event_typed_data.data["tool_input"]
                    )
            elif hasattr(update_event_typed_data, "data") and not isinstance(
                update_event_typed_data.data, dict
            ):
                output[SpanAttributes.OUTPUT_VALUE] = (
                    update_event_typed_data.data.final_answer
                    or update_event_typed_data.data.tool_output
                )
                output[SpanAttributes.OUTPUT_MIME_TYPE] = OpenInferenceMimeTypeValues.JSON.value
                output["thought"] = update_event_typed_data.data.thought

                if hasattr(update_event_typed_data.data, "tool_name"):
                    output[SpanAttributes.TOOL_NAME] = update_event_typed_data.data.tool_name

                if hasattr(update_event_typed_data.data, "tool_input"):
                    output[SpanAttributes.TOOL_PARAMETERS] = json.dumps(
                        update_event_typed_data.data.tool_input
                    )

            return output

        ## Tool events
        if meta.name in {"start", "success", "error", "retry", "finish"} and isinstance(
            meta.creator, Tool
        ):
            tool_event_typed_data = cast(
                ToolSuccessEvent | ToolErrorEvent | ToolRetryEvent | ToolStartEvent, data_object
            )

            output = {
                SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.TOOL.value,
            }

            if hasattr(tool_event_typed_data, "error"):
                typed_error = cast(InstanceOf[FrameworkError], tool_event_typed_data.error)

                output["exception.message"] = typed_error.message
                output["exception.stacktrace"] = getattr(typed_error, "stack", "")
                output["exception.type"] = typed_error.__class__.__name__

            if hasattr(tool_event_typed_data, "output"):
                output[SpanAttributes.OUTPUT_VALUE] = str(tool_event_typed_data.output)

            if hasattr(tool_event_typed_data, "input"):
                output["tool.parameters"] = (
                    tool_event_typed_data.input.model_dump_json()
                    if isinstance(tool_event_typed_data.input, BaseModel)
                    else json.dumps(tool_event_typed_data.input)
                )

            return output

        ## llm events
        if meta.name in {"start", "success", "finish"}:
            llm_event_typed_data = cast(
                ChatModelStartEvent | ChatModelErrorEvent | ChatModelSuccessEvent, data_object
            )
            creator = cast(ChatModel, meta.creator)
            output = {
                SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.LLM.value,
            }

            if (
                hasattr(llm_event_typed_data, "value")
                and getattr(llm_event_typed_data.value, "usage", None) is not None
                and isinstance(llm_event_typed_data.value.usage, ChatModelUsage)
            ):
                usage = llm_event_typed_data.value.usage

                if usage.completion_tokens is not None:
                    output[SpanAttributes.LLM_TOKEN_COUNT_COMPLETION] = usage.completion_tokens

                if usage.prompt_tokens is not None:
                    output[SpanAttributes.LLM_TOKEN_COUNT_PROMPT] = usage.prompt_tokens

                if usage.total_tokens is not None:
                    output[SpanAttributes.LLM_TOKEN_COUNT_TOTAL] = usage.total_tokens

            if hasattr(llm_event_typed_data, "input") and hasattr(
                llm_event_typed_data.input, "messages"
            ):
                output[SpanAttributes.INPUT_MIME_TYPE] = OpenInferenceMimeTypeValues.JSON.value
                output[SpanAttributes.INPUT_VALUE] = json.dumps(
                    [msg.to_plain() for msg in llm_event_typed_data.input.messages]
                )
                output.update(parse_llm_input_messages(llm_event_typed_data.input.messages))

            if hasattr(llm_event_typed_data, "value") and hasattr(
                llm_event_typed_data.value, "messages"
            ):
                output[SpanAttributes.OUTPUT_MIME_TYPE] = OpenInferenceMimeTypeValues.JSON.value
                output[SpanAttributes.OUTPUT_VALUE] = json.dumps(
                    [msg.to_plain() for msg in llm_event_typed_data.value.messages]
                )
                output.update(parse_llm_output_messages(llm_event_typed_data.value.messages))

            if hasattr(llm_event_typed_data, "error"):
                output["exception.message"] = llm_event_typed_data.error.message
                output["exception.stacktrace"] = getattr(llm_event_typed_data.error, "stack", None)
                output["exception.type"] = getattr(
                    llm_event_typed_data.error, "name", type(llm_event_typed_data.error).__name__
                )

            if hasattr(creator, "provider_id"):
                output[SpanAttributes.LLM_PROVIDER] = creator.provider_id

            if hasattr(creator, "model_id"):
                output[SpanAttributes.LLM_MODEL_NAME] = creator.model_id

            if hasattr(creator, "parameters"):
                output[SpanAttributes.LLM_INVOCATION_PARAMETERS] = (
                    creator.parameters.model_dump_json()
                )

            return output
        return None
    except Exception as e:
        print("Failed to parse event data", e)
        return None

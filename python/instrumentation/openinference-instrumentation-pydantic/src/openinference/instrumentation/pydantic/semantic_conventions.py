from typing import Any, Dict, Iterator, Mapping, Tuple, Optional, List, Iterable
from enum import Enum
from opentelemetry.util.types import AttributeValue
from openinference.semconv.trace import (
    SpanAttributes,
    MessageAttributes,
    ToolCallAttributes,
    ToolAttributes,
    EmbeddingAttributes,
    OpenInferenceSpanKindValues,
)
import json
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class OTELConventions:
    EVENTS = "events"
    EVENT_NAME = "event.name"


class GenAIOperationName(Enum):
    CHAT = "chat"
    CREATE_AGENT = "create_agent"
    EMBEDDINGS = "embeddings"
    EXECUTE_TOOL = "execute_tool"
    GENERATE_CONTENT = "generate_content"
    INVOKE_AGENT = "invoke_agent"
    TEXT_COMPLETION = "text_completion"


class GenAISystem(Enum):
    ANTHROPIC = "anthropic"
    AWS_BEDROCK = "aws.bedrock"
    AZURE_AI_INFERENCE = "az.ai.inference"
    AZURE_OPENAI = "az.ai.openai"
    COHERE = "cohere"
    DEEPSEEK = "deepseek"
    GCP_GEMINI = "gcp.gemini"
    GCP_GEN_AI = "gcp.gen_ai"
    GCP_VERTEX_AI = "gcp.vertex_ai"
    GROQ = "groq"
    IBM_WATSONX_AI = "ibm.watsonx.ai"
    MISTRAL_AI = "mistral_ai"
    OPENAI = "openai"
    PERPLEXITY = "perplexity"
    XAI = "xai"
    OTHER = "_OTHER"


class GenAIOutputType(Enum):
    IMAGE = "image"
    JSON = "json"
    SPEECH = "speech"
    TEXT = "text"


class GenAIFinishReason(Enum):
    CONTENT_FILTER = "content_filter"
    ERROR = "error"
    LENGTH = "length"
    STOP = "stop"
    TOOL_CALLS = "tool_calls"


class GenAIToolType(Enum):
    FUNCTION = "function"


class GenAICommonAttributes:
    SYSTEM = "gen_ai.system"
    OPERATION_NAME = "gen_ai.operation.name"
    REQUEST_MODEL = "gen_ai.request.model"
    RESPONSE_MODEL = "gen_ai.response.model"
    USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
    USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
    SERVER_ADDRESS = "server.address"
    SERVER_PORT = "server.port"
    ERROR_TYPE = "error.type"
    CONVERSATION_ID = "gen_ai.conversation.id"
    OUTPUT_TYPE = "gen_ai.output.type"


class GenAIRequestAttributes:
    CHOICE_COUNT = "gen_ai.request.choice.count"
    FREQUENCY_PENALTY = "gen_ai.request.frequency_penalty"
    MAX_TOKENS = "gen_ai.request.max_tokens"
    PRESENCE_PENALTY = "gen_ai.request.presence_penalty"
    SEED = "gen_ai.request.seed"
    STOP_SEQUENCES = "gen_ai.request.stop_sequences"
    TEMPERATURE = "gen_ai.request.temperature"
    TOP_K = "gen_ai.request.top_k"
    TOP_P = "gen_ai.request.top_p"
    ENCODING_FORMATS = "gen_ai.request.encoding_formats"


class GenAIResponseAttributes:
    FINISH_REASONS = "gen_ai.response.finish_reasons"
    ID = "gen_ai.response.id"
    MODEL = "gen_ai.response.model"


class GenAIToolAttributes:
    NAME = "gen_ai.tool.name"
    DESCRIPTION = "gen_ai.tool.description"
    CALL_ID = "gen_ai.tool.call.id"


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
    ID = "id"  # Tool call id that this message is responding to


class GenAIChoiceBodyFields:
    INDEX = "index"
    FINISH_REASON = "finish_reason"
    MESSAGE = "message"
    TOOL_CALLS = "tool_calls"


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


class OpenInferenceAttributesExtractor:
    """Extracts OpenInference attributes from GenAI attributes."""

    def __init__(self) -> None:
        pass

    def get_attributes(
        self, gen_ai_attrs: Mapping[str, Any]
    ) -> Iterator[Tuple[str, AttributeValue]]:
        """
        Main method to extract OpenInference attributes from GenAI attributes.

        Args:
            gen_ai_attrs: Dictionary with OTEL GenAI semantic convention attributes

        Returns:
            Iterator of (key, value) pairs for OpenInference attributes
        """
        yield from self._extract_common_attributes(gen_ai_attrs)

        operation_name = gen_ai_attrs.get(GenAICommonAttributes.OPERATION_NAME)
        if operation_name:
            try:
                operation = GenAIOperationName(operation_name)
                if (
                    operation == GenAIOperationName.CHAT
                    or operation == GenAIOperationName.TEXT_COMPLETION
                ):
                    yield from self._extract_llm_attributes(gen_ai_attrs)
                elif operation == GenAIOperationName.EMBEDDINGS:
                    yield from self._extract_embedding_attributes(gen_ai_attrs)
                elif operation == GenAIOperationName.EXECUTE_TOOL:
                    yield from self._extract_tool_attributes(gen_ai_attrs)
                elif operation in (
                    GenAIOperationName.CREATE_AGENT,
                    GenAIOperationName.INVOKE_AGENT,
                ):
                    yield from self._extract_agent_attributes(gen_ai_attrs)
                elif operation == GenAIOperationName.GENERATE_CONTENT:
                    yield from self._extract_llm_attributes(gen_ai_attrs)
            except ValueError:
                pass

    def _extract_common_attributes(
        self, gen_ai_attrs: Mapping[str, Any]
    ) -> Iterator[Tuple[str, AttributeValue]]:
        """Extract attributes common to all operation types."""

        if GenAICommonAttributes.OPERATION_NAME in gen_ai_attrs:
            try:
                operation = GenAIOperationName(gen_ai_attrs[GenAICommonAttributes.OPERATION_NAME])
                span_kind = self._map_operation_to_span_kind(operation)
                yield SpanAttributes.OPENINFERENCE_SPAN_KIND, span_kind.value
            except ValueError:
                yield (
                    SpanAttributes.OPENINFERENCE_SPAN_KIND,
                    OpenInferenceSpanKindValues.UNKNOWN.value,
                )

        if GenAICommonAttributes.SYSTEM in gen_ai_attrs:
            yield SpanAttributes.LLM_SYSTEM, gen_ai_attrs[GenAICommonAttributes.SYSTEM]

        if GenAICommonAttributes.REQUEST_MODEL in gen_ai_attrs:
            yield SpanAttributes.LLM_MODEL_NAME, gen_ai_attrs[GenAICommonAttributes.REQUEST_MODEL]

        if GenAICommonAttributes.USAGE_INPUT_TOKENS in gen_ai_attrs:
            yield (
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT,
                gen_ai_attrs[GenAICommonAttributes.USAGE_INPUT_TOKENS],
            )

        if GenAICommonAttributes.USAGE_OUTPUT_TOKENS in gen_ai_attrs:
            yield (
                SpanAttributes.LLM_TOKEN_COUNT_COMPLETION,
                gen_ai_attrs[GenAICommonAttributes.USAGE_OUTPUT_TOKENS],
            )

        if (
            GenAICommonAttributes.USAGE_INPUT_TOKENS in gen_ai_attrs
            and GenAICommonAttributes.USAGE_OUTPUT_TOKENS in gen_ai_attrs
        ):
            total_tokens = (
                gen_ai_attrs[GenAICommonAttributes.USAGE_INPUT_TOKENS]
                + gen_ai_attrs[GenAICommonAttributes.USAGE_OUTPUT_TOKENS]
            )
            yield SpanAttributes.LLM_TOKEN_COUNT_TOTAL, total_tokens

        if GenAICommonAttributes.CONVERSATION_ID in gen_ai_attrs:
            yield SpanAttributes.SESSION_ID, gen_ai_attrs[GenAICommonAttributes.CONVERSATION_ID]

    def _extract_llm_attributes(
        self, gen_ai_attrs: Mapping[str, Any]
    ) -> Iterator[Tuple[str, AttributeValue]]:
        """Extract attributes specific to LLM operations (chat, text completion)."""

        invocation_params = self._extract_invocation_parameters(gen_ai_attrs)
        if invocation_params:
            yield SpanAttributes.LLM_INVOCATION_PARAMETERS, json.dumps(invocation_params)

        yield from self._extract_tools_attributes(gen_ai_attrs)

        if OTELConventions.EVENTS in gen_ai_attrs:
            events = self._parse_events(gen_ai_attrs[OTELConventions.EVENTS])
            if events:
                input_messages = self._extract_llm_input_messages(events)
                for index, message in enumerate(input_messages):
                    for key, value in message.items():
                        yield f"{SpanAttributes.LLM_INPUT_MESSAGES}.{index}.{key}", value

                input_value = self._find_llm_input_value(input_messages)
                if input_value is not None:
                    yield SpanAttributes.INPUT_VALUE, input_value

                output_messages = self._extract_llm_output_messages(events)
                for index, message in enumerate(output_messages):
                    for key, value in self._flatten_message(message).items():
                        yield f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.{index}.{key}", value

                output_value = self._find_llm_output_value(output_messages)
                if output_value is not None:
                    yield SpanAttributes.OUTPUT_VALUE, output_value

    def _flatten_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
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

    def _extract_embedding_attributes(
        self, gen_ai_attrs: Mapping[str, Any]
    ) -> Iterator[Tuple[str, AttributeValue]]:
        """Extract attributes specific to embedding operations."""

        if GenAICommonAttributes.REQUEST_MODEL in gen_ai_attrs:
            yield (
                SpanAttributes.EMBEDDING_MODEL_NAME,
                gen_ai_attrs[GenAICommonAttributes.REQUEST_MODEL],
            )

        if OTELConventions.EVENTS in gen_ai_attrs:
            events = self._parse_events(gen_ai_attrs[OTELConventions.EVENTS])
            embeddings = self._extract_embeddings_from_events(events)

            for idx, embedding in enumerate(embeddings):
                for key, value in embedding.items():
                    yield f"{SpanAttributes.EMBEDDING_EMBEDDINGS}.{idx}.{key}", value

            input_value = self._find_embedding_input_value(events)
            if input_value is not None:
                yield SpanAttributes.INPUT_VALUE, input_value

    def _extract_tool_attributes(
        self, gen_ai_attrs: Mapping[str, Any]
    ) -> Iterator[Tuple[str, AttributeValue]]:
        """Extract attributes specific to tool operations."""

        if GenAIToolAttributes.NAME in gen_ai_attrs:
            yield SpanAttributes.TOOL_NAME, gen_ai_attrs[GenAIToolAttributes.NAME]

        if GenAIToolAttributes.DESCRIPTION in gen_ai_attrs:
            yield SpanAttributes.TOOL_DESCRIPTION, gen_ai_attrs[GenAIToolAttributes.DESCRIPTION]

        if GenAIToolAttributes.CALL_ID in gen_ai_attrs:
            yield ToolCallAttributes.TOOL_CALL_ID, gen_ai_attrs[GenAIToolAttributes.CALL_ID]

        if OTELConventions.EVENTS in gen_ai_attrs:
            events = self._parse_events(gen_ai_attrs[OTELConventions.EVENTS])

            for event in events:
                if event.get(OTELConventions.EVENT_NAME) == GenAIEventNames.TOOL_MESSAGE:
                    if GenAIToolMessageBodyFields.ID in event:
                        yield ToolCallAttributes.TOOL_CALL_ID, event[GenAIToolMessageBodyFields.ID]
                    if GenAIToolMessageBodyFields.CONTENT in event:
                        yield SpanAttributes.OUTPUT_VALUE, event[GenAIToolMessageBodyFields.CONTENT]

                if event.get(OTELConventions.EVENT_NAME) == GenAIEventNames.ASSISTANT_MESSAGE:
                    if GenAIAssistantMessageBodyFields.TOOL_CALLS in event and isinstance(
                        event[GenAIAssistantMessageBodyFields.TOOL_CALLS], list
                    ):
                        for tool_call in event[GenAIAssistantMessageBodyFields.TOOL_CALLS]:
                            if GenAIToolCallFields.FUNCTION in tool_call and isinstance(
                                tool_call[GenAIToolCallFields.FUNCTION], dict
                            ):
                                if (
                                    GenAIFunctionFields.NAME
                                    in tool_call[GenAIToolCallFields.FUNCTION]
                                ):
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
                                    yield (
                                        SpanAttributes.INPUT_VALUE,
                                        tool_call[GenAIToolCallFields.FUNCTION][
                                            GenAIFunctionFields.ARGUMENTS
                                        ],
                                    )

    def _extract_agent_attributes(
        self, gen_ai_attrs: Mapping[str, Any]
    ) -> Iterator[Tuple[str, AttributeValue]]:
        """Extract attributes specific to agent operations."""
        yield from self._extract_llm_attributes(gen_ai_attrs)

    def _extract_tools_attributes(
        self, gen_ai_attrs: Mapping[str, Any]
    ) -> Iterator[Tuple[str, AttributeValue]]:
        """Extract tool definitions from model request parameters."""
        if PydanticCustomAttributes.MODEL_REQUEST_PARAMETERS not in gen_ai_attrs:
            return

        try:
            params = self._parse_json_value(
                gen_ai_attrs[PydanticCustomAttributes.MODEL_REQUEST_PARAMETERS]
            )
            if not params or not isinstance(params, dict):
                return

            tools = []
            if PydanticModelRequestParameters.TOOLS in params and isinstance(
                params[PydanticModelRequestParameters.TOOLS], list
            ):
                for tool in params[PydanticModelRequestParameters.TOOLS]:
                    if not isinstance(tool, dict):
                        continue

                    tool_info = {}
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
                        tool_info[ToolAttributes.TOOL_JSON_SCHEMA] = json.dumps(
                            tool[PydanticModelRequestParametersTool.PARAMETERS]
                        )

                    if tool_info:
                        tools.append(tool_info)

            for idx, tool in enumerate(tools):
                for key, value in tool.items():
                    yield f"{SpanAttributes.LLM_TOOLS}.{idx}.{key}", value
        except Exception as e:
            logger.debug(f"Error parsing model request parameters: {e}")

    def _extract_invocation_parameters(self, gen_ai_attrs: Mapping[str, Any]) -> Dict[str, Any]:
        """Extract invocation parameters that map to OpenInference."""
        param_keys = [
            (GenAIRequestAttributes.TEMPERATURE, "temperature"),
            (GenAIRequestAttributes.TOP_P, "top_p"),
            (GenAIRequestAttributes.TOP_K, "top_k"),
            (GenAIRequestAttributes.MAX_TOKENS, "max_tokens"),
            (GenAIRequestAttributes.FREQUENCY_PENALTY, "frequency_penalty"),
            (GenAIRequestAttributes.PRESENCE_PENALTY, "presence_penalty"),
            (GenAIRequestAttributes.STOP_SEQUENCES, "stop"),
            (GenAIRequestAttributes.SEED, "seed"),
        ]

        invocation_params = {}
        for gen_ai_key, param_name in param_keys:
            if gen_ai_key in gen_ai_attrs:
                invocation_params[param_name] = gen_ai_attrs[gen_ai_key]

        return invocation_params

    def _extract_llm_input_messages(self, events: list) -> list:
        """Extract input messages from events for LLM operations."""
        input_messages = []

        for event in events:
            if event.get(OTELConventions.EVENT_NAME) == GenAIEventNames.USER_MESSAGE:
                message = {}
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
                    message[MessageAttributes.MESSAGE_ROLE] = event[
                        GenAISystemMessageBodyFields.ROLE
                    ]

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
                        tc = self._process_tool_call(tool_call)
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

    def _process_tool_call(self, tool_call: dict) -> dict:
        """Process a tool call and extract relevant information."""
        if not isinstance(tool_call, dict):
            return {}

        tc = {}
        if GenAIToolCallFields.ID in tool_call:
            tc[ToolCallAttributes.TOOL_CALL_ID] = tool_call[GenAIToolCallFields.ID]

        if GenAIToolCallFields.FUNCTION in tool_call and isinstance(
            tool_call[GenAIToolCallFields.FUNCTION], dict
        ):
            function = tool_call[GenAIToolCallFields.FUNCTION]
            if GenAIFunctionFields.NAME in function:
                tc[ToolCallAttributes.TOOL_CALL_FUNCTION_NAME] = function[GenAIFunctionFields.NAME]

            if GenAIFunctionFields.ARGUMENTS in function:
                tc[ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON] = function[
                    GenAIFunctionFields.ARGUMENTS
                ]

        return tc

    def _extract_llm_output_messages(self, events: list) -> list:
        """Extract output messages from events for LLM operations."""
        output_messages = []

        for event in events:
            if event.get(OTELConventions.EVENT_NAME) == GenAIEventNames.CHOICE:
                choice = {}

                if GenAIChoiceBodyFields.MESSAGE in event and isinstance(
                    event[GenAIChoiceBodyFields.MESSAGE], dict
                ):
                    message = {}
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
                            tc = self._process_tool_call(tool_call)
                            if tc:
                                tool_calls.append(tc)

                        if tool_calls:
                            message[MessageAttributes.MESSAGE_TOOL_CALLS] = tool_calls

                    message.update(choice)

                    if message:
                        output_messages.append(message)

        return output_messages

    def _extract_embeddings_from_events(self, events: list) -> list:
        """Extract embeddings from events for embedding operations."""
        embeddings = []

        return embeddings

    def _find_llm_input_value(self, input_messages: list) -> Optional[str]:
        """Find input value from first user message for LLM operations."""
        for message in input_messages:
            if GenAIUserMessageBodyFields.CONTENT in message:
                return message[GenAIUserMessageBodyFields.CONTENT]

        return None

    def _find_llm_output_value(self, output_messages: list) -> Optional[str]:
        """Find output value from message in choice events."""
        for message in output_messages:
            if MessageAttributes.MESSAGE_CONTENT in message:
                return message[MessageAttributes.MESSAGE_CONTENT]
            if MessageAttributes.MESSAGE_TOOL_CALLS in message and isinstance(
                message[MessageAttributes.MESSAGE_TOOL_CALLS], list
            ):
                for tool_call in message[MessageAttributes.MESSAGE_TOOL_CALLS]:
                    if (
                        ToolCallAttributes.TOOL_CALL_FUNCTION_NAME in tool_call
                        and tool_call[ToolCallAttributes.TOOL_CALL_FUNCTION_NAME]
                        == PydanticMessageToolCallFunctionFinalResult.FINAL_RESULT
                    ):
                        if ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON in tool_call:
                            return tool_call[ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON]
        return None

    def _find_embedding_input_value(self, events: list) -> Optional[str]:
        """Find input value from embedding events for embedding operations."""
        return None

    def _map_operation_to_span_kind(
        self, operation: GenAIOperationName
    ) -> OpenInferenceSpanKindValues:
        """Map GenAIOperationName to OpenInferenceSpanKindValues."""
        mapping = {
            GenAIOperationName.CHAT: OpenInferenceSpanKindValues.LLM,
            GenAIOperationName.TEXT_COMPLETION: OpenInferenceSpanKindValues.LLM,
            GenAIOperationName.CREATE_AGENT: OpenInferenceSpanKindValues.AGENT,
            GenAIOperationName.EMBEDDINGS: OpenInferenceSpanKindValues.EMBEDDING,
            GenAIOperationName.EXECUTE_TOOL: OpenInferenceSpanKindValues.TOOL,
            GenAIOperationName.INVOKE_AGENT: OpenInferenceSpanKindValues.AGENT,
            GenAIOperationName.GENERATE_CONTENT: OpenInferenceSpanKindValues.LLM,
        }
        return mapping.get(operation, OpenInferenceSpanKindValues.UNKNOWN)

    def _parse_events(self, events_value: Any) -> list:
        """Parse events from string or list."""
        if isinstance(events_value, str):
            try:
                return json.loads(events_value)
            except json.JSONDecodeError:
                return []
        elif isinstance(events_value, list):
            return events_value
        return []

    def _parse_json_value(self, json_value: Any) -> Any:
        """Parse JSON from string or return as is if already parsed."""
        if isinstance(json_value, str):
            try:
                return json.loads(json_value)
            except json.JSONDecodeError:
                return None
        return json_value

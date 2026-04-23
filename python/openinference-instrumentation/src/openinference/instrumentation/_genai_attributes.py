from enum import Enum

from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as _otel_gen_ai


class GenAIAttributes:
    GEN_AI_AGENT_DESCRIPTION = _otel_gen_ai.GEN_AI_AGENT_DESCRIPTION
    GEN_AI_AGENT_ID = _otel_gen_ai.GEN_AI_AGENT_ID
    GEN_AI_AGENT_NAME = _otel_gen_ai.GEN_AI_AGENT_NAME
    GEN_AI_AGENT_VERSION = _otel_gen_ai.GEN_AI_AGENT_VERSION
    GEN_AI_CONVERSATION_ID = _otel_gen_ai.GEN_AI_CONVERSATION_ID
    GEN_AI_DATA_SOURCE_ID = _otel_gen_ai.GEN_AI_DATA_SOURCE_ID
    GEN_AI_EMBEDDINGS_DIMENSION_COUNT = _otel_gen_ai.GEN_AI_EMBEDDINGS_DIMENSION_COUNT
    GEN_AI_INPUT_MESSAGES = _otel_gen_ai.GEN_AI_INPUT_MESSAGES
    GEN_AI_OPERATION_NAME = _otel_gen_ai.GEN_AI_OPERATION_NAME
    GEN_AI_OUTPUT_MESSAGES = _otel_gen_ai.GEN_AI_OUTPUT_MESSAGES
    GEN_AI_OUTPUT_TYPE = _otel_gen_ai.GEN_AI_OUTPUT_TYPE
    GEN_AI_PROVIDER_NAME = _otel_gen_ai.GEN_AI_PROVIDER_NAME
    GEN_AI_REQUEST_CHOICE_COUNT = _otel_gen_ai.GEN_AI_REQUEST_CHOICE_COUNT
    GEN_AI_REQUEST_ENCODING_FORMATS = _otel_gen_ai.GEN_AI_REQUEST_ENCODING_FORMATS
    GEN_AI_REQUEST_FREQUENCY_PENALTY = _otel_gen_ai.GEN_AI_REQUEST_FREQUENCY_PENALTY
    GEN_AI_REQUEST_MAX_TOKENS = _otel_gen_ai.GEN_AI_REQUEST_MAX_TOKENS
    GEN_AI_REQUEST_MODEL = _otel_gen_ai.GEN_AI_REQUEST_MODEL
    GEN_AI_REQUEST_PRESENCE_PENALTY = _otel_gen_ai.GEN_AI_REQUEST_PRESENCE_PENALTY
    GEN_AI_REQUEST_SEED = _otel_gen_ai.GEN_AI_REQUEST_SEED
    GEN_AI_REQUEST_STOP_SEQUENCES = _otel_gen_ai.GEN_AI_REQUEST_STOP_SEQUENCES
    GEN_AI_REQUEST_TEMPERATURE = _otel_gen_ai.GEN_AI_REQUEST_TEMPERATURE
    GEN_AI_REQUEST_TOP_K = _otel_gen_ai.GEN_AI_REQUEST_TOP_K
    GEN_AI_REQUEST_TOP_P = _otel_gen_ai.GEN_AI_REQUEST_TOP_P
    GEN_AI_RESPONSE_FINISH_REASONS = _otel_gen_ai.GEN_AI_RESPONSE_FINISH_REASONS
    GEN_AI_RESPONSE_ID = _otel_gen_ai.GEN_AI_RESPONSE_ID
    GEN_AI_RESPONSE_MODEL = _otel_gen_ai.GEN_AI_RESPONSE_MODEL
    GEN_AI_RETRIEVAL_DOCUMENTS = _otel_gen_ai.GEN_AI_RETRIEVAL_DOCUMENTS
    GEN_AI_RETRIEVAL_QUERY_TEXT = _otel_gen_ai.GEN_AI_RETRIEVAL_QUERY_TEXT
    GEN_AI_SYSTEM = _otel_gen_ai.GEN_AI_SYSTEM
    GEN_AI_SYSTEM_INSTRUCTIONS = _otel_gen_ai.GEN_AI_SYSTEM_INSTRUCTIONS
    GEN_AI_TOOL_CALL_ARGUMENTS = _otel_gen_ai.GEN_AI_TOOL_CALL_ARGUMENTS
    GEN_AI_TOOL_CALL_ID = _otel_gen_ai.GEN_AI_TOOL_CALL_ID
    GEN_AI_TOOL_CALL_RESULT = _otel_gen_ai.GEN_AI_TOOL_CALL_RESULT
    GEN_AI_TOOL_DEFINITIONS = _otel_gen_ai.GEN_AI_TOOL_DEFINITIONS
    GEN_AI_TOOL_DESCRIPTION = _otel_gen_ai.GEN_AI_TOOL_DESCRIPTION
    GEN_AI_TOOL_NAME = _otel_gen_ai.GEN_AI_TOOL_NAME
    GEN_AI_TOOL_TYPE = _otel_gen_ai.GEN_AI_TOOL_TYPE
    GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS = _otel_gen_ai.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS
    GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS = _otel_gen_ai.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS
    GEN_AI_USAGE_INPUT_TOKENS = _otel_gen_ai.GEN_AI_USAGE_INPUT_TOKENS
    GEN_AI_USAGE_OUTPUT_TOKENS = _otel_gen_ai.GEN_AI_USAGE_OUTPUT_TOKENS
    # Not yet in opentelemetry.semconv._incubating.attributes.gen_ai_attributes.
    GEN_AI_REQUEST_STREAM = "gen_ai.request.stream"


GenAIOperationNameValues = _otel_gen_ai.GenAiOperationNameValues
GenAIProviderNameValues = _otel_gen_ai.GenAiProviderNameValues
GenAIOutputTypeValues = _otel_gen_ai.GenAiOutputTypeValues


class GenAIToolTypeValues(Enum):
    DATASTORE = "datastore"
    EXTENSION = "extension"
    FUNCTION = "function"


class GenAIRoleValues(Enum):
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"
    USER = "user"


class GenAIMessagePartTypeValues(Enum):
    BLOB = "blob"
    FILE = "file"
    REASONING = "reasoning"
    SERVER_TOOL_CALL = "server_tool_call"
    SERVER_TOOL_CALL_RESPONSE = "server_tool_call_response"
    TEXT = "text"
    TOOL_CALL = "tool_call"
    TOOL_CALL_RESPONSE = "tool_call_response"
    URI = "uri"


class GenAIModalityValues(Enum):
    AUDIO = "audio"
    IMAGE = "image"
    VIDEO = "video"

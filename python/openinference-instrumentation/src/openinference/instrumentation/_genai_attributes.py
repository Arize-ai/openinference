from enum import Enum


class GenAIAttributes:
    GEN_AI_AGENT_DESCRIPTION = "gen_ai.agent.description"
    GEN_AI_AGENT_ID = "gen_ai.agent.id"
    GEN_AI_AGENT_NAME = "gen_ai.agent.name"
    GEN_AI_AGENT_VERSION = "gen_ai.agent.version"
    GEN_AI_CONVERSATION_ID = "gen_ai.conversation.id"
    GEN_AI_DATA_SOURCE_ID = "gen_ai.data_source.id"
    GEN_AI_EMBEDDINGS_DIMENSION_COUNT = "gen_ai.embeddings.dimension.count"
    GEN_AI_INPUT_MESSAGES = "gen_ai.input.messages"
    GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
    GEN_AI_OUTPUT_MESSAGES = "gen_ai.output.messages"
    GEN_AI_OUTPUT_TYPE = "gen_ai.output.type"
    GEN_AI_PROVIDER_NAME = "gen_ai.provider.name"
    GEN_AI_REQUEST_CHOICE_COUNT = "gen_ai.request.choice.count"
    GEN_AI_REQUEST_ENCODING_FORMATS = "gen_ai.request.encoding_formats"
    GEN_AI_REQUEST_FREQUENCY_PENALTY = "gen_ai.request.frequency_penalty"
    GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
    GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
    GEN_AI_REQUEST_PRESENCE_PENALTY = "gen_ai.request.presence_penalty"
    GEN_AI_REQUEST_SEED = "gen_ai.request.seed"
    GEN_AI_REQUEST_STOP_SEQUENCES = "gen_ai.request.stop_sequences"
    GEN_AI_REQUEST_STREAM = "gen_ai.request.stream"
    GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
    GEN_AI_REQUEST_TOP_K = "gen_ai.request.top_k"
    GEN_AI_REQUEST_TOP_P = "gen_ai.request.top_p"
    GEN_AI_RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"
    GEN_AI_RESPONSE_ID = "gen_ai.response.id"
    GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
    GEN_AI_RETRIEVAL_DOCUMENTS = "gen_ai.retrieval.documents"
    GEN_AI_RETRIEVAL_QUERY_TEXT = "gen_ai.retrieval.query.text"
    GEN_AI_SYSTEM = "gen_ai.system"
    GEN_AI_SYSTEM_INSTRUCTIONS = "gen_ai.system_instructions"
    GEN_AI_TOOL_CALL_ARGUMENTS = "gen_ai.tool.call.arguments"
    GEN_AI_TOOL_CALL_ID = "gen_ai.tool.call.id"
    GEN_AI_TOOL_CALL_RESULT = "gen_ai.tool.call.result"
    GEN_AI_TOOL_DEFINITIONS = "gen_ai.tool.definitions"
    GEN_AI_TOOL_DESCRIPTION = "gen_ai.tool.description"
    GEN_AI_TOOL_NAME = "gen_ai.tool.name"
    GEN_AI_TOOL_TYPE = "gen_ai.tool.type"
    GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS = "gen_ai.usage.cache_creation.input_tokens"
    GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS = "gen_ai.usage.cache_read.input_tokens"
    GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
    GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"


class GenAIOperationNameValues(Enum):
    CHAT = "chat"
    CREATE_AGENT = "create_agent"
    EMBEDDINGS = "embeddings"
    EXECUTE_TOOL = "execute_tool"
    GENERATE_CONTENT = "generate_content"
    INVOKE_AGENT = "invoke_agent"
    RETRIEVAL = "retrieval"
    TEXT_COMPLETION = "text_completion"


class GenAIProviderNameValues(Enum):
    ANTHROPIC = "anthropic"
    AWS_BEDROCK = "aws.bedrock"
    AZURE_AI_INFERENCE = "azure.ai.inference"
    AZURE_AI_OPENAI = "azure.ai.openai"
    COHERE = "cohere"
    DEEPSEEK = "deepseek"
    GCP_GEN_AI = "gcp.gen_ai"
    GCP_GEMINI = "gcp.gemini"
    GCP_VERTEX_AI = "gcp.vertex_ai"
    GROQ = "groq"
    IBM_WATSONX_AI = "ibm.watsonx.ai"
    MISTRAL_AI = "mistral_ai"
    OPENAI = "openai"
    PERPLEXITY = "perplexity"
    X_AI = "x_ai"


class GenAIOutputTypeValues(Enum):
    IMAGE = "image"
    JSON = "json"
    SPEECH = "speech"
    TEXT = "text"


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

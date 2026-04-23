# ruff: noqa: E501
"""
Enumerated values defined by the OpenTelemetry GenAI semantic conventions.
"""

from enum import Enum


class GenAIOperationNameValues(Enum):
    """Values for gen_ai.operation.name."""

    CHAT = "chat"
    TEXT_COMPLETION = "text_completion"
    GENERATE_CONTENT = "generate_content"
    EMBEDDINGS = "embeddings"
    RETRIEVAL = "retrieval"
    EXECUTE_TOOL = "execute_tool"
    CREATE_AGENT = "create_agent"
    INVOKE_AGENT = "invoke_agent"
    INVOKE_WORKFLOW = "invoke_workflow"


class GenAIProviderNameValues(Enum):
    """
    Values for gen_ai.provider.name. GenAI uses composite names for
    cloud-hosted variants of providers (e.g. "azure.ai.openai").
    """

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    COHERE = "cohere"
    DEEPSEEK = "deepseek"
    GROQ = "groq"
    MISTRAL_AI = "mistral_ai"
    PERPLEXITY = "perplexity"
    X_AI = "x_ai"
    AZURE_AI_OPENAI = "azure.ai.openai"
    AZURE_AI_INFERENCE = "azure.ai.inference"
    AWS_BEDROCK = "aws.bedrock"
    GCP_VERTEX_AI = "gcp.vertex_ai"
    GCP_GEMINI = "gcp.gemini"
    GCP_GEN_AI = "gcp.gen_ai"
    IBM_WATSONX_AI = "ibm.watsonx.ai"


class GenAIOutputTypeValues(Enum):
    """Values for gen_ai.output.type."""

    TEXT = "text"
    JSON = "json"
    IMAGE = "image"
    SPEECH = "speech"


class GenAIToolTypeValues(Enum):
    """Values for gen_ai.tool.type."""

    FUNCTION = "function"
    EXTENSION = "extension"
    DATASTORE = "datastore"


class GenAIMessagePartTypeValues(Enum):
    """Values for message part `type` inside gen_ai.input.messages / gen_ai.output.messages."""

    TEXT = "text"
    TOOL_CALL = "tool_call"
    TOOL_CALL_RESPONSE = "tool_call_response"
    REASONING = "reasoning"
    SERVER_TOOL_CALL = "server_tool_call"
    SERVER_TOOL_CALL_RESPONSE = "server_tool_call_response"
    BLOB = "blob"
    URI = "uri"
    FILE = "file"

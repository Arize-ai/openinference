package com.arize.semconv.trace;

import lombok.experimental.UtilityClass;

public class SemanticConventions {
    @UtilityClass
    public static class SemanticAttributePrefixes {
        public static final String INPUT = "input";
        public static final String OUTPUT = "output";
        public static final String LLM = "llm";
        public static final String RETRIEVAL = "retrieval";
        public static final String RERANKER = "reranker";
        public static final String MESSAGES = "messages";
        public static final String MESSAGE = "message";
        public static final String DOCUMENT = "document";
        public static final String EMBEDDING = "embedding";
        public static final String TOOL = "tool";
        public static final String TOOL_CALL = "tool_call";
        public static final String METADATA = "metadata";
        public static final String TAG = "tag";
        public static final String SESSION = "session";
        public static final String USER = "user";
        public static final String OPENINFERENCE = "openinference";
        public static final String MESSAGE_CONTENT = "message_content";
        public static final String IMAGE = "image";
        public static final String AUDIO = "audio";
        public static final String PROMPT = "prompt";
        public static final String AGENT = "agent";
        public static final String GRAPH = "graph";
    }

    @UtilityClass
    public static class LLMAttributePostfixes {
        public static final String PROVIDER = "provider";
        public static final String SYSTEM = "system";
        public static final String MODEL_NAME = "model_name";
        public static final String TOKEN_COUNT = "token_count";
        public static final String INPUT_MESSAGES = "input_messages";
        public static final String OUTPUT_MESSAGES = "output_messages";
        public static final String INVOCATION_PARAMETERS = "invocation_parameters";
        public static final String PROMPTS = "prompts";
        public static final String PROMPT_TEMPLATE = "prompt_template";
        public static final String FUNCTION_CALL = "function_call";
        public static final String TOOLS = "tools";
        public static final String COST = "cost";
    }

    @UtilityClass
    public static class LLMPromptTemplateAttributePostfixes {
        public static final String VARIABLES = "variables";
        public static final String TEMPLATE = "template";
    }

    @UtilityClass
    public static class RetrievalAttributePostfixes {
        public static final String DOCUMENTS = "documents";
    }

    @UtilityClass
    public static class RerankerAttributePostfixes {
        public static final String INPUT_DOCUMENTS = "input_documents";
        public static final String OUTPUT_DOCUMENTS = "output_documents";
        public static final String QUERY = "query";
        public static final String MODEL_NAME = "model_name";
        public static final String TOP_K = "top_k";
    }

    @UtilityClass
    public static class EmbeddingAttributePostfixes {
        public static final String EMBEDDINGS = "embeddings";
        public static final String TEXT = "text";
        public static final String MODEL_NAME = "model_name";
        public static final String VECTOR = "vector";
    }

    @UtilityClass
    public static class ToolAttributePostfixes {
        public static final String NAME = "name";
        public static final String DESCRIPTION = "description";
        public static final String PARAMETERS = "parameters";
        public static final String JSON_SCHEMA = "json_schema";
    }

    @UtilityClass
    public static class MessageAttributePostfixes {
        public static final String ROLE = "role";
        public static final String CONTENT = "content";
        public static final String CONTENTS = "contents";
        public static final String NAME = "name";
        public static final String FUNCTION_CALL_NAME = "function_call_name";
        public static final String FUNCTION_CALL_ARGUMENTS_JSON = "function_call_arguments_json";
        public static final String TOOL_CALLS = "tool_calls";
        public static final String TOOL_CALL_ID = "tool_call_id";
    }

    @UtilityClass
    public static class MessageContentsAttributePostfixes {
        public static final String TYPE = "type";
        public static final String TEXT = "text";
        public static final String IMAGE = "image";
    }

    @UtilityClass
    public static class ImageAttributesPostfixes {
        public static final String URL = "url";
    }

    @UtilityClass
    public static class ToolCallAttributePostfixes {
        public static final String FUNCTION_NAME = "function.name";
        public static final String FUNCTION_ARGUMENTS = "function.arguments";
        public static final String ID = "id";
    }

    @UtilityClass
    public static class DocumentAttributePostfixes {
        public static final String ID = "id";
        public static final String CONTENT = "content";
        public static final String SCORE = "score";
        public static final String METADATA = "metadata";
    }

    @UtilityClass
    public static class TagAttributePostfixes {
        public static final String TAGS = "tags";
    }

    @UtilityClass
    public static class SessionAttributePostfixes {
        public static final String ID = "id";
    }

    @UtilityClass
    public static class UserAttributePostfixes {
        public static final String ID = "id";
    }

    @UtilityClass
    public static class AudioAttributesPostfixes {
        public static final String URL = "url";
        public static final String MIME_TYPE = "mime_type";
        public static final String TRANSCRIPT = "transcript";
    }

    @UtilityClass
    public static class PromptAttributePostfixes {
        public static final String VENDOR = "vendor";
        public static final String ID = "id";
        public static final String URL = "url";
    }

    @UtilityClass
    public static class AgentPostfixes {
        public static final String NAME = "name";
    }

    @UtilityClass
    public static class GraphPostfixes {
        public static final String NODE_ID = "node.id";
        public static final String NODE_NAME = "node.name";
        public static final String NODE_PARENT_ID = "node.parent_id";
    }

    /**
     * The input to any span
     */
    public static final String INPUT_VALUE = SemanticAttributePrefixes.INPUT + ".value";

    public static final String INPUT_MIME_TYPE = SemanticAttributePrefixes.INPUT + ".mime_type";

    /**
     * The output of any span
     */
    public static final String OUTPUT_VALUE = SemanticAttributePrefixes.OUTPUT + ".value";

    public static final String OUTPUT_MIME_TYPE = SemanticAttributePrefixes.OUTPUT + ".mime_type";

    /**
     * The messages sent to the LLM for completions
     * Typically seen in OpenAI chat completions
     * @see <a href="https://beta.openai.com/docs/api-reference/completions/create">OpenAI API Reference</a>
     */
    public static final String LLM_INPUT_MESSAGES =
            SemanticAttributePrefixes.LLM + "." + LLMAttributePostfixes.INPUT_MESSAGES;

    /**
     * The output messages from the LLM
     */
    public static final String LLM_OUTPUT_MESSAGES =
            SemanticAttributePrefixes.LLM + "." + LLMAttributePostfixes.OUTPUT_MESSAGES;

    /**
     * The model name used for the LLM
     */
    public static final String LLM_MODEL_NAME = SemanticAttributePrefixes.LLM + "." + LLMAttributePostfixes.MODEL_NAME;

    /**
     * Document content in retrieval operations
     */
    public static final String RETRIEVAL_DOCUMENTS =
            SemanticAttributePrefixes.RETRIEVAL + "." + RetrievalAttributePostfixes.DOCUMENTS;

    /**
     * Message role (e.g., "user", "assistant", "system")
     */
    public static final String MESSAGE_ROLE = SemanticAttributePrefixes.MESSAGE + "." + MessageAttributePostfixes.ROLE;

    /**
     * Message content
     */
    public static final String MESSAGE_CONTENT =
            SemanticAttributePrefixes.MESSAGE + "." + MessageAttributePostfixes.CONTENT;

    /**
     * The prompts sent to the LLM for completions
     * Typically seen in OpenAI legacy completions
     * @see <a href="https://beta.openai.com/docs/api-reference/completions/create">OpenAI API Reference</a>
     */
    public static final String LLM_PROMPTS = SemanticAttributePrefixes.LLM + "." + LLMAttributePostfixes.PROMPTS;

    /**
     * The JSON representation of the parameters passed to the LLM
     */
    public static final String LLM_INVOCATION_PARAMETERS =
            SemanticAttributePrefixes.LLM + "." + LLMAttributePostfixes.INVOCATION_PARAMETERS;

    /**
     * The provider of the inferences. E.g. the cloud provider
     */
    public static final String LLM_PROVIDER = SemanticAttributePrefixes.LLM + "." + LLMAttributePostfixes.PROVIDER;

    /**
     * The AI product as identified by the client or server
     */
    public static final String LLM_SYSTEM = SemanticAttributePrefixes.LLM + "." + LLMAttributePostfixes.SYSTEM;

    /** Token count for the completion by the llm (in tokens) */
    public static final String LLM_TOKEN_COUNT_COMPLETION =
            SemanticAttributePrefixes.LLM + "." + LLMAttributePostfixes.TOKEN_COUNT + ".completion";

    /** Token count for the reasoning steps in the completion (in tokens) */
    public static final String LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING =
            SemanticAttributePrefixes.LLM + "." + LLMAttributePostfixes.TOKEN_COUNT + ".completion_details.reasoning";

    /** Token count for audio input generated by the model (in tokens) */
    public static final String LLM_TOKEN_COUNT_COMPLETION_DETAILS_AUDIO =
            SemanticAttributePrefixes.LLM + "." + LLMAttributePostfixes.TOKEN_COUNT + ".completion_details.audio";

    /** Token count for the prompt to the llm (in tokens) */
    public static final String LLM_TOKEN_COUNT_PROMPT =
            SemanticAttributePrefixes.LLM + "." + LLMAttributePostfixes.TOKEN_COUNT + ".prompt";

    /** Token count for the tokens written to cache (in tokens) */
    public static final String LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE =
            SemanticAttributePrefixes.LLM + "." + LLMAttributePostfixes.TOKEN_COUNT + ".prompt_details.cache_write";

    /** Token count for the tokens retrieved from cache (in tokens) */
    public static final String LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ =
            SemanticAttributePrefixes.LLM + "." + LLMAttributePostfixes.TOKEN_COUNT + ".prompt_details.cache_read";

    /** Token count for audio input presented in the prompt (in tokens) */
    public static final String LLM_TOKEN_COUNT_PROMPT_DETAILS_AUDIO =
            SemanticAttributePrefixes.LLM + "." + LLMAttributePostfixes.TOKEN_COUNT + ".prompt_details.audio";

    /** Token count for the entire transaction with the llm (in tokens) */
    public static final String LLM_TOKEN_COUNT_TOTAL =
            SemanticAttributePrefixes.LLM + "." + LLMAttributePostfixes.TOKEN_COUNT + ".total";

    /**
     * Key prefix for additional prompt token count details. Each detail should be a separate attribute
     * with this prefix, e.g. llm.token_count.prompt_details.reasoning, llm.token_count.prompt_details.audio.
     * All values should be in tokens (integer count of tokens).
     */
    public static final String LLM_TOKEN_COUNT_PROMPT_DETAILS =
            SemanticAttributePrefixes.LLM + "." + LLMAttributePostfixes.TOKEN_COUNT + ".prompt_details";

    /**
     * Key prefix for additional completion token count details. Each detail should be a separate attribute
     * with this prefix, e.g. llm.token_count.completion_details.reasoning, llm.token_count.completion_details.audio.
     * All values should be in tokens (integer count of tokens).
     */
    public static final String LLM_TOKEN_COUNT_COMPLETION_DETAILS =
            SemanticAttributePrefixes.LLM + "." + LLMAttributePostfixes.TOKEN_COUNT + ".completion_details";

    /**
     * Key prefix for cost information. When these keys are transformed into a JSON-like structure, it would look like:
     * {
     *     "prompt": 0.0021,  # Cost in USD
     *     "completion": 0.0045,  # Cost in USD
     *     "total": 0.0066,  # Cost in USD
     *     "completion_details": {
     *         "output": 0.0009,  # Cost in USD
     *         "reasoning": 0.0024,    # Cost in USD (e.g., 80 tokens * $0.03/1K tokens)
     *         "audio": 0.0012  # Cost in USD (e.g., 40 tokens * $0.03/1K tokens)
     *     },
     *     "prompt_details": {
     *         "input": 0.0003,  # Cost in USD
     *         "cache_write": 0.0006,  # Cost in USD (e.g., 20 tokens * $0.03/1K tokens)
     *         "cache_read": 0.0003,   # Cost in USD (e.g., 10 tokens * $0.03/1K tokens)
     *         "cache_input": 0.0006,  # Cost in USD (e.g., 20 tokens * $0.03/1K tokens)
     *         "audio": 0.0003   # Cost in USD (e.g., 10 tokens * $0.03/1K tokens)
     *     }
     * }
     * Note: This is a key prefix - individual attributes are stored as separate span attributes with this prefix,
     * e.g. llm.cost.prompt, llm.cost.completion_details.reasoning, etc. The JSON structure shown above represents
     * how these separate attributes can be conceptually organized.
     * All monetary values are in USD with floating point precision.
     */
    public static final String LLM_COST = SemanticAttributePrefixes.LLM + "." + LLMAttributePostfixes.COST;

    /** Cost of the prompt tokens in USD */
    public static final String LLM_COST_PROMPT =
            SemanticAttributePrefixes.LLM + "." + LLMAttributePostfixes.COST + ".prompt";

    /** Cost of the completion tokens in USD */
    public static final String LLM_COST_COMPLETION =
            SemanticAttributePrefixes.LLM + "." + LLMAttributePostfixes.COST + ".completion";

    /** Total cost of the LLM call in USD (prompt + completion) */
    public static final String LLM_COST_TOTAL =
            SemanticAttributePrefixes.LLM + "." + LLMAttributePostfixes.COST + ".total";

    /** Total cost of input tokens in USD. This represents the cost of tokens that were used as input
     * to the model, which may be different from the prompt cost if there are additional processing steps. */
    public static final String LLM_COST_INPUT =
            SemanticAttributePrefixes.LLM + "." + LLMAttributePostfixes.COST + ".prompt_details.input";

    /** Total cost of output tokens in USD. This represents the cost of tokens that were generated as output
     * by the model, which may be different from the completion cost if there are additional processing steps. */
    public static final String LLM_COST_OUTPUT =
            SemanticAttributePrefixes.LLM + "." + LLMAttributePostfixes.COST + ".completion_details.output";

    /** Cost of reasoning steps in the completion in USD */
    public static final String LLM_COST_COMPLETION_DETAILS_REASONING =
            SemanticAttributePrefixes.LLM + "." + LLMAttributePostfixes.COST + ".completion_details.reasoning";

    /** Cost of audio tokens in the completion in USD */
    public static final String LLM_COST_COMPLETION_DETAILS_AUDIO =
            SemanticAttributePrefixes.LLM + "." + LLMAttributePostfixes.COST + ".completion_details.audio";

    /** Cost of prompt tokens written to cache in USD */
    public static final String LLM_COST_PROMPT_DETAILS_CACHE_WRITE =
            SemanticAttributePrefixes.LLM + "." + LLMAttributePostfixes.COST + ".prompt_details.cache_write";

    /** Cost of prompt tokens read from cache in USD */
    public static final String LLM_COST_PROMPT_DETAILS_CACHE_READ =
            SemanticAttributePrefixes.LLM + "." + LLMAttributePostfixes.COST + ".prompt_details.cache_read";

    /** Cost of input tokens in the prompt that were cached in USD */
    public static final String LLM_COST_PROMPT_DETAILS_CACHE_INPUT =
            SemanticAttributePrefixes.LLM + "." + LLMAttributePostfixes.COST + ".prompt_details.cache_input";

    /** Cost of audio tokens in the prompt in USD */
    public static final String LLM_COST_PROMPT_DETAILS_AUDIO =
            SemanticAttributePrefixes.LLM + "." + LLMAttributePostfixes.COST + ".prompt_details.audio";

    /**
     * The name of the message. This is only used for role 'function' where the name
     * of the function is captured in the name field and the parameters are captured in the
     * content.
     */
    public static final String MESSAGE_NAME = SemanticAttributePrefixes.MESSAGE + "." + MessageAttributePostfixes.NAME;

    /**
     * The tool calls generated by the model, such as function calls.
     */
    public static final String MESSAGE_TOOL_CALLS =
            SemanticAttributePrefixes.MESSAGE + "." + MessageAttributePostfixes.TOOL_CALLS;

    /**
     * The id of the tool call on a "tool" role message
     */
    public static final String MESSAGE_TOOL_CALL_ID =
            SemanticAttributePrefixes.MESSAGE + "." + MessageAttributePostfixes.TOOL_CALL_ID;

    /**
     * tool_call.function.name
     */
    public static final String TOOL_CALL_FUNCTION_NAME =
            SemanticAttributePrefixes.TOOL_CALL + "." + ToolCallAttributePostfixes.FUNCTION_NAME;

    /**
     * tool_call.function.argument (JSON string)
     */
    public static final String TOOL_CALL_FUNCTION_ARGUMENTS_JSON =
            SemanticAttributePrefixes.TOOL_CALL + "." + ToolCallAttributePostfixes.FUNCTION_ARGUMENTS;

    /**
     * The id of the tool call
     */
    public static final String TOOL_CALL_ID = SemanticAttributePrefixes.TOOL_CALL + "." + ToolCallAttributePostfixes.ID;

    /**
     * The LLM function call function name
     */
    public static final String MESSAGE_FUNCTION_CALL_NAME =
            SemanticAttributePrefixes.MESSAGE + "." + MessageAttributePostfixes.FUNCTION_CALL_NAME;

    /**
     * The LLM function call function arguments in a json string
     */
    public static final String MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON =
            SemanticAttributePrefixes.MESSAGE + "." + MessageAttributePostfixes.FUNCTION_CALL_ARGUMENTS_JSON;

    /**
     * The array of contents for the message sent to the LLM. Each element of the array is
     * an `message_content` object.
     */
    public static final String MESSAGE_CONTENTS =
            SemanticAttributePrefixes.MESSAGE + "." + MessageAttributePostfixes.CONTENTS;

    /**
     * The type of content sent to the LLM
     */
    public static final String MESSAGE_CONTENT_TYPE =
            SemanticAttributePrefixes.MESSAGE_CONTENT + "." + MessageContentsAttributePostfixes.TYPE;

    /**
     * The text content of the message sent to the LLM
     */
    public static final String MESSAGE_CONTENT_TEXT =
            SemanticAttributePrefixes.MESSAGE_CONTENT + "." + MessageContentsAttributePostfixes.TEXT;

    /**
     * The image content of the message sent to the LLM
     */
    public static final String MESSAGE_CONTENT_IMAGE =
            SemanticAttributePrefixes.MESSAGE_CONTENT + "." + MessageContentsAttributePostfixes.IMAGE;

    /**
     * The http or base64 link to the image
     */
    public static final String IMAGE_URL = SemanticAttributePrefixes.IMAGE + "." + ImageAttributesPostfixes.URL;

    public static final String DOCUMENT_ID = SemanticAttributePrefixes.DOCUMENT + "." + DocumentAttributePostfixes.ID;

    public static final String DOCUMENT_CONTENT =
            SemanticAttributePrefixes.DOCUMENT + "." + DocumentAttributePostfixes.CONTENT;

    public static final String DOCUMENT_SCORE =
            SemanticAttributePrefixes.DOCUMENT + "." + DocumentAttributePostfixes.SCORE;

    public static final String DOCUMENT_METADATA =
            SemanticAttributePrefixes.DOCUMENT + "." + DocumentAttributePostfixes.METADATA;

    /**
     * The text that was embedded to create the vector
     */
    public static final String EMBEDDING_TEXT =
            SemanticAttributePrefixes.EMBEDDING + "." + EmbeddingAttributePostfixes.TEXT;

    /**
     * The name of the model that was used to create the vector
     */
    public static final String EMBEDDING_MODEL_NAME =
            SemanticAttributePrefixes.EMBEDDING + "." + EmbeddingAttributePostfixes.MODEL_NAME;

    /**
     * The embedding vector. Typically a high dimensional vector of floats or ints
     */
    public static final String EMBEDDING_VECTOR =
            SemanticAttributePrefixes.EMBEDDING + "." + EmbeddingAttributePostfixes.VECTOR;

    /**
     * The embedding list root
     */
    public static final String EMBEDDING_EMBEDDINGS =
            SemanticAttributePrefixes.EMBEDDING + "." + EmbeddingAttributePostfixes.EMBEDDINGS;

    // Helper constant for prompt template prefix
    private static final String PROMPT_TEMPLATE_PREFIX =
            SemanticAttributePrefixes.LLM + "." + LLMAttributePostfixes.PROMPT_TEMPLATE;

    /**
     * The JSON representation of the variables used in the prompt template
     */
    public static final String PROMPT_TEMPLATE_VARIABLES = PROMPT_TEMPLATE_PREFIX + ".variables";

    /**
     * A prompt template
     */
    public static final String PROMPT_TEMPLATE_TEMPLATE = PROMPT_TEMPLATE_PREFIX + ".template";

    /**
     * A prompt template version
     */
    public static final String PROMPT_TEMPLATE_VERSION = PROMPT_TEMPLATE_PREFIX + ".version";

    /**
     * The JSON representation of a function call of an LLM
     */
    public static final String LLM_FUNCTION_CALL =
            SemanticAttributePrefixes.LLM + "." + LLMAttributePostfixes.FUNCTION_CALL;

    /**
     * List of tools that are advertised to the LLM to be able to call
     */
    public static final String LLM_TOOLS = SemanticAttributePrefixes.LLM + "." + LLMAttributePostfixes.TOOLS;

    /**
     * The name of a tool
     */
    public static final String TOOL_NAME = SemanticAttributePrefixes.TOOL + "." + ToolAttributePostfixes.NAME;

    /**
     * The description of a tool
     */
    public static final String TOOL_DESCRIPTION =
            SemanticAttributePrefixes.TOOL + "." + ToolAttributePostfixes.DESCRIPTION;

    /**
     * The parameters of the tool represented as a JSON string
     */
    public static final String TOOL_PARAMETERS =
            SemanticAttributePrefixes.TOOL + "." + ToolAttributePostfixes.PARAMETERS;

    /**
     * The json schema of a tool input, It is RECOMMENDED that this be in the
     * OpenAI tool calling format: https://platform.openai.com/docs/assistants/tools
     */
    public static final String TOOL_JSON_SCHEMA =
            SemanticAttributePrefixes.TOOL + "." + ToolAttributePostfixes.JSON_SCHEMA;

    /**
     * The session id of a trace. Used to correlate spans in a single session.
     */
    public static final String SESSION_ID = SemanticAttributePrefixes.SESSION + "." + SessionAttributePostfixes.ID;

    /**
     * The user id of a trace. Used to correlate spans for a single user.
     */
    public static final String USER_ID = SemanticAttributePrefixes.USER + "." + UserAttributePostfixes.ID;

    /**
     * The documents used as input to the reranker
     */
    public static final String RERANKER_INPUT_DOCUMENTS =
            SemanticAttributePrefixes.RERANKER + "." + RerankerAttributePostfixes.INPUT_DOCUMENTS;

    /**
     * The documents output by the reranker
     */
    public static final String RERANKER_OUTPUT_DOCUMENTS =
            SemanticAttributePrefixes.RERANKER + "." + RerankerAttributePostfixes.OUTPUT_DOCUMENTS;

    /**
     * The query string for the reranker
     */
    public static final String RERANKER_QUERY =
            SemanticAttributePrefixes.RERANKER + "." + RerankerAttributePostfixes.QUERY;

    /**
     * The model name for the reranker
     */
    public static final String RERANKER_MODEL_NAME =
            SemanticAttributePrefixes.RERANKER + "." + RerankerAttributePostfixes.MODEL_NAME;

    /**
     * The top k parameter for the reranker
     */
    public static final String RERANKER_TOP_K =
            SemanticAttributePrefixes.RERANKER + "." + RerankerAttributePostfixes.TOP_K;

    /**
     * Metadata for a span, used to store user-defined key-value pairs
     */
    public static final String METADATA = "metadata";

    /**
     * The tags associated with a span
     */
    public static final String TAG_TAGS = SemanticAttributePrefixes.TAG + "." + TagAttributePostfixes.TAGS;

    /**
     * The url of an audio file
     */
    public static final String AUDIO_URL = SemanticAttributePrefixes.AUDIO + "." + AudioAttributesPostfixes.URL;

    /**
     * The audio mime type
     */
    public static final String AUDIO_MIME_TYPE =
            SemanticAttributePrefixes.AUDIO + "." + AudioAttributesPostfixes.MIME_TYPE;

    /**
     * The audio transcript as text
     */
    public static final String AUDIO_TRANSCRIPT =
            SemanticAttributePrefixes.AUDIO + "." + AudioAttributesPostfixes.TRANSCRIPT;

    /**
     * The vendor or origin of the prompt, e.g. a prompt library, a specialized service, etc.
     */
    public static final String PROMPT_VENDOR = SemanticAttributePrefixes.PROMPT + "." + PromptAttributePostfixes.VENDOR;

    /**
     * A vendor-specific id used to locate the prompt
     */
    public static final String PROMPT_ID = SemanticAttributePrefixes.PROMPT + "." + PromptAttributePostfixes.ID;

    /**
     * A vendor-specific URL used to locate the prompt
     */
    public static final String PROMPT_URL = SemanticAttributePrefixes.PROMPT + "." + PromptAttributePostfixes.URL;

    /**
     * The name of the agent. Agents that perform the same functions should have the same name.
     */
    public static final String AGENT_NAME = SemanticAttributePrefixes.AGENT + "." + AgentPostfixes.NAME;

    /**
     * The id of the node in the execution graph. This along with graph.node.parent_id are used to visualize the execution graph.
     */
    public static final String GRAPH_NODE_ID = SemanticAttributePrefixes.GRAPH + "." + GraphPostfixes.NODE_ID;

    /**
     * The name of the node in the execution graph. Use this to present a human readable name for the node. Optional
     */
    public static final String GRAPH_NODE_NAME = SemanticAttributePrefixes.GRAPH + "." + GraphPostfixes.NODE_NAME;

    /**
     * This references the id of the parent node. Leaving this unset or set as empty string implies that the current span is the root node.
     */
    public static final String GRAPH_NODE_PARENT_ID =
            SemanticAttributePrefixes.GRAPH + "." + GraphPostfixes.NODE_PARENT_ID;

    public static final String OPENINFERENCE_SPAN_KIND = SemanticAttributePrefixes.OPENINFERENCE + ".span.kind";

    /**
     * Semantic conventions for OpenInference span kinds
     */
    public enum OpenInferenceSpanKind {
        LLM("LLM"),
        CHAIN("CHAIN"),
        TOOL("TOOL"),
        RETRIEVER("RETRIEVER"),
        RERANKER("RERANKER"),
        EMBEDDING("EMBEDDING"),
        AGENT("AGENT"),
        GUARDRAIL("GUARDRAIL"),
        EVALUATOR("EVALUATOR");

        private final String value;

        OpenInferenceSpanKind(String value) {
            this.value = value;
        }

        public String getValue() {
            return value;
        }

        @Override
        public String toString() {
            return value;
        }
    }

    /**
     * An enum of common mime types. Not exhaustive.
     */
    public enum MimeType {
        TEXT("text/plain"),
        JSON("application/json"),
        AUDIO_WAV("audio/wav");

        private final String value;

        MimeType(String value) {
            this.value = value;
        }

        public String getValue() {
            return value;
        }

        @Override
        public String toString() {
            return value;
        }
    }

    public enum LLMSystem {
        OPENAI("openai"),
        ANTHROPIC("anthropic"),
        MISTRALAI("mistralai"),
        COHERE("cohere"),
        VERTEXAI("vertexai");

        private final String value;

        LLMSystem(String value) {
            this.value = value;
        }

        public String getValue() {
            return value;
        }

        @Override
        public String toString() {
            return value;
        }
    }

    public enum LLMProvider {
        OPENAI("openai"),
        ANTHROPIC("anthropic"),
        MISTRALAI("mistralai"),
        COHERE("cohere"),
        // Cloud Providers of LLM systems
        GOOGLE("google"),
        AWS("aws"),
        AZURE("azure"),
        XAI("xai"),
        DEEPSEEK("deepseek");

        private final String value;

        LLMProvider(String value) {
            this.value = value;
        }

        public String getValue() {
            return value;
        }

        @Override
        public String toString() {
            return value;
        }
    }
}

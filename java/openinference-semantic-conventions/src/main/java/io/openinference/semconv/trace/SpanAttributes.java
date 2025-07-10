package io.openinference.semconv.trace;

import io.opentelemetry.api.common.AttributeKey;
import java.util.List;

/**
 * Semantic convention attributes for OpenInference spans.
 */
public final class SpanAttributes {

    // Output attributes
    public static final AttributeKey<String> OUTPUT_VALUE = AttributeKey.stringKey("output.value");
    public static final AttributeKey<String> OUTPUT_MIME_TYPE = AttributeKey.stringKey("output.mime_type");

    // Input attributes
    public static final AttributeKey<String> INPUT_VALUE = AttributeKey.stringKey("input.value");
    public static final AttributeKey<String> INPUT_MIME_TYPE = AttributeKey.stringKey("input.mime_type");

    // Embedding attributes
    public static final AttributeKey<String> EMBEDDING_EMBEDDINGS = AttributeKey.stringKey("embedding.embeddings");
    public static final AttributeKey<String> EMBEDDING_MODEL_NAME = AttributeKey.stringKey("embedding.model_name");

    // LLM attributes
    public static final AttributeKey<String> LLM_FUNCTION_CALL = AttributeKey.stringKey("llm.function_call");
    public static final AttributeKey<String> LLM_INVOCATION_PARAMETERS =
            AttributeKey.stringKey("llm.invocation_parameters");
    public static final AttributeKey<String> LLM_INPUT_MESSAGES = AttributeKey.stringKey("llm.input_messages");
    public static final AttributeKey<String> LLM_OUTPUT_MESSAGES = AttributeKey.stringKey("llm.output_messages");
    public static final AttributeKey<String> LLM_MODEL_NAME = AttributeKey.stringKey("llm.model_name");
    public static final AttributeKey<String> LLM_PROVIDER = AttributeKey.stringKey("llm.provider");
    public static final AttributeKey<String> LLM_SYSTEM = AttributeKey.stringKey("llm.system");
    public static final AttributeKey<String> LLM_PROMPTS = AttributeKey.stringKey("llm.prompts");
    public static final AttributeKey<String> LLM_PROMPT_TEMPLATE =
            AttributeKey.stringKey("llm.prompt_template.template");
    public static final AttributeKey<String> LLM_PROMPT_TEMPLATE_VARIABLES =
            AttributeKey.stringKey("llm.prompt_template.variables");
    public static final AttributeKey<String> LLM_PROMPT_TEMPLATE_VERSION =
            AttributeKey.stringKey("llm.prompt_template.version");

    // LLM request parameters
    public static final AttributeKey<Double> LLM_REQUEST_TEMPERATURE =
            AttributeKey.doubleKey("llm.request.temperature");
    public static final AttributeKey<Long> LLM_REQUEST_MAX_TOKENS = AttributeKey.longKey("llm.request.max_tokens");
    public static final AttributeKey<Double> LLM_REQUEST_TOP_P = AttributeKey.doubleKey("llm.request.top_p");
    public static final AttributeKey<Double> LLM_REQUEST_TOP_K = AttributeKey.doubleKey("llm.request.top_k");
    public static final AttributeKey<Double> LLM_REQUEST_FREQUENCY_PENALTY =
            AttributeKey.doubleKey("llm.request.frequency_penalty");
    public static final AttributeKey<Double> LLM_REQUEST_PRESENCE_PENALTY =
            AttributeKey.doubleKey("llm.request.presence_penalty");
    public static final AttributeKey<String> LLM_REQUEST_STOP_SEQUENCES =
            AttributeKey.stringKey("llm.request.stop_sequences");
    public static final AttributeKey<Long> LLM_REQUEST_SEED = AttributeKey.longKey("llm.request.seed");

    // LLM token count attributes
    public static final AttributeKey<Long> LLM_TOKEN_COUNT_COMPLETION =
            AttributeKey.longKey("llm.token_count.completion");
    public static final AttributeKey<Long> LLM_TOKEN_COUNT_COMPLETION_DETAILS_AUDIO =
            AttributeKey.longKey("llm.token_count.completion_details.audio");
    public static final AttributeKey<Long> LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING =
            AttributeKey.longKey("llm.token_count.completion_details.reasoning");
    public static final AttributeKey<Long> LLM_TOKEN_COUNT_PROMPT = AttributeKey.longKey("llm.token_count.prompt");
    public static final AttributeKey<Long> LLM_TOKEN_COUNT_PROMPT_DETAILS_AUDIO =
            AttributeKey.longKey("llm.token_count.prompt_details.audio");
    public static final AttributeKey<Long> LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_INPUT =
            AttributeKey.longKey("llm.token_count.prompt_details.cache_input");
    public static final AttributeKey<Long> LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ =
            AttributeKey.longKey("llm.token_count.prompt_details.cache_read");
    public static final AttributeKey<Long> LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE =
            AttributeKey.longKey("llm.token_count.prompt_details.cache_write");
    public static final AttributeKey<Long> LLM_TOKEN_COUNT_TOTAL = AttributeKey.longKey("llm.token_count.total");

    // LLM cost attributes
    public static final AttributeKey<Double> LLM_COST_COMPLETION = AttributeKey.doubleKey("llm.cost.completion");
    public static final AttributeKey<Double> LLM_COST_COMPLETION_DETAILS_AUDIO =
            AttributeKey.doubleKey("llm.cost.completion_details.audio");
    public static final AttributeKey<Double> LLM_COST_COMPLETION_DETAILS_OUTPUT =
            AttributeKey.doubleKey("llm.cost.completion_details.output");
    public static final AttributeKey<Double> LLM_COST_COMPLETION_DETAILS_REASONING =
            AttributeKey.doubleKey("llm.cost.completion_details.reasoning");
    public static final AttributeKey<Double> LLM_COST_PROMPT = AttributeKey.doubleKey("llm.cost.prompt");
    public static final AttributeKey<Double> LLM_COST_PROMPT_DETAILS_AUDIO =
            AttributeKey.doubleKey("llm.cost.prompt_details.audio");
    public static final AttributeKey<Double> LLM_COST_PROMPT_DETAILS_CACHE_INPUT =
            AttributeKey.doubleKey("llm.cost.prompt_details.cache_input");
    public static final AttributeKey<Double> LLM_COST_PROMPT_DETAILS_CACHE_READ =
            AttributeKey.doubleKey("llm.cost.prompt_details.cache_read");
    public static final AttributeKey<Double> LLM_COST_PROMPT_DETAILS_CACHE_WRITE =
            AttributeKey.doubleKey("llm.cost.prompt_details.cache_write");
    public static final AttributeKey<Double> LLM_COST_PROMPT_DETAILS_INPUT =
            AttributeKey.doubleKey("llm.cost.prompt_details.input");
    public static final AttributeKey<Double> LLM_COST_TOTAL = AttributeKey.doubleKey("llm.cost.total");

    // LLM tools
    public static final AttributeKey<String> LLM_TOOLS = AttributeKey.stringKey("llm.tools");

    // Tool attributes
    public static final AttributeKey<String> TOOL_NAME = AttributeKey.stringKey("tool.name");
    public static final AttributeKey<String> TOOL_DESCRIPTION = AttributeKey.stringKey("tool.description");
    public static final AttributeKey<String> TOOL_PARAMETERS = AttributeKey.stringKey("tool.parameters");

    // Retrieval attributes
    public static final AttributeKey<String> RETRIEVAL_DOCUMENTS = AttributeKey.stringKey("retrieval.documents");

    // Metadata attributes
    public static final AttributeKey<String> METADATA = AttributeKey.stringKey("metadata");

    // Tag attributes
    public static final AttributeKey<String> TAG_TAGS = AttributeKey.stringKey("tag.tags");

    // OpenInference specific attributes
    public static final AttributeKey<String> OPENINFERENCE_SPAN_KIND =
            AttributeKey.stringKey("openinference.span.kind");

    // Session and user attributes
    public static final AttributeKey<String> SESSION_ID = AttributeKey.stringKey("session.id");
    public static final AttributeKey<String> USER_ID = AttributeKey.stringKey("user.id");

    // Agent attributes
    public static final AttributeKey<String> AGENT_NAME = AttributeKey.stringKey("agent.name");

    // Graph attributes
    public static final AttributeKey<String> GRAPH_NODE_ID = AttributeKey.stringKey("graph.node.id");
    public static final AttributeKey<String> GRAPH_NODE_NAME = AttributeKey.stringKey("graph.node.name");
    public static final AttributeKey<String> GRAPH_NODE_PARENT_ID = AttributeKey.stringKey("graph.node.parent_id");

    // Prompt attributes
    public static final AttributeKey<String> PROMPT_VENDOR = AttributeKey.stringKey("prompt.vendor");
    public static final AttributeKey<String> PROMPT_ID = AttributeKey.stringKey("prompt.id");
    public static final AttributeKey<String> PROMPT_URL = AttributeKey.stringKey("prompt.url");

    // URL attributes
    public static final AttributeKey<String> URL_FULL = AttributeKey.stringKey("url.full");
    public static final AttributeKey<String> URL_PATH = AttributeKey.stringKey("url.path");

    // Response attributes
    public static final AttributeKey<List<String>> LLM_RESPONSE_FINISH_REASONS =
            AttributeKey.stringArrayKey("llm.response.finish_reasons");

    private SpanAttributes() {
        // Private constructor to prevent instantiation
    }
}

package com.arize.instrumentation.annotations;

import com.arize.semconv.trace.SemanticConventions;
import lombok.Getter;

@Getter
public enum SpanAttribute {
    // Common
    INPUT_VALUE(SemanticConventions.INPUT_VALUE),
    OUTPUT_VALUE(SemanticConventions.OUTPUT_VALUE),
    INPUT_MIME_TYPE(SemanticConventions.INPUT_MIME_TYPE),
    OUTPUT_MIME_TYPE(SemanticConventions.OUTPUT_MIME_TYPE),
    METADATA(SemanticConventions.METADATA),
    TAG_TAGS(SemanticConventions.TAG_TAGS),
    SESSION_ID(SemanticConventions.SESSION_ID),
    USER_ID(SemanticConventions.USER_ID),

    // LLM
    LLM_MODEL_NAME(SemanticConventions.LLM_MODEL_NAME),
    LLM_SYSTEM(SemanticConventions.LLM_SYSTEM),
    LLM_PROVIDER(SemanticConventions.LLM_PROVIDER),
    LLM_INPUT_MESSAGES(SemanticConventions.LLM_INPUT_MESSAGES),
    LLM_OUTPUT_MESSAGES(SemanticConventions.LLM_OUTPUT_MESSAGES),
    LLM_INVOCATION_PARAMETERS(SemanticConventions.LLM_INVOCATION_PARAMETERS),
    LLM_TOKEN_COUNT_PROMPT(SemanticConventions.LLM_TOKEN_COUNT_PROMPT),
    LLM_TOKEN_COUNT_COMPLETION(SemanticConventions.LLM_TOKEN_COUNT_COMPLETION),
    LLM_TOKEN_COUNT_TOTAL(SemanticConventions.LLM_TOKEN_COUNT_TOTAL),
    LLM_COST_PROMPT(SemanticConventions.LLM_COST_PROMPT),
    LLM_COST_COMPLETION(SemanticConventions.LLM_COST_COMPLETION),
    LLM_COST_TOTAL(SemanticConventions.LLM_COST_TOTAL),
    LLM_PROMPTS(SemanticConventions.LLM_PROMPTS),
    PROMPT_TEMPLATE_TEMPLATE(SemanticConventions.PROMPT_TEMPLATE_TEMPLATE),
    PROMPT_TEMPLATE_VARIABLES(SemanticConventions.PROMPT_TEMPLATE_VARIABLES),
    PROMPT_TEMPLATE_VERSION(SemanticConventions.PROMPT_TEMPLATE_VERSION),

    // Tool
    TOOL_NAME(SemanticConventions.TOOL_NAME),
    TOOL_DESCRIPTION(SemanticConventions.TOOL_DESCRIPTION),
    TOOL_PARAMETERS(SemanticConventions.TOOL_PARAMETERS),
    TOOL_JSON_SCHEMA(SemanticConventions.TOOL_JSON_SCHEMA),

    // Retrieval
    RETRIEVAL_DOCUMENTS(SemanticConventions.RETRIEVAL_DOCUMENTS),

    // Embedding
    EMBEDDING_MODEL_NAME(SemanticConventions.EMBEDDING_MODEL_NAME),
    EMBEDDING_TEXT(SemanticConventions.EMBEDDING_TEXT),
    EMBEDDING_EMBEDDINGS(SemanticConventions.EMBEDDING_EMBEDDINGS),

    // Reranker
    RERANKER_INPUT_DOCUMENTS(SemanticConventions.RERANKER_INPUT_DOCUMENTS),
    RERANKER_OUTPUT_DOCUMENTS(SemanticConventions.RERANKER_OUTPUT_DOCUMENTS),
    RERANKER_QUERY(SemanticConventions.RERANKER_QUERY),
    RERANKER_MODEL_NAME(SemanticConventions.RERANKER_MODEL_NAME),
    RERANKER_TOP_K(SemanticConventions.RERANKER_TOP_K),

    // Agent
    AGENT_NAME(SemanticConventions.AGENT_NAME),

    // Graph
    GRAPH_NODE_ID(SemanticConventions.GRAPH_NODE_ID),
    GRAPH_NODE_NAME(SemanticConventions.GRAPH_NODE_NAME),
    GRAPH_NODE_PARENT_ID(SemanticConventions.GRAPH_NODE_PARENT_ID);

    private final String key;

    SpanAttribute(String key) {
        this.key = key;
    }
}

package com.arize.instrumentation.annotations;

import static org.assertj.core.api.Assertions.assertThat;

import com.arize.semconv.trace.SemanticConventions;
import org.junit.jupiter.api.Test;

class SpanAttributeTest {

    @Test
    void allEnumValuesHaveNonNullKeys() {
        for (SpanAttribute attr : SpanAttribute.values()) {
            assertThat(attr.getKey()).isNotNull().isNotEmpty();
        }
    }

    @Test
    void commonAttributesMappedCorrectly() {
        assertThat(SpanAttribute.INPUT_VALUE.getKey()).isEqualTo(SemanticConventions.INPUT_VALUE);
        assertThat(SpanAttribute.OUTPUT_VALUE.getKey()).isEqualTo(SemanticConventions.OUTPUT_VALUE);
        assertThat(SpanAttribute.SESSION_ID.getKey()).isEqualTo(SemanticConventions.SESSION_ID);
        assertThat(SpanAttribute.USER_ID.getKey()).isEqualTo(SemanticConventions.USER_ID);
        assertThat(SpanAttribute.METADATA.getKey()).isEqualTo(SemanticConventions.METADATA);
    }

    @Test
    void llmAttributesMappedCorrectly() {
        assertThat(SpanAttribute.LLM_MODEL_NAME.getKey()).isEqualTo(SemanticConventions.LLM_MODEL_NAME);
        assertThat(SpanAttribute.LLM_SYSTEM.getKey()).isEqualTo(SemanticConventions.LLM_SYSTEM);
        assertThat(SpanAttribute.LLM_PROVIDER.getKey()).isEqualTo(SemanticConventions.LLM_PROVIDER);
        assertThat(SpanAttribute.LLM_INPUT_MESSAGES.getKey()).isEqualTo(SemanticConventions.LLM_INPUT_MESSAGES);
        assertThat(SpanAttribute.LLM_TOKEN_COUNT_TOTAL.getKey()).isEqualTo(SemanticConventions.LLM_TOKEN_COUNT_TOTAL);
        assertThat(SpanAttribute.LLM_COST_TOTAL.getKey()).isEqualTo(SemanticConventions.LLM_COST_TOTAL);
    }

    @Test
    void toolAttributesMappedCorrectly() {
        assertThat(SpanAttribute.TOOL_NAME.getKey()).isEqualTo(SemanticConventions.TOOL_NAME);
        assertThat(SpanAttribute.TOOL_DESCRIPTION.getKey()).isEqualTo(SemanticConventions.TOOL_DESCRIPTION);
        assertThat(SpanAttribute.TOOL_PARAMETERS.getKey()).isEqualTo(SemanticConventions.TOOL_PARAMETERS);
    }

    @Test
    void retrievalAttributesMappedCorrectly() {
        assertThat(SpanAttribute.RETRIEVAL_DOCUMENTS.getKey()).isEqualTo(SemanticConventions.RETRIEVAL_DOCUMENTS);
    }

    @Test
    void embeddingAttributesMappedCorrectly() {
        assertThat(SpanAttribute.EMBEDDING_MODEL_NAME.getKey()).isEqualTo(SemanticConventions.EMBEDDING_MODEL_NAME);
        assertThat(SpanAttribute.EMBEDDING_TEXT.getKey()).isEqualTo(SemanticConventions.EMBEDDING_TEXT);
    }

    @Test
    void agentAttributesMappedCorrectly() {
        assertThat(SpanAttribute.AGENT_NAME.getKey()).isEqualTo(SemanticConventions.AGENT_NAME);
    }
}

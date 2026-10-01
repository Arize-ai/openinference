package com.arize.semconv.trace;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

/**
 * Checks the literal attribute keys. These keys are the wire format shared with the Go, JavaScript,
 * and Python packages, so any change to them breaks consumers. The expected values are spelled out
 * in full rather than built from the constants under test.
 */
class SemanticConventionsTest {

    @Test
    void spanLevelImageAttributesUseTheDocumentedKeys() {
        assertThat(SemanticConventions.INPUT_IMAGES).isEqualTo("input.images");
        assertThat(SemanticConventions.OUTPUT_IMAGES).isEqualTo("output.images");
    }

    @Test
    void imageObjectAttributesUseTheDocumentedKeys() {
        assertThat(SemanticConventions.IMAGE_URL).isEqualTo("image.url");
    }

    @Test
    void indexedImagePathsMatchTheFlattenedPattern() {
        assertThat(SemanticConventions.INPUT_IMAGES + ".0." + SemanticConventions.IMAGE_URL)
                .isEqualTo("input.images.0.image.url");
        assertThat(SemanticConventions.OUTPUT_IMAGES + ".1." + SemanticConventions.IMAGE_URL)
                .isEqualTo("output.images.1.image.url");
    }

    @Test
    void messageContentImagePathIsUnchanged() {
        assertThat(SemanticConventions.MESSAGE_CONTENT_IMAGE + "." + SemanticConventions.IMAGE_URL)
                .isEqualTo("message_content.image.image.url");
    }

    @Test
    void decisionAttributesUseTheDocumentedKeys() {
        assertThat(SemanticConventions.DECISION_MODEL_NAME).isEqualTo("decision.model_name");
        assertThat(SemanticConventions.DECISION_REQUEST_MODEL_NAME).isEqualTo("decision.request.model_name");
        assertThat(SemanticConventions.DECISION_RESPONSE_MODEL_NAME).isEqualTo("decision.response.model_name");
        assertThat(SemanticConventions.DECISION_PROVIDER).isEqualTo("decision.provider");
        assertThat(SemanticConventions.DECISION_SYSTEM).isEqualTo("decision.system");
    }

    @Test
    void decisionSystemAndProviderValuesAliasTheLlmValues() {
        assertThat(SemanticConventions.DecisionSystem.TYPESAFE.getValue()).isEqualTo("typesafe");
        assertThat(SemanticConventions.DecisionSystem.OPENAI.getValue()).isEqualTo("openai");
        assertThat(SemanticConventions.DecisionProvider.TYPESAFE.getValue()).isEqualTo("typesafe");
        assertThat(SemanticConventions.DecisionProvider.OPENAI.getValue()).isEqualTo("openai");
        for (SemanticConventions.DecisionSystem system : SemanticConventions.DecisionSystem.values()) {
            assertThat(system.getLLMSystem()).isEqualTo(SemanticConventions.LLMSystem.valueOf(system.name()));
            assertThat(system.toString()).isEqualTo(system.getLLMSystem().getValue());
        }
        for (SemanticConventions.DecisionProvider provider : SemanticConventions.DecisionProvider.values()) {
            assertThat(provider.getLLMProvider()).isEqualTo(SemanticConventions.LLMProvider.valueOf(provider.name()));
            assertThat(provider.toString()).isEqualTo(provider.getLLMProvider().getValue());
        }
        assertThat(SemanticConventions.LLMSystem.TYPESAFE.getValue()).isEqualTo("typesafe");
        assertThat(SemanticConventions.LLMProvider.TYPESAFE.getValue()).isEqualTo("typesafe");
    }

    @Test
    void decisionSpanKindUsesTheDocumentedValue() {
        assertThat(SemanticConventions.OpenInferenceSpanKind.DECISION.getValue())
                .isEqualTo("DECISION");
        assertThat(SemanticConventions.OpenInferenceSpanKind.DECISION.toString())
                .isEqualTo("DECISION");
    }
}

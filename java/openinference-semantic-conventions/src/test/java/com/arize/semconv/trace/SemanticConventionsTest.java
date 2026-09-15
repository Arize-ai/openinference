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
}

package com.arize.instrumentation;

import lombok.Builder;
import lombok.Getter;

/**
 * Configuration for OpenInference tracing.
 */
@Getter
@Builder
public class TraceConfig {

    @Builder.Default
    private final boolean hideInputs = false;

    @Builder.Default
    private final boolean hideOutputs = false;

    @Builder.Default
    private final boolean hideInputMessages = false;

    @Builder.Default
    private final boolean hideOutputMessages = false;

    @Builder.Default
    private final boolean hideInputImages = false;

    @Builder.Default
    private final boolean hideOutputImages = false;

    @Builder.Default
    private final boolean hideInputText = false;

    @Builder.Default
    private final boolean hideOutputText = false;

    @Builder.Default
    private final boolean hideInputAudio = false;

    @Builder.Default
    private final boolean hideOutputAudio = false;

    @Builder.Default
    private final boolean hideInputEmbeddings = false;

    @Builder.Default
    private final boolean hideOutputEmbeddings = false;

    @Builder.Default
    private final boolean hidePromptTemplate = false;

    @Builder.Default
    private final boolean hidePromptTemplateVariables = false;

    @Builder.Default
    private final boolean hidePromptTemplateVersion = false;

    @Builder.Default
    private final boolean hideToolParameters = false;

    @Builder.Default
    private final String base64ImageMaxLength = "unlimited";

    public static TraceConfig getDefault() {
        return builder().build();
    }
}

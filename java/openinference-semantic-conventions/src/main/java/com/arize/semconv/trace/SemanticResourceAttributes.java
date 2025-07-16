package com.arize.semconv.trace;

import lombok.experimental.UtilityClass;

/**
 * Semantic conventions for OpenInference resource attributes
 */
@UtilityClass
public class SemanticResourceAttributes {
    private static final String PROJECT_NAME = "openinference.project.name";

    /**
     * The project name to group traces under for openinference compatible services
     */
    public static final String SEMRESATTRS_PROJECT_NAME = PROJECT_NAME;
}

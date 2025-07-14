package com.arize.instrumentation.trace.config;

import com.arize.instrumentation.TraceConfig;
import com.arize.semconv.trace.SemanticConventions;
import io.opentelemetry.api.common.Value;
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;
import java.util.function.Predicate;
import lombok.Builder;
import lombok.Getter;

/**
 * Arguments for masking rules
 */
@Getter
@Builder
class MaskingRuleArgs {
    private TraceConfig config;
    private String key;
    private Value value;
}

/**
 * A masking rule that defines when and how to mask attributes
 */
@Getter
class MaskingRule {
    private final Predicate<MaskingRuleArgs> condition;
    private final Function<MaskingRuleArgs, Value> action;

    public MaskingRule(Predicate<MaskingRuleArgs> condition, Function<MaskingRuleArgs, Value> action) {
        this.condition = condition;
        this.action = action;
    }
}

/**
 * Utility class for masking sensitive information in span attributes
 */
class MaskingUtils {

    private static final Value<String> REDACTED_VALUE = Value.of("[REDACTED]");

    /**
     * Masks (redacts) input text in LLM input messages.
     * Will mask information stored under the key `llm.input_messages.[i].message.content`.
     */
    private static final MaskingRule maskInputTextRule = new MaskingRule(
            args -> args.getConfig().isHideInputText()
                    && args.getKey().contains(SemanticConventions.LLM_INPUT_MESSAGES)
                    && args.getKey().contains(SemanticConventions.MESSAGE_CONTENT)
                    && !args.getKey().contains(SemanticConventions.MESSAGE_CONTENTS),
            args -> REDACTED_VALUE);

    /**
     * Masks (redacts) output text in LLM output messages.
     * Will mask information stored under the key `llm.output_messages.[i].message.content`.
     */
    private static final MaskingRule maskOutputTextRule = new MaskingRule(
            args -> args.getConfig().isHideOutputText()
                    && args.getKey().contains(SemanticConventions.LLM_OUTPUT_MESSAGES)
                    && args.getKey().contains(SemanticConventions.MESSAGE_CONTENT)
                    && !args.getKey().contains(SemanticConventions.MESSAGE_CONTENTS),
            args -> REDACTED_VALUE);

    /**
     * Masks (redacts) input text content in LLM input messages.
     * Will mask information stored under the key `llm.input_messages.[i].message.contents.[j].message_content.text`.
     */
    private static final MaskingRule maskInputTextContentRule = new MaskingRule(
            args -> args.getConfig().isHideInputText()
                    && args.getKey().contains(SemanticConventions.LLM_INPUT_MESSAGES)
                    && args.getKey().contains(SemanticConventions.MESSAGE_CONTENT_TEXT),
            args -> REDACTED_VALUE);

    /**
     * Masks (redacts) output text content in LLM output messages.
     */
    private static final MaskingRule maskOutputTextContentRule = new MaskingRule(
            args -> args.getConfig().isHideOutputText()
                    && args.getKey().contains(SemanticConventions.LLM_OUTPUT_MESSAGES)
                    && args.getKey().contains(SemanticConventions.MESSAGE_CONTENT_TEXT),
            args -> REDACTED_VALUE);

    /**
     * Masks (removes) input images in LLM input messages.
     */
    private static final MaskingRule maskInputImagesRule = new MaskingRule(
            args -> args.getConfig().isHideInputImages()
                    && args.getKey().contains(SemanticConventions.LLM_INPUT_MESSAGES)
                    && args.getKey().contains(SemanticConventions.MESSAGE_CONTENT_IMAGE),
            args -> null);

    /**
     * Masks (redacts) base64 images that are too long.
     */
    private static final MaskingRule maskLongBase64ImageRule = new MaskingRule(
            args -> {
                Object value = args.getValue();
                if (value instanceof String) {
                    isBase64Url((String) value);
                }
                return false;
            },
            args -> REDACTED_VALUE);

    /**
     * Masks (removes) embedding vectors.
     */
    private static final MaskingRule maskEmbeddingVectorsRule = new MaskingRule(
            args -> args.getConfig().isHideInputEmbeddings()
                    && args.getKey().contains(SemanticConventions.EMBEDDING_EMBEDDINGS)
                    && args.getKey().contains(SemanticConventions.EMBEDDING_VECTOR),
            args -> null);

    /**
     * A list of masking rules that are applied to span attributes to either redact or remove sensitive information.
     * The order of these rules is important as it can ensure appropriate masking of information.
     * Rules should go from more specific to more general.
     */
    private static final List<MaskingRule> maskingRules = Arrays.asList(
            new MaskingRule(
                    args -> args.getConfig().isHideInputs() && args.getKey().equals(SemanticConventions.INPUT_VALUE),
                    args -> REDACTED_VALUE),
            new MaskingRule(
                    args -> args.getConfig().isHideInputs()
                            && args.getKey().equals(SemanticConventions.INPUT_MIME_TYPE),
                    args -> null),
            new MaskingRule(
                    args -> args.getConfig().isHideOutputs() && args.getKey().equals(SemanticConventions.OUTPUT_VALUE),
                    args -> REDACTED_VALUE),
            new MaskingRule(
                    args -> args.getConfig().isHideOutputs()
                            && args.getKey().equals(SemanticConventions.OUTPUT_MIME_TYPE),
                    args -> null),
            new MaskingRule(
                    args -> (args.getConfig().isHideInputs() || args.getConfig().isHideInputMessages())
                            && args.getKey().contains(SemanticConventions.LLM_INPUT_MESSAGES),
                    args -> null),
            new MaskingRule(
                    args -> (args.getConfig().isHideOutputs()
                                    || args.getConfig().isHideOutputMessages())
                            && args.getKey().contains(SemanticConventions.LLM_OUTPUT_MESSAGES),
                    args -> null),
            maskInputTextRule,
            maskOutputTextRule,
            maskInputTextContentRule,
            maskOutputTextContentRule,
            maskInputImagesRule,
            maskLongBase64ImageRule,
            maskEmbeddingVectorsRule);

    /**
     * Checks if a URL is a base64 encoded image
     */
    private static boolean isBase64Url(String url) {
        return url != null && url.startsWith("data:image/") && url.contains("base64");
    }

    /**
     * A function that masks (redacts or removes) sensitive information from span attributes based on the trace config.
     * @param args The masking rule arguments containing config, key, and value
     * @return The redacted value or null if the value should be masked, otherwise the original value
     */
    public static Value mask(MaskingRuleArgs args) {
        for (MaskingRule rule : maskingRules) {
            if (rule.getCondition().test(args)) {
                return rule.getAction().apply(args);
            }
        }
        return args.getValue();
    }

    /**
     * Convenience method for masking with individual parameters
     */
    public static Value mask(TraceConfig config, String key, Value value) {
        return mask(
                MaskingRuleArgs.builder().config(config).key(key).value(value).build());
    }
}

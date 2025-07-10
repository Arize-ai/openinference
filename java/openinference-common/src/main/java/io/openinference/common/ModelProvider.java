package io.openinference.common;

import java.util.HashMap;
import java.util.Map;
import java.util.regex.Pattern;
import lombok.experimental.UtilityClass;

/**
 * Utility class for detecting LLM providers based on model names.
 * Based on the Phoenix model cost manifest.
 */
@UtilityClass
public class ModelProvider {

    // Provider constants
    public enum Provider {
        OPENAI,
        AZURE_OPENAI,
        ANTHROPIC,
        GOOGLE,
        DEEPSEEK,
        XAI,
        OLLAMA,
        AWS,
    }

    // Model patterns mapped to providers
    private static final Map<Pattern, String> MODEL_PATTERNS = new HashMap<>();

    static {

        // Generated from phoenix/src/phoenix/server/cost_tracking/model_cost_manifest.json
        // OpenAI
        MODEL_PATTERNS.put(
                Pattern.compile("(?i)(^chatgpt-4o-latest|" + "^gpt-(35|3.5)-turbo|"
                        + "^gpt-(35|3.5)-turbo-0125|"
                        + "^gpt-(35|3.5)-turbo-0301|"
                        + "^gpt-(35|3.5)-turbo-0613|"
                        + "^gpt-(35|3.5)-turbo-1106|"
                        + "^gpt-(35|3.5)-turbo-16k|"
                        + "^gpt-(35|3.5)-turbo-16k-0613|"
                        + "^gpt-(35|3.5)-turbo-instruct|"
                        + "^gpt-3\\.5-turbo-instruct-0914|"
                        + "^gpt-4|"
                        + "^gpt-4-0125-preview|"
                        + "^gpt-4-0314|"
                        + "^gpt-4-0613|"
                        + "^gpt-4-1106-preview|"
                        + "^gpt-4-1106-vision-preview|"
                        + "^gpt-4-32k|"
                        + "^gpt-4-32k-0314|"
                        + "^gpt-4-32k-0613|"
                        + "^gpt-4-turbo|"
                        + "^gpt-4-turbo-2024-04-09|"
                        + "^gpt-4-turbo-preview|"
                        + "^gpt-4-vision-preview|"
                        + "^gpt-4.1|"
                        + "^gpt-4\\.1-2025-04-14|"
                        + "^gpt-4\\.1-mini|"
                        + "^gpt-4\\.1-mini-2025-04-14|"
                        + "^gpt-4\\.1-nano|"
                        + "^gpt-4\\.1-nano-2025-04-14|"
                        + "^gpt-4\\.5-preview|"
                        + "^gpt-4\\.5-preview-2025-02-27|"
                        + "^gpt-4o|"
                        + "^gpt-4o-2024-05-13|"
                        + "^gpt-4o-2024-08-06|"
                        + "^gpt-4o-2024-11-20|"
                        + "^gpt-4o-audio-preview|"
                        + "^gpt-4o-audio-preview-2024-10-01|"
                        + "^gpt-4o-audio-preview-2024-12-17|"
                        + "^gpt-4o-audio-preview-2025-06-03|"
                        + "^gpt-4o-mini|"
                        + "^gpt-4o-mini-2024-07-18|"
                        + "^gpt-4o-mini-audio-preview|"
                        + "^gpt-4o-mini-audio-preview-2024-12-17|"
                        + "^gpt-4o-mini-realtime-preview|"
                        + "^gpt-4o-mini-realtime-preview-2024-12-17|"
                        + "^gpt-4o-mini-search-preview|"
                        + "^gpt-4o-mini-search-preview-2025-03-11|"
                        + "^gpt-4o-mini-transcribe|"
                        + "^gpt-4o-mini-tts|"
                        + "^gpt-4o-realtime-preview|"
                        + "^gpt-4o-realtime-preview-2024-10-01|"
                        + "^gpt-4o-realtime-preview-2024-12-17|"
                        + "^gpt-4o-search-preview|"
                        + "^gpt-4o-search-preview-2025-03-11|"
                        + "^gpt-4o-transcribe)"),
                Provider.OPENAI.name());

        // Anthropic
        MODEL_PATTERNS.put(
                Pattern.compile(
                        "(?i)(^claude-3-5-haiku-20241022|anthropic\\.claude-3-5-haiku-20241022-v1:0|claude-3-5-haiku-V1@20241022|"
                                + "^claude-3-5-haiku-latest|"
                                + "^claude-3-5-sonnet-20240620|anthropic\\.claude-3-5-sonnet-20240620-v1:0|claude-3-5-sonnet@20240620|"
                                + "^claude-3-5-sonnet-20241022|anthropic\\.claude-3-5-sonnet-20241022-v2:0|claude-3-5-sonnet-V2@20241022|"
                                + "^claude-3-5-sonnet-latest|"
                                + "^claude-3-7-sonnet-latest|"
                                + "^claude-3-haiku-20240307|anthropic\\.claude-3-haiku-20240307-v1:0|claude-3-haiku@20240307|"
                                + "^claude-3-opus-20240229|anthropic\\.claude-3-opus-20240229-v1:0|claude-3-opus@20240229|"
                                + "^claude-3-opus-latest|"
                                + "^claude-3-sonnet-20240229|anthropic\\.claude-3-sonnet-20240229-v1:0|claude-3-sonnet@20240229|"
                                + "^claude-3.7-sonnet-20250219|anthropic\\.claude-3.7-sonnet-20250219-v1:0|claude-3-7-sonnet-V1@20250219|"
                                + "^claude-4-opus-20250514|"
                                + "^claude-4-sonnet-20250514|"
                                + "^claude-opus-4-20250514|"
                                + "^claude-sonnet-4-20250514|anthropic\\.claude-sonnet-4-20250514-v1:0|claude-sonnet-4@20250514)"),
                Provider.ANTHROPIC.name());

        // Google
        MODEL_PATTERNS.put(
                Pattern.compile("(?i)(^gemini-2.0-flash(@[a-zA-Z0-9]+)?|" + "^gemini-2.0-flash-001(@[a-zA-Z0-9]+)?|"
                        + "^gemini-2.0-flash-lite(@[a-zA-Z0-9]+)?|"
                        + "^gemini-2.5-flash(@[a-zA-Z0-9]+)?|"
                        + "^gemini-2.5-pro(@[a-zA-Z0-9]+)?|"
                        + "^gemini-2\\.0-flash-exp|"
                        + "^gemini-2\\.0-flash-lite-001|"
                        + "^gemini-2\\.0-flash-preview-image-generation|"
                        + "^gemini-2\\.0-pro-exp-02-05|"
                        + "^gemini-2\\.5-flash-lite-preview-06-17|"
                        + "^gemini-2\\.5-flash-preview-04-17|"
                        + "^gemini-2\\.5-flash-preview-05-20|"
                        + "^gemini-2\\.5-pro-exp-03-25|"
                        + "^gemini-2\\.5-pro-preview-03-25|"
                        + "^gemini-2\\.5-pro-preview-05-06|"
                        + "^gemini-2\\.5-pro-preview-06-05|"
                        + "^gemini-2\\.5-pro-preview-tts)"),
                Provider.GOOGLE.name());
    }

    /**
     * Detects the provider based on the model name.
     *
     * @param modelName the name of the model
     * @return the detected provider or null if not detected
     */
    public static String detectProvider(String modelName) {
        if (modelName == null || modelName.isEmpty()) {
            return null;
        }

        // Check patterns
        for (Map.Entry<Pattern, String> entry : MODEL_PATTERNS.entrySet()) {
            if (entry.getKey().matcher(modelName).find()) {
                return entry.getValue();
            }
        }

        return null;
    }

    /**
     * Checks if a model name belongs to a specific provider.
     *
     * @param modelName the model name
     * @param provider the provider to check
     * @return true if the model belongs to the provider
     */
    public static boolean isProvider(String modelName, String provider) {
        String detectedProvider = detectProvider(modelName);
        return provider != null && provider.equals(detectedProvider);
    }
}

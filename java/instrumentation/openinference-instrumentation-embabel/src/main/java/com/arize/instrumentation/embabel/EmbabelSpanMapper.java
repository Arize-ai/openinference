package com.arize.instrumentation.embabel;

import static com.arize.semconv.trace.SemanticConventions.*;

import com.arize.instrumentation.TraceConfig;
import com.arize.semconv.trace.SemanticConventions.LLMProvider;
import com.arize.semconv.trace.SemanticConventions.MimeType;
import com.arize.semconv.trace.SemanticConventions.OpenInferenceSpanKind;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.opentelemetry.api.common.AttributeKey;
import io.opentelemetry.api.common.Attributes;
import io.opentelemetry.api.common.AttributesBuilder;
import io.opentelemetry.sdk.trace.data.DelegatingSpanData;
import io.opentelemetry.sdk.trace.data.SpanData;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;

/**
 * Translates the spans emitted by Embabel's {@code embabel-agent-observability} module (and the
 * Spring AI chat model spans it enriches) into OpenInference spans.
 *
 * <p>Embabel classifies every span it emits with {@code embabel.event.type}; the Spring AI chat
 * model span carries no Embabel tag but has {@code gen_ai.operation.name=chat}. Spans that match
 * neither, or that already carry {@code openinference.span.kind}, are returned unchanged.
 *
 * <p>Micrometer's OpenTelemetry bridge records every tag as a string, so numeric GenAI values
 * (token counts, temperature) are parsed here.
 */
final class EmbabelSpanMapper {

    // Keys written by embabel-agent-observability (com.embabel.agent.observability.SpanAttributes).
    static final String EMBABEL_EVENT_TYPE = "embabel.event.type";
    static final String EMBABEL_AGENT_NAME = "embabel.agent.name";
    static final String EMBABEL_AGENT_RESULT = "embabel.agent.result";
    static final String EMBABEL_ACTION_RESULT = "embabel.action.result";
    static final String EMBABEL_GOAL_RESULT = "embabel.goal.result";
    static final String EMBABEL_TRACKED_ARGS = "embabel.tracked.args";
    static final String EMBABEL_TRACKED_RESULT = "embabel.tracked.result";
    static final String EMBABEL_RAG_QUERY = "embabel.rag.query";
    static final String EMBABEL_DYNAMIC_AGENT_BASIS = "embabel.dynamic_agent.basis";
    static final String EMBABEL_REPLAN_REASON = "embabel.replan.reason";

    /**
     * Internal marker that {@link OpenInferenceEmbabelSpanProcessor} sets on spans started while
     * tracing is suppressed. Marked spans are exported unmapped, and the marker is removed.
     */
    static final AttributeKey<Boolean> SUPPRESSED = AttributeKey.booleanKey("openinference.embabel.suppressed");

    // OpenTelemetry GenAI semantic convention keys, as Embabel and Spring AI write them.
    static final String GEN_AI_OPERATION_NAME = "gen_ai.operation.name";
    static final String GEN_AI_CONVERSATION_ID = "gen_ai.conversation.id";
    static final String GEN_AI_PROVIDER_NAME = "gen_ai.provider.name";
    static final String GEN_AI_SYSTEM = "gen_ai.system";
    static final String GEN_AI_REQUEST_MODEL = "gen_ai.request.model";
    static final String GEN_AI_RESPONSE_MODEL = "gen_ai.response.model";
    static final String GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature";
    static final String GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens";
    static final String GEN_AI_REQUEST_TOP_P = "gen_ai.request.top_p";
    static final String GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens";
    static final String GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens";
    static final String GEN_AI_USAGE_TOTAL_TOKENS = "gen_ai.usage.total_tokens";
    static final String GEN_AI_INPUT_MESSAGES = "gen_ai.input.messages";
    static final String GEN_AI_OUTPUT_MESSAGES = "gen_ai.output.messages";
    static final String GEN_AI_TOOL_NAME = "gen_ai.tool.name";
    static final String GEN_AI_TOOL_DESCRIPTION = "gen_ai.tool.description";
    static final String GEN_AI_TOOL_CALL_ARGUMENTS = "gen_ai.tool.call.arguments";
    static final String GEN_AI_TOOL_CALL_RESULT = "gen_ai.tool.call.result";

    /** Source keys that hold input content: dropped from the exported span when inputs are hidden. */
    private static final Set<String> INPUT_CONTENT_KEYS = Set.of(
            INPUT_VALUE,
            INPUT_MIME_TYPE,
            GEN_AI_INPUT_MESSAGES,
            EMBABEL_TRACKED_ARGS,
            EMBABEL_RAG_QUERY,
            // Embabel's Autonomy fills the basis of a dynamic agent from the user's input.
            EMBABEL_DYNAMIC_AGENT_BASIS);

    /** Source keys that hold output content: dropped from the exported span when outputs are hidden. */
    private static final Set<String> OUTPUT_CONTENT_KEYS = Set.of(
            OUTPUT_VALUE,
            OUTPUT_MIME_TYPE,
            GEN_AI_OUTPUT_MESSAGES,
            GEN_AI_TOOL_CALL_RESULT,
            EMBABEL_AGENT_RESULT,
            EMBABEL_ACTION_RESULT,
            EMBABEL_GOAL_RESULT,
            EMBABEL_TRACKED_RESULT,
            // Written by the tool or action that asks for a replan, so it is the step's output.
            EMBABEL_REPLAN_REASON);

    private static final Map<String, LLMProvider> PROVIDERS = Map.ofEntries(
            Map.entry("openai", LLMProvider.OPENAI),
            Map.entry("anthropic", LLMProvider.ANTHROPIC),
            Map.entry("mistral_ai", LLMProvider.MISTRALAI),
            Map.entry("mistralai", LLMProvider.MISTRALAI),
            Map.entry("cohere", LLMProvider.COHERE),
            Map.entry("vertex_ai", LLMProvider.GOOGLE),
            Map.entry("google_genai", LLMProvider.GOOGLE),
            Map.entry("gcp.gemini", LLMProvider.GOOGLE),
            Map.entry("gcp.vertex_ai", LLMProvider.GOOGLE),
            Map.entry("bedrock_converse", LLMProvider.AWS),
            Map.entry("aws.bedrock", LLMProvider.AWS),
            Map.entry("azure_openai", LLMProvider.AZURE),
            Map.entry("azure.ai.openai", LLMProvider.AZURE),
            Map.entry("deepseek", LLMProvider.DEEPSEEK),
            Map.entry("groq", LLMProvider.GROQ),
            Map.entry("xai", LLMProvider.XAI));

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private final TraceConfig config;

    EmbabelSpanMapper(TraceConfig config) {
        this.config = config;
    }

    /**
     * Returns the OpenInference view of {@code span}, or {@code span} itself when it is not an Embabel
     * span or tracing is suppressed.
     */
    SpanData map(SpanData span) {
        Attributes source = span.getAttributes();
        if (source.get(SUPPRESSED) != null) {
            return new MappedSpanData(
                    span, source.toBuilder().remove(SUPPRESSED).build());
        }
        if (config.isSuppressTracing() || source.get(AttributeKey.stringKey(OPENINFERENCE_SPAN_KIND)) != null) {
            return span;
        }
        OpenInferenceSpanKind kind = classify(source);
        if (kind == null) {
            return span;
        }

        AttributesBuilder target = Attributes.builder();
        source.forEach((key, value) -> {
            if (!isHiddenContent(key.getKey(), kind)) {
                putUnchecked(target, key, value);
            }
        });
        Attributes kept = target.build();
        target.put(OPENINFERENCE_SPAN_KIND, kind.getValue());

        switch (kind) {
            case AGENT -> writeAgent(source, kept, target);
            case LLM -> writeLlm(source, target);
            case TOOL -> writeTool(source, kept, target);
            case EMBEDDING -> set(target, EMBEDDING_MODEL_NAME, string(source, GEN_AI_REQUEST_MODEL));
            case RETRIEVER -> setTextIfAbsent(
                    kept, target, INPUT_VALUE, INPUT_MIME_TYPE, string(source, EMBABEL_RAG_QUERY));
            default -> {}
        }
        writeMimeTypes(target);
        return new MappedSpanData(span, target.build());
    }

    static OpenInferenceSpanKind classify(Attributes attributes) {
        String eventType = string(attributes, EMBABEL_EVENT_TYPE);
        if (eventType != null) {
            // The "llm <name>" wrapper and the llm.invocation cost record stay CHAIN: the nested
            // chat model span is the generation, and a second LLM span would double-count tokens.
            return switch (eventType) {
                case "agent_process" -> OpenInferenceSpanKind.AGENT;
                case "tool_call" -> OpenInferenceSpanKind.TOOL;
                case "embedding" -> OpenInferenceSpanKind.EMBEDDING;
                case "rag" -> OpenInferenceSpanKind.RETRIEVER;
                default -> OpenInferenceSpanKind.CHAIN;
            };
        }
        String operation = string(attributes, GEN_AI_OPERATION_NAME);
        if (operation == null) {
            return null;
        }
        return switch (operation) {
            case "chat", "text_completion", "generate_content" -> OpenInferenceSpanKind.LLM;
            case "embeddings" -> OpenInferenceSpanKind.EMBEDDING;
            case "execute_tool" -> OpenInferenceSpanKind.TOOL;
            default -> null;
        };
    }

    private void writeAgent(Attributes source, Attributes kept, AttributesBuilder target) {
        set(target, AGENT_NAME, string(source, EMBABEL_AGENT_NAME));
        if (kept.get(AttributeKey.stringKey(SESSION_ID)) == null) {
            set(target, SESSION_ID, string(source, GEN_AI_CONVERSATION_ID));
        }
    }

    private void writeLlm(Attributes source, AttributesBuilder target) {
        String model = string(source, GEN_AI_RESPONSE_MODEL);
        set(target, LLM_MODEL_NAME, model != null ? model : string(source, GEN_AI_REQUEST_MODEL));
        set(target, LLM_PROVIDER, provider(source));

        Long prompt = number(source, GEN_AI_USAGE_INPUT_TOKENS, Long::valueOf);
        Long completion = number(source, GEN_AI_USAGE_OUTPUT_TOKENS, Long::valueOf);
        Long total = number(source, GEN_AI_USAGE_TOTAL_TOKENS, Long::valueOf);
        if (total == null && prompt != null && completion != null) {
            total = prompt + completion;
        }
        set(target, LLM_TOKEN_COUNT_PROMPT, prompt);
        set(target, LLM_TOKEN_COUNT_COMPLETION, completion);
        set(target, LLM_TOKEN_COUNT_TOTAL, total);

        Map<String, Object> parameters = new LinkedHashMap<>();
        putIfPresent(parameters, "temperature", number(source, GEN_AI_REQUEST_TEMPERATURE, Double::valueOf));
        putIfPresent(parameters, "max_tokens", number(source, GEN_AI_REQUEST_MAX_TOKENS, Long::valueOf));
        putIfPresent(parameters, "top_p", number(source, GEN_AI_REQUEST_TOP_P, Double::valueOf));
        if (!parameters.isEmpty()) {
            set(target, LLM_INVOCATION_PARAMETERS, toJson(parameters));
        }

        writeMessages(target, LLM_INPUT_MESSAGES, string(source, GEN_AI_INPUT_MESSAGES));
        writeMessages(target, LLM_OUTPUT_MESSAGES, string(source, GEN_AI_OUTPUT_MESSAGES));
    }

    private void writeTool(Attributes source, Attributes kept, AttributesBuilder target) {
        set(target, TOOL_NAME, string(source, GEN_AI_TOOL_NAME));
        set(target, TOOL_DESCRIPTION, string(source, GEN_AI_TOOL_DESCRIPTION));
        String arguments = string(source, GEN_AI_TOOL_CALL_ARGUMENTS);
        set(target, TOOL_PARAMETERS, arguments);
        if (!config.isHideToolParameters()) {
            setTextIfAbsent(kept, target, INPUT_VALUE, INPUT_MIME_TYPE, arguments);
        }
        setTextIfAbsent(kept, target, OUTPUT_VALUE, OUTPUT_MIME_TYPE, string(source, GEN_AI_TOOL_CALL_RESULT));
    }

    /**
     * Flattens a GenAI messages array ({@code [{"role": ..., "parts": [...]}]}) into
     * {@code llm.input_messages.<i>.message.*} / {@code llm.output_messages.<i>.message.*}.
     */
    private void writeMessages(AttributesBuilder target, String prefix, String json) {
        JsonNode messages = parse(json);
        if (messages == null || !messages.isArray()) {
            return;
        }
        for (int i = 0; i < messages.size(); i++) {
            JsonNode message = messages.get(i);
            String base = prefix + "." + i + ".";
            set(target, base + MESSAGE_ROLE, text(message.get("role")));

            StringBuilder content = new StringBuilder();
            int toolCall = 0;
            for (JsonNode part : message.path("parts")) {
                String type = part.path("type").asText("");
                switch (type) {
                    case "text" -> {
                        if (content.length() > 0) {
                            content.append('\n');
                        }
                        content.append(part.path("content").asText(""));
                    }
                    case "tool_call" -> {
                        String call = base + MESSAGE_TOOL_CALLS + "." + toolCall++ + ".";
                        set(target, call + TOOL_CALL_ID, text(part.get("id")));
                        set(target, call + TOOL_CALL_FUNCTION_NAME, text(part.get("name")));
                        JsonNode arguments = part.get("arguments");
                        set(
                                target,
                                call + TOOL_CALL_FUNCTION_ARGUMENTS_JSON,
                                arguments == null || arguments.isTextual() ? text(arguments) : arguments.toString());
                    }
                    case "tool_call_response" -> {
                        set(target, base + MESSAGE_TOOL_CALL_ID, text(part.get("id")));
                        JsonNode response = part.get("response");
                        if (response != null) {
                            content.append(response.isTextual() ? response.asText() : response.toString());
                        }
                    }
                    default -> {}
                }
            }
            if (content.length() > 0) {
                set(target, base + MESSAGE_CONTENT, content.toString());
            }
        }
    }

    private void writeMimeTypes(AttributesBuilder target) {
        Attributes current = target.build();
        writeMimeType(current, target, INPUT_VALUE, INPUT_MIME_TYPE);
        writeMimeType(current, target, OUTPUT_VALUE, OUTPUT_MIME_TYPE);
    }

    private void writeMimeType(Attributes current, AttributesBuilder target, String valueKey, String mimeKey) {
        String value = current.get(AttributeKey.stringKey(valueKey));
        if (value != null && current.get(AttributeKey.stringKey(mimeKey)) == null) {
            set(target, mimeKey, mimeType(value));
        }
    }

    private void setTextIfAbsent(
            Attributes kept, AttributesBuilder target, String valueKey, String mimeKey, String value) {
        if (value == null || kept.get(AttributeKey.stringKey(valueKey)) != null) {
            return;
        }
        set(target, valueKey, value);
        set(target, mimeKey, mimeType(value));
    }

    /** Whether a source attribute carries content that the TraceConfig hides for this span. */
    private boolean isHiddenContent(String key, OpenInferenceSpanKind kind) {
        boolean llm = kind == OpenInferenceSpanKind.LLM;
        // On LLM spans input.value / output.value hold the prompt and completion text, so the
        // narrower message and text flags must hide them too.
        boolean hideInput =
                config.isHideInputs() || (llm && (config.isHideInputMessages() || config.isHideInputText()));
        boolean hideOutput =
                config.isHideOutputs() || (llm && (config.isHideOutputMessages() || config.isHideOutputText()));
        if (hideInput && INPUT_CONTENT_KEYS.contains(key)) return true;
        if (hideOutput && OUTPUT_CONTENT_KEYS.contains(key)) return true;
        if ((config.isHideInputs() || config.isHideToolParameters()) && GEN_AI_TOOL_CALL_ARGUMENTS.equals(key))
            return true;
        if ((config.isHideInputMessages() || config.isHideInputText()) && GEN_AI_INPUT_MESSAGES.equals(key))
            return true;
        if ((config.isHideOutputMessages() || config.isHideOutputText()) && GEN_AI_OUTPUT_MESSAGES.equals(key))
            return true;
        return false;
    }

    /** Mirrors the subset of {@code TracedSpan.shouldHide} that applies to the attributes written here. */
    private boolean shouldHide(String key) {
        if (config.isHideInputs()) {
            if (INPUT_VALUE.equals(key) || INPUT_MIME_TYPE.equals(key)) return true;
            if (key.startsWith(LLM_INPUT_MESSAGES)) return true;
        }
        if (config.isHideOutputs()) {
            if (OUTPUT_VALUE.equals(key) || OUTPUT_MIME_TYPE.equals(key)) return true;
            if (key.startsWith(LLM_OUTPUT_MESSAGES)) return true;
        }
        if (config.isHideInputMessages() && key.startsWith(LLM_INPUT_MESSAGES)) return true;
        if (config.isHideOutputMessages() && key.startsWith(LLM_OUTPUT_MESSAGES)) return true;
        if (config.isHideInputText()
                && key.startsWith(LLM_INPUT_MESSAGES)
                && key.contains(MESSAGE_CONTENT)
                && !key.contains(MESSAGE_CONTENTS)) return true;
        if (config.isHideOutputText()
                && key.startsWith(LLM_OUTPUT_MESSAGES)
                && key.contains(MESSAGE_CONTENT)
                && !key.contains(MESSAGE_CONTENTS)) return true;
        if ((config.isHideInputs() || config.isHideToolParameters()) && TOOL_PARAMETERS.equals(key)) return true;
        return false;
    }

    private void set(AttributesBuilder target, String key, String value) {
        if (value != null && !value.isEmpty() && !shouldHide(key)) {
            target.put(key, value);
        }
    }

    private void set(AttributesBuilder target, String key, Long value) {
        if (value != null && !shouldHide(key)) {
            target.put(key, value);
        }
    }

    private static String provider(Attributes source) {
        String name = string(source, GEN_AI_PROVIDER_NAME);
        if (name == null) {
            name = string(source, GEN_AI_SYSTEM);
        }
        if (name == null) {
            return null;
        }
        String normalized = name.toLowerCase(Locale.ROOT);
        LLMProvider provider = PROVIDERS.get(normalized);
        return provider != null ? provider.getValue() : normalized;
    }

    private static String mimeType(String value) {
        JsonNode node = parse(value);
        return node != null && node.isContainerNode() ? MimeType.JSON.getValue() : MimeType.TEXT.getValue();
    }

    private static JsonNode parse(String json) {
        if (json == null || json.isEmpty()) {
            return null;
        }
        char first = json.charAt(0);
        if (first != '{' && first != '[') {
            return null;
        }
        try {
            return MAPPER.readTree(json);
        } catch (JsonProcessingException e) {
            return null;
        }
    }

    private static String toJson(Object value) {
        try {
            return MAPPER.writeValueAsString(value);
        } catch (JsonProcessingException e) {
            return null;
        }
    }

    private static String text(JsonNode node) {
        return node == null || node.isNull() ? null : node.asText();
    }

    static String string(Attributes attributes, String key) {
        String value = attributes.get(AttributeKey.stringKey(key));
        return value == null || value.isEmpty() ? null : value;
    }

    private static <T extends Number> T number(Attributes attributes, String key, Function<String, T> parser) {
        String value = string(attributes, key);
        if (value != null) {
            try {
                return parser.apply(value.trim());
            } catch (NumberFormatException e) {
                return null;
            }
        }
        Long longValue = attributes.get(AttributeKey.longKey(key));
        if (longValue != null) {
            return parser.apply(longValue.toString());
        }
        Double doubleValue = attributes.get(AttributeKey.doubleKey(key));
        return doubleValue != null ? parser.apply(doubleValue.toString()) : null;
    }

    private static void putIfPresent(Map<String, Object> map, String key, Object value) {
        if (value != null) {
            map.put(key, value);
        }
    }

    @SuppressWarnings("unchecked")
    private static void putUnchecked(AttributesBuilder builder, AttributeKey<?> key, Object value) {
        builder.put((AttributeKey<Object>) key, value);
    }

    /** The original span with the mapped attribute set. */
    private static final class MappedSpanData extends DelegatingSpanData {
        private final Attributes attributes;

        MappedSpanData(SpanData delegate, Attributes attributes) {
            super(delegate);
            this.attributes = attributes;
        }

        @Override
        public Attributes getAttributes() {
            return attributes;
        }

        @Override
        public int getTotalAttributeCount() {
            int dropped = super.getTotalAttributeCount() - super.getAttributes().size();
            return attributes.size() + Math.max(dropped, 0);
        }
    }
}

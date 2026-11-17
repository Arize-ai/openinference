package com.arize.instrumentation.embabel;

import static com.arize.semconv.trace.SemanticConventions.*;
import static org.assertj.core.api.Assertions.assertThat;

import com.arize.instrumentation.TraceConfig;
import io.opentelemetry.api.common.AttributeKey;
import io.opentelemetry.api.common.Attributes;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.sdk.testing.exporter.InMemorySpanExporter;
import io.opentelemetry.sdk.trace.SdkTracerProvider;
import io.opentelemetry.sdk.trace.data.SpanData;
import io.opentelemetry.sdk.trace.export.SimpleSpanProcessor;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

/**
 * Spans are built with the names and string-valued attributes that Embabel's Micrometer
 * observations produce through the OpenTelemetry bridge, then exported through the exporter under
 * test into an in-memory exporter.
 */
class OpenInferenceEmbabelSpanExporterTest {

    private static final String CHAT_INPUT_MESSAGES = "[{\"role\":\"system\",\"parts\":[{\"type\":\"text\","
            + "\"content\":\"You are a travel agent.\"}]},{\"role\":\"user\",\"parts\":[{\"type\":\"text\","
            + "\"content\":\"Find a hotel in Rome\"}]},{\"role\":\"tool\",\"parts\":[{\"type\":\"tool_call_response\","
            + "\"id\":\"call_1\",\"response\":{\"hotels\":2}}]}]";
    private static final String CHAT_OUTPUT_MESSAGES = "[{\"role\":\"assistant\",\"parts\":[{\"type\":\"text\","
            + "\"content\":\"Checking availability.\"},{\"type\":\"tool_call\",\"id\":\"call_2\","
            + "\"name\":\"book_hotel\",\"arguments\":{\"city\":\"Rome\"}}],\"finish_reason\":\"tool_call\"}]";

    private SdkTracerProvider tracerProvider;

    @AfterEach
    void tearDown() {
        if (tracerProvider != null) {
            tracerProvider.close();
        }
    }

    @Test
    void agentRunBecomesAgentSpanWithSession() {
        SpanData span = export(
                TraceConfig.getDefault(),
                "agent TravelAgent",
                Map.of(
                        "embabel.event.type", "agent_process",
                        "gen_ai.operation.name", "agent",
                        "embabel.agent.name", "TravelAgent",
                        "gen_ai.conversation.id", "conv-42",
                        "user.id", "alice",
                        "input.value", "Plan a weekend in Rome",
                        "output.value", "{\"hotel\":\"Roma\"}"));

        Attributes attributes = span.getAttributes();
        assertThat(string(attributes, OPENINFERENCE_SPAN_KIND)).isEqualTo("AGENT");
        assertThat(string(attributes, AGENT_NAME)).isEqualTo("TravelAgent");
        assertThat(string(attributes, SESSION_ID)).isEqualTo("conv-42");
        assertThat(string(attributes, USER_ID)).isEqualTo("alice");
        assertThat(string(attributes, INPUT_VALUE)).isEqualTo("Plan a weekend in Rome");
        assertThat(string(attributes, INPUT_MIME_TYPE)).isEqualTo("text/plain");
        assertThat(string(attributes, OUTPUT_MIME_TYPE)).isEqualTo("application/json");
        // Embabel's own attributes are kept alongside the OpenInference ones.
        assertThat(string(attributes, "embabel.agent.name")).isEqualTo("TravelAgent");
    }

    @Test
    void chatModelSpanBecomesLlmSpan() {
        SpanData span = export(TraceConfig.getDefault(), "chat gpt-4o", chatAttributes());

        Attributes attributes = span.getAttributes();
        assertThat(string(attributes, OPENINFERENCE_SPAN_KIND)).isEqualTo("LLM");
        assertThat(string(attributes, LLM_MODEL_NAME)).isEqualTo("gpt-4o-2024-08-06");
        assertThat(string(attributes, LLM_PROVIDER)).isEqualTo("openai");
        assertThat(attributes.get(AttributeKey.longKey(LLM_TOKEN_COUNT_PROMPT))).isEqualTo(120L);
        assertThat(attributes.get(AttributeKey.longKey(LLM_TOKEN_COUNT_COMPLETION)))
                .isEqualTo(30L);
        assertThat(attributes.get(AttributeKey.longKey(LLM_TOKEN_COUNT_TOTAL))).isEqualTo(150L);
        assertThat(string(attributes, LLM_INVOCATION_PARAMETERS)).isEqualTo("{\"temperature\":0.2,\"max_tokens\":512}");

        assertThat(string(attributes, LLM_INPUT_MESSAGES + ".0." + MESSAGE_ROLE))
                .isEqualTo("system");
        assertThat(string(attributes, LLM_INPUT_MESSAGES + ".1." + MESSAGE_CONTENT))
                .isEqualTo("Find a hotel in Rome");
        assertThat(string(attributes, LLM_INPUT_MESSAGES + ".2." + MESSAGE_ROLE))
                .isEqualTo("tool");
        assertThat(string(attributes, LLM_INPUT_MESSAGES + ".2." + MESSAGE_TOOL_CALL_ID))
                .isEqualTo("call_1");
        assertThat(string(attributes, LLM_INPUT_MESSAGES + ".2." + MESSAGE_CONTENT))
                .isEqualTo("{\"hotels\":2}");

        String output = LLM_OUTPUT_MESSAGES + ".0.";
        String toolCall = output + MESSAGE_TOOL_CALLS + ".0.";
        assertThat(string(attributes, output + MESSAGE_ROLE)).isEqualTo("assistant");
        assertThat(string(attributes, output + MESSAGE_CONTENT)).isEqualTo("Checking availability.");
        assertThat(string(attributes, toolCall + TOOL_CALL_ID)).isEqualTo("call_2");
        assertThat(string(attributes, toolCall + TOOL_CALL_FUNCTION_NAME)).isEqualTo("book_hotel");
        assertThat(string(attributes, toolCall + TOOL_CALL_FUNCTION_ARGUMENTS_JSON))
                .isEqualTo("{\"city\":\"Rome\"}");

        assertThat(string(attributes, INPUT_MIME_TYPE)).isEqualTo("text/plain");
    }

    @Test
    void toolCallBecomesToolSpan() {
        SpanData span = export(
                TraceConfig.getDefault(),
                "execute_tool book_hotel",
                Map.of(
                        "embabel.event.type", "tool_call",
                        "gen_ai.operation.name", "execute_tool",
                        "gen_ai.tool.name", "book_hotel",
                        "gen_ai.tool.description", "Books a hotel room",
                        "gen_ai.tool.call.arguments", "{\"city\":\"Rome\"}",
                        "gen_ai.tool.call.result", "Booked: Hotel Roma"));

        Attributes attributes = span.getAttributes();
        assertThat(string(attributes, OPENINFERENCE_SPAN_KIND)).isEqualTo("TOOL");
        assertThat(string(attributes, TOOL_NAME)).isEqualTo("book_hotel");
        assertThat(string(attributes, TOOL_DESCRIPTION)).isEqualTo("Books a hotel room");
        assertThat(string(attributes, TOOL_PARAMETERS)).isEqualTo("{\"city\":\"Rome\"}");
        assertThat(string(attributes, INPUT_VALUE)).isEqualTo("{\"city\":\"Rome\"}");
        assertThat(string(attributes, INPUT_MIME_TYPE)).isEqualTo("application/json");
        assertThat(string(attributes, OUTPUT_VALUE)).isEqualTo("Booked: Hotel Roma");
        assertThat(string(attributes, OUTPUT_MIME_TYPE)).isEqualTo("text/plain");
    }

    @Test
    void embabelStepsBecomeChainEmbeddingAndRetrieverSpans() {
        assertThat(kindOf("action FindHotel", Map.of("embabel.event.type", "action")))
                .isEqualTo("CHAIN");
        assertThat(kindOf("llm gpt-4o", Map.of("embabel.event.type", "llm_call")))
                .isEqualTo("CHAIN");
        assertThat(kindOf("tool-loop", Map.of("embabel.event.type", "tool_loop")))
                .isEqualTo("CHAIN");

        SpanData embedding = export(
                TraceConfig.getDefault(),
                "embeddings text-embedding-3-small",
                Map.of(
                        "embabel.event.type", "embedding",
                        "gen_ai.operation.name", "embeddings",
                        "gen_ai.request.model", "text-embedding-3-small"));
        assertThat(string(embedding.getAttributes(), OPENINFERENCE_SPAN_KIND)).isEqualTo("EMBEDDING");
        assertThat(string(embedding.getAttributes(), EMBEDDING_MODEL_NAME)).isEqualTo("text-embedding-3-small");

        SpanData rag = export(
                TraceConfig.getDefault(),
                "rag pgvector",
                Map.of("embabel.event.type", "rag", "embabel.rag.query", "hotels near the Colosseum"));
        assertThat(string(rag.getAttributes(), OPENINFERENCE_SPAN_KIND)).isEqualTo("RETRIEVER");
        assertThat(string(rag.getAttributes(), INPUT_VALUE)).isEqualTo("hotels near the Colosseum");
    }

    @Test
    void otherSpansPassThroughUnchanged() {
        SpanData http = export(TraceConfig.getDefault(), "GET /hotels", Map.of("http.request.method", "GET"));
        assertThat(http.getAttributes().size()).isEqualTo(1);
        assertThat(string(http.getAttributes(), OPENINFERENCE_SPAN_KIND)).isNull();

        SpanData alreadyMapped = export(
                TraceConfig.getDefault(),
                "chat gpt-4o",
                Map.of(OPENINFERENCE_SPAN_KIND, "LLM", "gen_ai.operation.name", "chat"));
        assertThat(alreadyMapped.getAttributes().size()).isEqualTo(2);
    }

    @Test
    void hideInputsRemovesInputContentFromEverySource() {
        TraceConfig config = TraceConfig.builder().hideInputs(true).build();

        Attributes llm = export(config, "chat gpt-4o", chatAttributes()).getAttributes();
        assertThat(string(llm, OPENINFERENCE_SPAN_KIND)).isEqualTo("LLM");
        assertThat(string(llm, INPUT_VALUE)).isNull();
        assertThat(string(llm, "gen_ai.input.messages")).isNull();
        assertThat(keysStartingWith(llm, LLM_INPUT_MESSAGES)).isEmpty();
        assertThat(string(llm, OUTPUT_VALUE)).isEqualTo("Checking availability.");

        Attributes tool = export(
                        config,
                        "execute_tool book_hotel",
                        Map.of(
                                "embabel.event.type", "tool_call",
                                "gen_ai.tool.name", "book_hotel",
                                "gen_ai.tool.call.arguments", "{\"city\":\"Rome\"}"))
                .getAttributes();
        assertThat(string(tool, TOOL_NAME)).isEqualTo("book_hotel");
        assertThat(string(tool, INPUT_VALUE)).isNull();
        assertThat(string(tool, TOOL_PARAMETERS)).isNull();
        assertThat(string(tool, "gen_ai.tool.call.arguments")).isNull();
    }

    @Test
    void hideToolParametersAlsoHidesTheArgumentsInInputValue() {
        TraceConfig config = TraceConfig.builder().hideToolParameters(true).build();

        Attributes tool = export(
                        config,
                        "execute_tool book_hotel",
                        Map.of(
                                "embabel.event.type", "tool_call",
                                "gen_ai.tool.name", "book_hotel",
                                "gen_ai.tool.call.arguments", "{\"city\":\"Rome\"}",
                                "gen_ai.tool.call.result", "Booked"))
                .getAttributes();
        assertThat(string(tool, TOOL_PARAMETERS)).isNull();
        assertThat(string(tool, INPUT_VALUE)).isNull();
        assertThat(string(tool, "gen_ai.tool.call.arguments")).isNull();
        assertThat(string(tool, OUTPUT_VALUE)).isEqualTo("Booked");
    }

    @Test
    void hideInputMessagesAlsoHidesThePromptInInputValue() {
        TraceConfig config = TraceConfig.builder().hideInputMessages(true).build();

        Attributes llm = export(config, "chat gpt-4o", chatAttributes()).getAttributes();
        assertThat(keysStartingWith(llm, LLM_INPUT_MESSAGES)).isEmpty();
        assertThat(string(llm, "gen_ai.input.messages")).isNull();
        assertThat(string(llm, INPUT_VALUE)).isNull();
        assertThat(keysStartingWith(llm, LLM_OUTPUT_MESSAGES)).isNotEmpty();
    }

    @Test
    void hideOutputsRemovesOutputContentFromEverySource() {
        TraceConfig config = TraceConfig.builder().hideOutputs(true).build();

        Attributes llm = export(config, "chat gpt-4o", chatAttributes()).getAttributes();
        assertThat(string(llm, OUTPUT_VALUE)).isNull();
        assertThat(string(llm, "gen_ai.output.messages")).isNull();
        assertThat(keysStartingWith(llm, LLM_OUTPUT_MESSAGES)).isEmpty();
        assertThat(string(llm, INPUT_VALUE)).isEqualTo("Find a hotel in Rome");

        Attributes agent = export(
                        config,
                        "agent TravelAgent",
                        Map.of(
                                "embabel.event.type", "agent_process",
                                "embabel.agent.result", "Hotel Roma",
                                "output.value", "Hotel Roma"))
                .getAttributes();
        assertThat(string(agent, OUTPUT_VALUE)).isNull();
        assertThat(string(agent, "embabel.agent.result")).isNull();
    }

    @Test
    void malformedMessagesDoNotStopTheSpanFromBeingMapped() {
        SpanData span = export(
                TraceConfig.getDefault(),
                "chat gpt-4o",
                Map.of(
                        "gen_ai.operation.name", "chat",
                        "gen_ai.request.model", "gpt-4o",
                        "gen_ai.usage.input_tokens", "not-a-number",
                        "gen_ai.input.messages", "[{\"role\":"));

        Attributes attributes = span.getAttributes();
        assertThat(string(attributes, OPENINFERENCE_SPAN_KIND)).isEqualTo("LLM");
        assertThat(string(attributes, LLM_MODEL_NAME)).isEqualTo("gpt-4o");
        assertThat(attributes.get(AttributeKey.longKey(LLM_TOKEN_COUNT_PROMPT))).isNull();
        assertThat(keysStartingWith(attributes, LLM_INPUT_MESSAGES)).isEmpty();
    }

    @Test
    void suppressTracingLeavesSpansUnchanged() {
        TraceConfig config = TraceConfig.builder().suppressTracing(true).build();
        SpanData span = export(config, "agent TravelAgent", Map.of("embabel.event.type", "agent_process"));
        assertThat(string(span.getAttributes(), OPENINFERENCE_SPAN_KIND)).isNull();
    }

    private static Map<String, String> chatAttributes() {
        return Map.ofEntries(
                Map.entry("gen_ai.operation.name", "chat"),
                Map.entry("gen_ai.provider.name", "openai"),
                Map.entry("gen_ai.request.model", "gpt-4o"),
                Map.entry("gen_ai.response.model", "gpt-4o-2024-08-06"),
                Map.entry("gen_ai.request.temperature", "0.2"),
                Map.entry("gen_ai.request.max_tokens", "512"),
                Map.entry("gen_ai.usage.input_tokens", "120"),
                Map.entry("gen_ai.usage.output_tokens", "30"),
                Map.entry("gen_ai.input.messages", CHAT_INPUT_MESSAGES),
                Map.entry("gen_ai.output.messages", CHAT_OUTPUT_MESSAGES),
                Map.entry("input.value", "Find a hotel in Rome"),
                Map.entry("output.value", "Checking availability."));
    }

    private String kindOf(String name, Map<String, String> attributes) {
        return string(export(TraceConfig.getDefault(), name, attributes).getAttributes(), OPENINFERENCE_SPAN_KIND);
    }

    private SpanData export(TraceConfig config, String name, Map<String, String> attributes) {
        if (tracerProvider != null) {
            tracerProvider.close();
        }
        InMemorySpanExporter exported = InMemorySpanExporter.create();
        tracerProvider = SdkTracerProvider.builder()
                .addSpanProcessor(SimpleSpanProcessor.create(new OpenInferenceEmbabelSpanExporter(exported, config)))
                .build();
        Span span =
                tracerProvider.get("io.micrometer.tracing").spanBuilder(name).startSpan();
        attributes.forEach(span::setAttribute);
        span.end();
        List<SpanData> spans = exported.getFinishedSpanItems();
        assertThat(spans).hasSize(1);
        return spans.get(0);
    }

    private static String string(Attributes attributes, String key) {
        return attributes.get(AttributeKey.stringKey(key));
    }

    private static List<String> keysStartingWith(Attributes attributes, String prefix) {
        return attributes.asMap().keySet().stream()
                .map(AttributeKey::getKey)
                .filter(key -> key.startsWith(prefix))
                .toList();
    }
}

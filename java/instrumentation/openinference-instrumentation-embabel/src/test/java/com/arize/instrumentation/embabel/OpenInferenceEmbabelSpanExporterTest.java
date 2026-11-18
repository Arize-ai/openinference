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
import java.util.HashMap;
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
        Map<String, Object> attributes = attributes(export(
                TraceConfig.getDefault(),
                "agent TravelAgent",
                Map.of(
                        "embabel.event.type", "agent_process",
                        "gen_ai.operation.name", "agent",
                        "embabel.agent.name", "TravelAgent",
                        "gen_ai.conversation.id", "conv-42",
                        "user.id", "alice",
                        "input.value", "Plan a weekend in Rome",
                        "output.value", "{\"hotel\":\"Roma\"}")));

        assertThat(attributes.remove(OPENINFERENCE_SPAN_KIND)).isEqualTo("AGENT");
        assertThat(attributes.remove(AGENT_NAME)).isEqualTo("TravelAgent");
        assertThat(attributes.remove(SESSION_ID)).isEqualTo("conv-42");
        assertThat(attributes.remove(USER_ID)).isEqualTo("alice");
        assertThat(attributes.remove(INPUT_VALUE)).isEqualTo("Plan a weekend in Rome");
        assertThat(attributes.remove(INPUT_MIME_TYPE)).isEqualTo("text/plain");
        assertThat(attributes.remove(OUTPUT_VALUE)).isEqualTo("{\"hotel\":\"Roma\"}");
        assertThat(attributes.remove(OUTPUT_MIME_TYPE)).isEqualTo("application/json");
        // Embabel's own attributes are kept alongside the OpenInference ones.
        assertThat(attributes.remove("embabel.event.type")).isEqualTo("agent_process");
        assertThat(attributes.remove("gen_ai.operation.name")).isEqualTo("agent");
        assertThat(attributes.remove("embabel.agent.name")).isEqualTo("TravelAgent");
        assertThat(attributes.remove("gen_ai.conversation.id")).isEqualTo("conv-42");
        assertThat(attributes).isEmpty();
    }

    @Test
    void chatModelSpanBecomesLlmSpan() {
        Map<String, Object> attributes = attributes(export(TraceConfig.getDefault(), "chat gpt-4o", chatAttributes()));

        assertThat(attributes.remove(OPENINFERENCE_SPAN_KIND)).isEqualTo("LLM");
        assertThat(attributes.remove(LLM_MODEL_NAME)).isEqualTo("gpt-4o-2024-08-06");
        assertThat(attributes.remove(LLM_PROVIDER)).isEqualTo("openai");
        assertThat(attributes.remove(LLM_TOKEN_COUNT_PROMPT)).isEqualTo(120L);
        assertThat(attributes.remove(LLM_TOKEN_COUNT_COMPLETION)).isEqualTo(30L);
        assertThat(attributes.remove(LLM_TOKEN_COUNT_TOTAL)).isEqualTo(150L);
        assertThat(attributes.remove(LLM_INVOCATION_PARAMETERS)).isEqualTo("{\"temperature\":0.2,\"max_tokens\":512}");

        String input = LLM_INPUT_MESSAGES + ".";
        assertThat(attributes.remove(input + "0." + MESSAGE_ROLE)).isEqualTo("system");
        assertThat(attributes.remove(input + "0." + MESSAGE_CONTENT)).isEqualTo("You are a travel agent.");
        assertThat(attributes.remove(input + "1." + MESSAGE_ROLE)).isEqualTo("user");
        assertThat(attributes.remove(input + "1." + MESSAGE_CONTENT)).isEqualTo("Find a hotel in Rome");
        assertThat(attributes.remove(input + "2." + MESSAGE_ROLE)).isEqualTo("tool");
        assertThat(attributes.remove(input + "2." + MESSAGE_TOOL_CALL_ID)).isEqualTo("call_1");
        assertThat(attributes.remove(input + "2." + MESSAGE_CONTENT)).isEqualTo("{\"hotels\":2}");

        String output = LLM_OUTPUT_MESSAGES + ".0.";
        String toolCall = output + MESSAGE_TOOL_CALLS + ".0.";
        assertThat(attributes.remove(output + MESSAGE_ROLE)).isEqualTo("assistant");
        assertThat(attributes.remove(output + MESSAGE_CONTENT)).isEqualTo("Checking availability.");
        assertThat(attributes.remove(toolCall + TOOL_CALL_ID)).isEqualTo("call_2");
        assertThat(attributes.remove(toolCall + TOOL_CALL_FUNCTION_NAME)).isEqualTo("book_hotel");
        assertThat(attributes.remove(toolCall + TOOL_CALL_FUNCTION_ARGUMENTS_JSON))
                .isEqualTo("{\"city\":\"Rome\"}");

        assertThat(attributes.remove(INPUT_VALUE)).isEqualTo("Find a hotel in Rome");
        assertThat(attributes.remove(INPUT_MIME_TYPE)).isEqualTo("text/plain");
        assertThat(attributes.remove(OUTPUT_VALUE)).isEqualTo("Checking availability.");
        assertThat(attributes.remove(OUTPUT_MIME_TYPE)).isEqualTo("text/plain");
        // The source GenAI attributes are kept.
        chatAttributes().keySet().forEach(attributes::remove);
        assertThat(attributes).isEmpty();
    }

    @Test
    void toolCallBecomesToolSpan() {
        Map<String, String> source = Map.of(
                "embabel.event.type", "tool_call",
                "gen_ai.operation.name", "execute_tool",
                "gen_ai.tool.name", "book_hotel",
                "gen_ai.tool.description", "Books a hotel room",
                "gen_ai.tool.call.arguments", "{\"city\":\"Rome\"}",
                "gen_ai.tool.call.result", "Booked: Hotel Roma");
        Map<String, Object> attributes =
                attributes(export(TraceConfig.getDefault(), "execute_tool book_hotel", source));

        assertThat(attributes.remove(OPENINFERENCE_SPAN_KIND)).isEqualTo("TOOL");
        assertThat(attributes.remove(TOOL_NAME)).isEqualTo("book_hotel");
        assertThat(attributes.remove(TOOL_DESCRIPTION)).isEqualTo("Books a hotel room");
        assertThat(attributes.remove(TOOL_PARAMETERS)).isEqualTo("{\"city\":\"Rome\"}");
        assertThat(attributes.remove(INPUT_VALUE)).isEqualTo("{\"city\":\"Rome\"}");
        assertThat(attributes.remove(INPUT_MIME_TYPE)).isEqualTo("application/json");
        assertThat(attributes.remove(OUTPUT_VALUE)).isEqualTo("Booked: Hotel Roma");
        assertThat(attributes.remove(OUTPUT_MIME_TYPE)).isEqualTo("text/plain");
        source.keySet().forEach(attributes::remove);
        assertThat(attributes).isEmpty();
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

        Map<String, Object> llm = attributes(export(config, "chat gpt-4o", chatAttributes()));
        assertThat(llm.remove(OPENINFERENCE_SPAN_KIND)).isEqualTo("LLM");
        assertThat(llm.remove(OUTPUT_VALUE)).isEqualTo("Checking availability.");
        assertThat(llm.remove(OUTPUT_MIME_TYPE)).isEqualTo("text/plain");
        assertThat(llm.remove("gen_ai.output.messages")).isEqualTo(CHAT_OUTPUT_MESSAGES);
        removeLlmMetadata(llm);
        removeKeysStartingWith(llm, LLM_OUTPUT_MESSAGES);
        assertThat(llm).isEmpty();

        Map<String, Object> tool = attributes(export(
                config,
                "execute_tool book_hotel",
                Map.of(
                        "embabel.event.type", "tool_call",
                        "gen_ai.tool.name", "book_hotel",
                        "gen_ai.tool.call.arguments", "{\"city\":\"Rome\"}")));
        assertThat(tool.remove(OPENINFERENCE_SPAN_KIND)).isEqualTo("TOOL");
        assertThat(tool.remove(TOOL_NAME)).isEqualTo("book_hotel");
        assertThat(tool.remove("embabel.event.type")).isEqualTo("tool_call");
        assertThat(tool.remove("gen_ai.tool.name")).isEqualTo("book_hotel");
        assertThat(tool).isEmpty();
    }

    @Test
    void hideInputsRemovesTheBasisOfADynamicAgent() {
        Map<String, String> source = Map.of(
                "embabel.event.type", "dynamic_agent_creation",
                "gen_ai.operation.name", "dynamic_agent_creation",
                "embabel.agent.name", "TripPlanner",
                "embabel.dynamic_agent.basis", "Plan a weekend in Rome for my family");

        Map<String, Object> shown = attributes(export(TraceConfig.getDefault(), "dynamic_agent TripPlanner", source));
        assertThat(shown.remove("embabel.dynamic_agent.basis")).isEqualTo("Plan a weekend in Rome for my family");

        TraceConfig config = TraceConfig.builder().hideInputs(true).build();
        Map<String, Object> hidden = attributes(export(config, "dynamic_agent TripPlanner", source));
        assertThat(hidden.remove(OPENINFERENCE_SPAN_KIND)).isEqualTo("CHAIN");
        assertThat(hidden.remove("embabel.event.type")).isEqualTo("dynamic_agent_creation");
        assertThat(hidden.remove("gen_ai.operation.name")).isEqualTo("dynamic_agent_creation");
        assertThat(hidden.remove("embabel.agent.name")).isEqualTo("TripPlanner");
        assertThat(hidden).isEmpty();
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

        Map<String, Object> llm = attributes(export(config, "chat gpt-4o", chatAttributes()));
        assertThat(llm.remove(OPENINFERENCE_SPAN_KIND)).isEqualTo("LLM");
        assertThat(llm.remove(INPUT_VALUE)).isEqualTo("Find a hotel in Rome");
        assertThat(llm.remove(INPUT_MIME_TYPE)).isEqualTo("text/plain");
        assertThat(llm.remove("gen_ai.input.messages")).isEqualTo(CHAT_INPUT_MESSAGES);
        removeLlmMetadata(llm);
        removeKeysStartingWith(llm, LLM_INPUT_MESSAGES);
        assertThat(llm).isEmpty();

        Map<String, Object> agent = attributes(export(
                config,
                "agent TravelAgent",
                Map.of(
                        "embabel.event.type", "agent_process",
                        "embabel.agent.result", "Hotel Roma",
                        "output.value", "Hotel Roma")));
        assertThat(agent.remove(OPENINFERENCE_SPAN_KIND)).isEqualTo("AGENT");
        assertThat(agent.remove("embabel.event.type")).isEqualTo("agent_process");
        assertThat(agent).isEmpty();

        Map<String, Object> replan = attributes(export(
                config,
                "replan",
                Map.of("embabel.event.type", "replan", "embabel.replan.reason", "Hotel Roma is full")));
        assertThat(replan.remove(OPENINFERENCE_SPAN_KIND)).isEqualTo("CHAIN");
        assertThat(replan.remove("embabel.event.type")).isEqualTo("replan");
        assertThat(replan).isEmpty();
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

    /** Removes the LLM attributes that no hide flag covers: model, provider, tokens, parameters, sources. */
    private static void removeLlmMetadata(Map<String, Object> attributes) {
        assertThat(attributes.remove(LLM_MODEL_NAME)).isEqualTo("gpt-4o-2024-08-06");
        assertThat(attributes.remove(LLM_PROVIDER)).isEqualTo("openai");
        assertThat(attributes.remove(LLM_TOKEN_COUNT_PROMPT)).isEqualTo(120L);
        assertThat(attributes.remove(LLM_TOKEN_COUNT_COMPLETION)).isEqualTo(30L);
        assertThat(attributes.remove(LLM_TOKEN_COUNT_TOTAL)).isEqualTo(150L);
        assertThat(attributes.remove(LLM_INVOCATION_PARAMETERS)).isNotNull();
        for (String key : List.of(
                "gen_ai.operation.name",
                "gen_ai.provider.name",
                "gen_ai.request.model",
                "gen_ai.response.model",
                "gen_ai.request.temperature",
                "gen_ai.request.max_tokens",
                "gen_ai.usage.input_tokens",
                "gen_ai.usage.output_tokens")) {
            assertThat(attributes.remove(key)).as(key).isNotNull();
        }
    }

    private static void removeKeysStartingWith(Map<String, Object> attributes, String prefix) {
        assertThat(attributes.keySet().removeIf(key -> key.startsWith(prefix)))
                .as(prefix)
                .isTrue();
    }

    private static Map<String, Object> attributes(SpanData span) {
        Map<String, Object> attributes = new HashMap<>();
        span.getAttributes().forEach((key, value) -> attributes.put(key.getKey(), value));
        return attributes;
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

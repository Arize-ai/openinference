package com.arize.instrumentation.embabel;

import static com.arize.semconv.trace.SemanticConventions.*;
import static org.assertj.core.api.Assertions.assertThat;

import com.arize.instrumentation.ContextAttributes;
import com.arize.instrumentation.SuppressTracing;
import com.arize.instrumentation.TraceConfig;
import io.opentelemetry.api.common.Attributes;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.context.Scope;
import io.opentelemetry.sdk.testing.exporter.InMemorySpanExporter;
import io.opentelemetry.sdk.trace.SdkTracerProvider;
import io.opentelemetry.sdk.trace.export.SimpleSpanProcessor;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class OpenInferenceEmbabelSpanProcessorTest {

    private static final Map<String, String> AGENT_ATTRIBUTES =
            Map.of("embabel.event.type", "agent_process", "gen_ai.conversation.id", "conv-42");

    private static final Map<String, String> CHAT_ATTRIBUTES = Map.of(
            "gen_ai.operation.name", "chat",
            "gen_ai.request.model", "gpt-4o",
            "gen_ai.input.messages",
                    "[{\"role\":\"user\",\"parts\":[{\"type\":\"text\",\"content\":\"Find a hotel in Rome\"}]}]");

    private InMemorySpanExporter exported;
    private SdkTracerProvider tracerProvider;

    @BeforeEach
    void setUp() {
        exported = InMemorySpanExporter.create();
    }

    @AfterEach
    void tearDown() {
        tracerProvider.close();
    }

    @Test
    void copiesContextAttributesOntoEmbabelSpans() {
        Map<String, Object> attributes = run(TraceConfig.getDefault(), false, "agent TravelAgent", AGENT_ATTRIBUTES);

        // The session from the context wins over Embabel's conversation id.
        assertThat(attributes.remove(SESSION_ID)).isEqualTo("session-7");
        assertThat(attributes.remove(USER_ID)).isEqualTo("bob");
        assertThat(attributes.remove(METADATA)).isEqualTo("{\"team\":\"travel\"}");
        assertThat(attributes.remove(TAG_TAGS)).isEqualTo(List.of("beta"));
        assertThat(attributes.remove(OPENINFERENCE_SPAN_KIND)).isEqualTo("AGENT");
        assertThat(attributes.remove("embabel.event.type")).isEqualTo("agent_process");
        assertThat(attributes.remove("gen_ai.conversation.id")).isEqualTo("conv-42");
        assertThat(attributes).isEmpty();
    }

    @Test
    void suppressedAgentSpanIsExportedUnmapped() {
        Map<String, Object> attributes = run(TraceConfig.getDefault(), true, "agent TravelAgent", AGENT_ATTRIBUTES);

        // Neither the context attributes nor the exporter's mapping are applied, and the internal
        // marker is removed before export.
        assertThat(attributes.remove("embabel.event.type")).isEqualTo("agent_process");
        assertThat(attributes.remove("gen_ai.conversation.id")).isEqualTo("conv-42");
        assertThat(attributes).isEmpty();
    }

    @Test
    void suppressedChatSpanIsExportedUnmapped() {
        Map<String, Object> attributes = run(TraceConfig.getDefault(), true, "chat gpt-4o", CHAT_ATTRIBUTES);

        assertThat(attributes).containsOnlyKeys(CHAT_ATTRIBUTES.keySet());
    }

    private Map<String, Object> run(
            TraceConfig config, boolean suppressed, String name, Map<String, String> spanAttributes) {
        tracerProvider = SdkTracerProvider.builder()
                .addSpanProcessor(new OpenInferenceEmbabelSpanProcessor(config))
                .addSpanProcessor(SimpleSpanProcessor.create(new OpenInferenceEmbabelSpanExporter(exported, config)))
                .build();
        try (Scope context = ContextAttributes.builder()
                        .sessionId("session-7")
                        .userId("bob")
                        .metadata("{\"team\":\"travel\"}")
                        .tags(List.of("beta"))
                        .build();
                Scope suppression = suppressed ? SuppressTracing.begin() : Scope.noop()) {
            Span span = tracerProvider
                    .get("io.micrometer.tracing")
                    .spanBuilder(name)
                    .startSpan();
            spanAttributes.forEach(span::setAttribute);
            span.end();
        }
        assertThat(exported.getFinishedSpanItems()).hasSize(1);
        return toMap(exported.getFinishedSpanItems().get(0).getAttributes());
    }

    private static Map<String, Object> toMap(Attributes attributes) {
        Map<String, Object> map = new HashMap<>();
        attributes.forEach((key, value) -> map.put(key.getKey(), value));
        return map;
    }
}

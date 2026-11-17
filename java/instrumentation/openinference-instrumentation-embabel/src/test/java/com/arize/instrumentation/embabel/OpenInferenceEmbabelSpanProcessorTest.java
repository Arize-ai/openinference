package com.arize.instrumentation.embabel;

import static com.arize.semconv.trace.SemanticConventions.*;
import static org.assertj.core.api.Assertions.assertThat;

import com.arize.instrumentation.ContextAttributes;
import com.arize.instrumentation.SuppressTracing;
import com.arize.instrumentation.TraceConfig;
import io.opentelemetry.api.common.AttributeKey;
import io.opentelemetry.api.common.Attributes;
import io.opentelemetry.context.Scope;
import io.opentelemetry.sdk.testing.exporter.InMemorySpanExporter;
import io.opentelemetry.sdk.trace.SdkTracerProvider;
import io.opentelemetry.sdk.trace.export.SimpleSpanProcessor;
import java.util.List;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class OpenInferenceEmbabelSpanProcessorTest {

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
        Attributes attributes = runAgentSpan(TraceConfig.getDefault(), false);

        // The session from the context wins over Embabel's conversation id.
        assertThat(attributes.get(AttributeKey.stringKey(SESSION_ID))).isEqualTo("session-7");
        assertThat(attributes.get(AttributeKey.stringKey(USER_ID))).isEqualTo("bob");
        assertThat(attributes.get(AttributeKey.stringKey(METADATA))).isEqualTo("{\"team\":\"travel\"}");
        assertThat(attributes.get(AttributeKey.stringArrayKey(TAG_TAGS))).containsExactly("beta");
        assertThat(attributes.get(AttributeKey.stringKey(OPENINFERENCE_SPAN_KIND)))
                .isEqualTo("AGENT");
    }

    @Test
    void copiesNothingWhileTracingIsSuppressed() {
        Attributes attributes = runAgentSpan(TraceConfig.getDefault(), true);

        assertThat(attributes.get(AttributeKey.stringKey(SESSION_ID))).isEqualTo("conv-42");
        assertThat(attributes.get(AttributeKey.stringKey(USER_ID))).isNull();
        assertThat(attributes.get(AttributeKey.stringArrayKey(TAG_TAGS))).isNull();
    }

    private Attributes runAgentSpan(TraceConfig config, boolean suppressed) {
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
            tracerProvider
                    .get("io.micrometer.tracing")
                    .spanBuilder("agent TravelAgent")
                    .setAttribute("embabel.event.type", "agent_process")
                    .setAttribute("gen_ai.conversation.id", "conv-42")
                    .startSpan()
                    .end();
        }
        assertThat(exported.getFinishedSpanItems()).hasSize(1);
        return exported.getFinishedSpanItems().get(0).getAttributes();
    }
}

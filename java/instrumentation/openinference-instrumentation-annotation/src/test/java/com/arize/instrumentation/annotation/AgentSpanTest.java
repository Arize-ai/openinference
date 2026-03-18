package com.arize.instrumentation.annotation;

import static org.assertj.core.api.Assertions.assertThat;

import com.arize.instrumentation.OITracer;
import com.arize.semconv.trace.SemanticConventions;
import io.opentelemetry.api.common.AttributeKey;
import io.opentelemetry.sdk.testing.exporter.InMemorySpanExporter;
import io.opentelemetry.sdk.trace.SdkTracerProvider;
import io.opentelemetry.sdk.trace.data.SpanData;
import io.opentelemetry.sdk.trace.export.SimpleSpanProcessor;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class AgentSpanTest {

    private InMemorySpanExporter exporter;
    private OITracer tracer;

    @BeforeEach
    void setUp() {
        exporter = InMemorySpanExporter.create();
        SdkTracerProvider provider = SdkTracerProvider.builder()
                .addSpanProcessor(SimpleSpanProcessor.create(exporter))
                .build();
        tracer = new OITracer(provider.get("test"));
    }

    @Test
    void setsAgentSpanKindAndName() {
        try (AgentSpan span = AgentSpan.start(tracer, "test-agent")) {
            span.setAgentName("qa-agent");
        }

        SpanData data = exporter.getFinishedSpanItems().get(0);
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.OPENINFERENCE_SPAN_KIND)))
                .isEqualTo("AGENT");
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.AGENT_NAME)))
                .isEqualTo("qa-agent");
    }
}

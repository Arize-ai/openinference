package com.arize.instrumentation.trace;

import static org.assertj.core.api.Assertions.assertThat;

import com.arize.instrumentation.OITracer;
import io.opentelemetry.api.common.AttributeKey;
import io.opentelemetry.sdk.testing.exporter.InMemorySpanExporter;
import io.opentelemetry.sdk.trace.SdkTracerProvider;
import io.opentelemetry.sdk.trace.data.SpanData;
import io.opentelemetry.sdk.trace.export.SimpleSpanProcessor;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class TracedSpanAttributeTest {

    private InMemorySpanExporter exporter;
    private OITracer tracer;

    @BeforeEach
    void setUp() {
        exporter = InMemorySpanExporter.create();
        SdkTracerProvider provider = SdkTracerProvider.builder()
                .addSpanProcessor(SimpleSpanProcessor.create(exporter))
                .build();
        tracer = new OITracer(provider.get("typed-attributes"));
    }

    @AfterEach
    void tearDown() {
        exporter.reset();
    }

    @Test
    void setsNumericAndBooleanAttributesWithNativeTypes() {
        try (AgentSpan span = AgentSpan.start(tracer, "attrs")) {
            span.setAttribute("inputs.count", 42);
            span.setAttribute("token.total", 123L);
            span.setAttribute("cost", 0.5d);
            span.setAttribute("flag", true);
        }

        SpanData data = exporter.getFinishedSpanItems().get(0);
        assertThat(data.getAttributes().get(AttributeKey.longKey("inputs.count")))
                .isEqualTo(42L);
        assertThat(data.getAttributes().get(AttributeKey.longKey("token.total")))
                .isEqualTo(123L);
        assertThat(data.getAttributes().get(AttributeKey.doubleKey("cost"))).isEqualTo(0.5d);
        assertThat(data.getAttributes().get(AttributeKey.booleanKey("flag"))).isTrue();
    }

    @Test
    void setsStringCollectionAttributes() {
        try (AgentSpan span = AgentSpan.start(tracer, "list")) {
            span.setAttribute("tags", List.of("one", "two"));
            span.setAttribute("alt", new String[] {"a", "b"});
        }

        SpanData data = exporter.getFinishedSpanItems().get(0);
        assertThat(data.getAttributes().get(AttributeKey.stringArrayKey("tags")))
                .containsExactly("one", "two");
        assertThat(data.getAttributes().get(AttributeKey.stringArrayKey("alt"))).containsExactly("a", "b");
    }

    @Test
    void fallsBackToJsonForComplexObjects() {
        Map<String, Object> complex = Map.of("usage", Map.of("totalTokens", 150));

        try (AgentSpan span = AgentSpan.start(tracer, "map")) {
            span.setAttribute("metadata", complex);
        }

        SpanData data = exporter.getFinishedSpanItems().get(0);
        String stored = data.getAttributes().get(AttributeKey.stringKey("metadata"));
        assertThat(stored).contains("usage").contains("totalTokens");
    }
}

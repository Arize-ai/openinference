package com.arize.instrumentation.annotations;

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

class TracedToolSpanTest {

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
    void setsToolSpanKindAndAttributes() {
        try (TracedToolSpan span = TracedToolSpan.start(tracer, "test-tool")) {
            span.setToolName("weather");
            span.setToolDescription("Gets current weather");
        }

        SpanData data = exporter.getFinishedSpanItems().get(0);
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.OPENINFERENCE_SPAN_KIND)))
                .isEqualTo("TOOL");
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.TOOL_NAME)))
                .isEqualTo("weather");
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.TOOL_DESCRIPTION)))
                .isEqualTo("Gets current weather");
    }
}

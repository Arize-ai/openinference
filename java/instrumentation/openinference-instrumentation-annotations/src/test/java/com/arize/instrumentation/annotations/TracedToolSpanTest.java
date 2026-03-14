package com.arize.instrumentation.annotations;

import static org.assertj.core.api.Assertions.assertThat;

import com.arize.instrumentation.OITracer;
import com.arize.instrumentation.TraceConfig;
import com.arize.semconv.trace.SemanticConventions;
import io.opentelemetry.api.common.AttributeKey;
import io.opentelemetry.sdk.testing.exporter.InMemorySpanExporter;
import io.opentelemetry.sdk.trace.SdkTracerProvider;
import io.opentelemetry.sdk.trace.data.SpanData;
import io.opentelemetry.sdk.trace.export.SimpleSpanProcessor;
import java.util.Map;
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

    @Test
    void setsToolParameters() {
        try (TracedToolSpan span = TracedToolSpan.start(tracer, "params-tool")) {
            span.setToolParameters(Map.of("location", "San Francisco", "unit", "celsius"));
        }

        SpanData data = exporter.getFinishedSpanItems().get(0);
        String params = data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.TOOL_PARAMETERS));
        assertThat(params).isNotNull();
        assertThat(params).contains("San Francisco");
        assertThat(params).contains("celsius");
    }

    @Test
    void hideToolParametersRespected() {
        TraceConfig config = TraceConfig.builder().hideToolParameters(true).build();
        OITracer maskedTracer = new OITracer(
                SdkTracerProvider.builder()
                        .addSpanProcessor(SimpleSpanProcessor.create(exporter))
                        .build()
                        .get("test"),
                config);

        try (TracedToolSpan span = TracedToolSpan.start(maskedTracer, "hidden-params")) {
            span.setToolParameters(Map.of("secret", "value"));
        }

        SpanData data = exporter.getFinishedSpanItems().get(0);
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.TOOL_PARAMETERS)))
                .isNull();
    }

    @Test
    void setsToolJsonSchema() {
        String schema = "{\"type\": \"object\", \"properties\": {\"location\": {\"type\": \"string\"}}}";
        try (TracedToolSpan span = TracedToolSpan.start(tracer, "schema-tool")) {
            span.setToolJsonSchema(schema);
        }

        SpanData data = exporter.getFinishedSpanItems().get(0);
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.TOOL_JSON_SCHEMA)))
                .isEqualTo(schema);
    }
}

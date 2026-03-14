package com.arize.instrumentation.annotations;

import static org.assertj.core.api.Assertions.assertThat;

import com.arize.instrumentation.OITracer;
import com.arize.instrumentation.TraceConfig;
import com.arize.semconv.trace.SemanticConventions;
import io.opentelemetry.api.common.AttributeKey;
import io.opentelemetry.api.trace.StatusCode;
import io.opentelemetry.sdk.testing.exporter.InMemorySpanExporter;
import io.opentelemetry.sdk.trace.SdkTracerProvider;
import io.opentelemetry.sdk.trace.data.SpanData;
import io.opentelemetry.sdk.trace.export.SimpleSpanProcessor;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class TracedSpanTest {

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

    @AfterEach
    void tearDown() {
        exporter.reset();
    }

    @Test
    void chainSpanSetsCorrectKind() {
        try (TracedChainSpan span = TracedChainSpan.start(tracer, "test-chain")) {
            span.setInput("hello");
            span.setOutput("world");
        }

        List<SpanData> spans = exporter.getFinishedSpanItems();
        assertThat(spans).hasSize(1);
        SpanData data = spans.get(0);
        assertThat(data.getName()).isEqualTo("test-chain");
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.OPENINFERENCE_SPAN_KIND)))
                .isEqualTo("CHAIN");
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.INPUT_VALUE)))
                .isEqualTo("hello");
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.OUTPUT_VALUE)))
                .isEqualTo("world");
        assertThat(data.getStatus().getStatusCode()).isEqualTo(StatusCode.OK);
    }

    @Test
    void spanRecordsErrorCorrectly() {
        RuntimeException error = new RuntimeException("test error");
        try (TracedChainSpan span = TracedChainSpan.start(tracer, "error-span")) {
            span.setInput("input");
            span.setError(error);
        }

        List<SpanData> spans = exporter.getFinishedSpanItems();
        assertThat(spans).hasSize(1);
        SpanData data = spans.get(0);
        assertThat(data.getStatus().getStatusCode()).isEqualTo(StatusCode.ERROR);
        assertThat(data.getStatus().getDescription()).isEqualTo("test error");
    }

    @Test
    void closeIsIdempotent() {
        TracedChainSpan span = TracedChainSpan.start(tracer, "idempotent");
        span.setInput("input");
        span.close();
        span.close(); // second close should not throw

        List<SpanData> spans = exporter.getFinishedSpanItems();
        assertThat(spans).hasSize(1);
    }

    @Test
    void hideInputsRespectsTraceConfig() {
        TraceConfig config = TraceConfig.builder().hideInputs(true).build();
        OITracer maskedTracer = new OITracer(
                SdkTracerProvider.builder()
                        .addSpanProcessor(SimpleSpanProcessor.create(exporter))
                        .build()
                        .get("test"),
                config);

        try (TracedChainSpan span = TracedChainSpan.start(maskedTracer, "masked")) {
            span.setInput("should-be-hidden");
            span.setOutput("visible");
        }

        List<SpanData> spans = exporter.getFinishedSpanItems();
        SpanData data = spans.get(0);
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.INPUT_VALUE)))
                .isNull();
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.OUTPUT_VALUE)))
                .isEqualTo("visible");
    }

    @Test
    void jsonObjectSerializedWithCorrectMimeType() {
        try (TracedChainSpan span = TracedChainSpan.start(tracer, "json-test")) {
            span.setInput(Map.of("key", "value"));
        }

        List<SpanData> spans = exporter.getFinishedSpanItems();
        SpanData data = spans.get(0);
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.INPUT_MIME_TYPE)))
                .isEqualTo(SemanticConventions.MimeType.JSON.getValue());
    }

    @Test
    void spansNestAutomatically() {
        try (TracedChainSpan parent = TracedChainSpan.start(tracer, "parent")) {
            parent.setInput("parent-input");
            try (TracedChainSpan child = TracedChainSpan.start(tracer, "child")) {
                child.setInput("child-input");
            }
        }

        List<SpanData> spans = exporter.getFinishedSpanItems();
        assertThat(spans).hasSize(2);

        SpanData childData = spans.stream()
                .filter(s -> s.getName().equals("child"))
                .findFirst()
                .orElseThrow();
        SpanData parentData = spans.stream()
                .filter(s -> s.getName().equals("parent"))
                .findFirst()
                .orElseThrow();

        assertThat(childData.getParentSpanId())
                .isEqualTo(parentData.getSpanContext().getSpanId());
    }

    @Test
    void setSessionIdAndUserId() {
        try (TracedChainSpan span = TracedChainSpan.start(tracer, "context-test")) {
            span.setSessionId("session-123");
            span.setUserId("user-456");
        }

        List<SpanData> spans = exporter.getFinishedSpanItems();
        SpanData data = spans.get(0);
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.SESSION_ID)))
                .isEqualTo("session-123");
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.USER_ID)))
                .isEqualTo("user-456");
    }
}

package com.arize.instrumentation.trace;

import static org.assertj.core.api.Assertions.assertThat;

import com.arize.instrumentation.OITracer;
import com.arize.instrumentation.TraceConfig;
import com.arize.semconv.trace.SemanticConventions;
import io.opentelemetry.api.common.AttributeKey;
import io.opentelemetry.sdk.testing.exporter.InMemorySpanExporter;
import io.opentelemetry.sdk.trace.SdkTracerProvider;
import io.opentelemetry.sdk.trace.data.SpanData;
import io.opentelemetry.sdk.trace.export.SimpleSpanProcessor;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class EmbeddingSpanTest {

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
    void setsEmbeddingSpanKind() {
        try (EmbeddingSpan span = EmbeddingSpan.start(tracer, "test-embedding")) {
            span.setInput("embed this text");
        }

        SpanData data = exporter.getFinishedSpanItems().get(0);
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.OPENINFERENCE_SPAN_KIND)))
                .isEqualTo("EMBEDDING");
    }

    @Test
    void setsModelName() {
        try (EmbeddingSpan span = EmbeddingSpan.start(tracer, "test-embedding")) {
            span.setModelName("text-embedding-3-small");
        }

        SpanData data = exporter.getFinishedSpanItems().get(0);
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.EMBEDDING_MODEL_NAME)))
                .isEqualTo("text-embedding-3-small");
    }

    @Test
    void setsEmbeddings() {
        List<Map<String, Object>> embeddings = List.of(Map.of("text", "hello", "vector", List.of(0.1, 0.2, 0.3)));

        try (EmbeddingSpan span = EmbeddingSpan.start(tracer, "test-embedding")) {
            span.setEmbeddings(embeddings);
        }

        SpanData data = exporter.getFinishedSpanItems().get(0);
        String embAttr = data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.EMBEDDING_EMBEDDINGS));
        assertThat(embAttr).isNotNull();
        assertThat(embAttr).contains("hello");
    }

    @Test
    void hideOutputEmbeddingsRespected() {
        TraceConfig config = TraceConfig.builder().hideOutputEmbeddings(true).build();
        OITracer maskedTracer = new OITracer(
                SdkTracerProvider.builder()
                        .addSpanProcessor(SimpleSpanProcessor.create(exporter))
                        .build()
                        .get("test"),
                config);

        try (EmbeddingSpan span = EmbeddingSpan.start(maskedTracer, "masked-embedding")) {
            span.setEmbeddings(List.of(Map.of("text", "secret", "vector", List.of(0.1))));
        }

        SpanData data = exporter.getFinishedSpanItems().get(0);
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.EMBEDDING_EMBEDDINGS)))
                .isNull();
    }
}

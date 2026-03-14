package com.arize.instrumentation.annotations;

import static org.assertj.core.api.Assertions.assertThat;

import com.arize.instrumentation.OITracer;
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

class TracedRetrievalSpanTest {

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
    void setsRetrieverSpanKind() {
        try (TracedRetrievalSpan span = TracedRetrievalSpan.start(tracer, "test-retrieval")) {
            span.setInput("search query");
        }

        SpanData data = exporter.getFinishedSpanItems().get(0);
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.OPENINFERENCE_SPAN_KIND)))
                .isEqualTo("RETRIEVER");
    }

    @Test
    void setsDocuments() {
        List<Map<String, Object>> docs =
                List.of(Map.of("content", "doc1 text", "score", 0.95), Map.of("content", "doc2 text", "score", 0.80));

        try (TracedRetrievalSpan span = TracedRetrievalSpan.start(tracer, "test-retrieval")) {
            span.setInput("search query");
            span.setDocuments(docs);
        }

        SpanData data = exporter.getFinishedSpanItems().get(0);
        String docsAttr = data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.RETRIEVAL_DOCUMENTS));
        assertThat(docsAttr).isNotNull();
        assertThat(docsAttr).contains("doc1 text");
        assertThat(docsAttr).contains("doc2 text");
    }

    @Test
    void setsDocumentsWithEmptyList() {
        try (TracedRetrievalSpan span = TracedRetrievalSpan.start(tracer, "empty-docs")) {
            span.setDocuments(List.of());
        }

        SpanData data = exporter.getFinishedSpanItems().get(0);
        String docsAttr = data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.RETRIEVAL_DOCUMENTS));
        assertThat(docsAttr).isNotNull();
        assertThat(docsAttr).isEqualTo("[]");
    }
}

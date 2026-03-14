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
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class TracedLLMSpanTest {

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
    void setsLLMSpanKindAndModelAttributes() {
        try (TracedLLMSpan span = TracedLLMSpan.start(tracer, "test-llm")) {
            span.setModelName("gpt-4o");
            span.setSystem(SemanticConventions.LLMSystem.OPENAI);
            span.setProvider(SemanticConventions.LLMProvider.OPENAI);
            span.setTokenCountPrompt(100);
            span.setTokenCountCompletion(50);
            span.setTokenCountTotal(150);
        }

        SpanData data = exporter.getFinishedSpanItems().get(0);
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.OPENINFERENCE_SPAN_KIND)))
                .isEqualTo("LLM");
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.LLM_MODEL_NAME)))
                .isEqualTo("gpt-4o");
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.LLM_SYSTEM)))
                .isEqualTo("openai");
        assertThat(data.getAttributes().get(AttributeKey.longKey(SemanticConventions.LLM_TOKEN_COUNT_TOTAL)))
                .isEqualTo(150L);
    }

    @Test
    void hideInputMessagesRespected() {
        TraceConfig config = TraceConfig.builder().hideInputMessages(true).build();
        OITracer maskedTracer = new OITracer(
                SdkTracerProvider.builder()
                        .addSpanProcessor(SimpleSpanProcessor.create(exporter))
                        .build()
                        .get("test"),
                config);

        try (TracedLLMSpan span = TracedLLMSpan.start(maskedTracer, "masked-llm")) {
            span.setInputMessages(List.of(Map.of("role", "user", "content", "secret")));
        }

        SpanData data = exporter.getFinishedSpanItems().get(0);
        // Input messages should not be present
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.LLM_INPUT_MESSAGES)))
                .isNull();
    }
}

package com.arize.instrumentation.springAI;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

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
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.NullAndEmptySource;
import org.junit.jupiter.params.provider.ValueSource;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.metadata.ChatGenerationMetadata;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.Generation;
import org.springframework.ai.chat.observation.ChatModelObservationContext;
import org.springframework.ai.chat.prompt.Prompt;

class SpringAIInstrumentorFinishReasonTest {

    private InMemorySpanExporter spanExporter;
    private SdkTracerProvider tracerProvider;
    private SpringAIInstrumentor instrumentor;

    @BeforeEach
    void setUp() {
        spanExporter = InMemorySpanExporter.create();
        tracerProvider = SdkTracerProvider.builder()
                .addSpanProcessor(SimpleSpanProcessor.create(spanExporter))
                .build();
        instrumentor = new SpringAIInstrumentor(new OITracer(tracerProvider.get("test"), TraceConfig.getDefault()));
    }

    @AfterEach
    void tearDown() {
        tracerProvider.close();
    }

    @ParameterizedTest
    @ValueSource(strings = {"stop", "STOP", "length", "tool_calls", "TOOL_CALLS", "content_filter", "end_turn"})
    void recordsFinishReason(String finishReason) {
        SpanData span = simulateFullCall(new ChatResponse(List.of(generation(finishReason))));

        assertThat(span.getAttributes().get(AttributeKey.stringKey(SemanticConventions.LLM_FINISH_REASON)))
                .isEqualTo(finishReason);
    }

    @ParameterizedTest
    @NullAndEmptySource
    @ValueSource(strings = {" ", "\t"})
    void omitsMissingOrBlankFinishReason(String finishReason) {
        ChatGenerationMetadata metadata = mock(ChatGenerationMetadata.class);
        when(metadata.getFinishReason()).thenReturn(finishReason);
        Generation generation = new Generation(new AssistantMessage("Response"), metadata);

        SpanData span = simulateFullCall(new ChatResponse(List.of(generation)));

        assertThat(span.getAttributes().get(AttributeKey.stringKey(SemanticConventions.LLM_FINISH_REASON)))
                .isNull();
    }

    @Test
    void omitsFinishReasonWithoutResponse() {
        SpanData span = simulateFullCall(null);

        assertThat(span.getAttributes().get(AttributeKey.stringKey(SemanticConventions.LLM_FINISH_REASON)))
                .isNull();
    }

    @Test
    void omitsFinishReasonWithoutGenerations() {
        SpanData span = simulateFullCall(new ChatResponse(List.of()));

        assertThat(span.getAttributes().get(AttributeKey.stringKey(SemanticConventions.LLM_FINISH_REASON)))
                .isNull();
    }

    @Test
    void omitsFinishReasonWithDefaultMetadata() {
        SpanData span = simulateFullCall(new ChatResponse(List.of(new Generation(new AssistantMessage("Response")))));

        assertThat(span.getAttributes().get(AttributeKey.stringKey(SemanticConventions.LLM_FINISH_REASON)))
                .isNull();
    }

    @Test
    void usesFirstGenerationFinishReason() {
        SpanData span = simulateFullCall(new ChatResponse(List.of(generation("stop"), generation("length"))));

        assertThat(span.getAttributes().get(AttributeKey.stringKey(SemanticConventions.LLM_FINISH_REASON)))
                .isEqualTo("stop");
    }

    @Test
    void omitsFinishReasonWhenOnlyLaterGenerationHasOne() {
        Generation firstGeneration = new Generation(new AssistantMessage("Response"));
        SpanData span = simulateFullCall(new ChatResponse(List.of(firstGeneration, generation("length"))));

        assertThat(span.getAttributes().get(AttributeKey.stringKey(SemanticConventions.LLM_FINISH_REASON)))
                .isNull();
    }

    private Generation generation(String finishReason) {
        return new Generation(
                new AssistantMessage("Response"),
                ChatGenerationMetadata.builder().finishReason(finishReason).build());
    }

    private SpanData simulateFullCall(ChatResponse response) {
        ChatModelObservationContext context = mock(ChatModelObservationContext.class);
        when(context.getRequest()).thenReturn(new Prompt("Hello"));
        when(context.getResponse()).thenReturn(response);

        instrumentor.onStart(context);
        instrumentor.onStop(context);

        List<SpanData> spans = spanExporter.getFinishedSpanItems();
        assertThat(spans).hasSize(1);
        SpanData span = spans.get(0);
        assertThat(span.getStatus().getStatusCode()).isEqualTo(StatusCode.OK);
        return span;
    }
}

package com.arize.instrumentation.annotation;

import static org.assertj.core.api.Assertions.assertThat;

import com.arize.instrumentation.OITracer;
import com.arize.instrumentation.OpenInferenceAgent;
import com.arize.instrumentation.trace.*;
import com.arize.semconv.trace.SemanticConventions;
import io.opentelemetry.api.common.AttributeKey;
import io.opentelemetry.sdk.testing.exporter.InMemorySpanExporter;
import io.opentelemetry.sdk.trace.SdkTracerProvider;
import io.opentelemetry.sdk.trace.data.SpanData;
import io.opentelemetry.sdk.trace.export.SimpleSpanProcessor;
import java.lang.reflect.Method;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class TraceAdviceTest {

    private InMemorySpanExporter exporter;
    private OITracer tracer;

    @BeforeEach
    void setUp() {
        exporter = InMemorySpanExporter.create();
        SdkTracerProvider provider = SdkTracerProvider.builder()
                .addSpanProcessor(SimpleSpanProcessor.create(exporter))
                .build();
        tracer = new OITracer(provider.get("test"));
        OpenInferenceAgent.register(tracer);
    }

    @AfterEach
    void tearDown() {
        OpenInferenceAgent.unregister();
        exporter.reset();
    }

    // --- resolveSpanName ---

    @Test
    void resolveSpanNameUsesAnnotationName() throws Exception {
        Method method = AnnotatedTarget.class.getMethod("namedChain");
        assertThat(TraceAdvice.resolveSpanName(method)).isEqualTo("my-chain");
    }

    @Test
    void resolveSpanNameFallsBackToMethodName() throws Exception {
        Method method = AnnotatedTarget.class.getMethod("unnamedChain");
        assertThat(TraceAdvice.resolveSpanName(method)).isEqualTo("unnamedChain");
    }

    @Test
    void resolveSpanNameForLLM() throws Exception {
        Method method = AnnotatedTarget.class.getMethod("namedLLM");
        assertThat(TraceAdvice.resolveSpanName(method)).isEqualTo("my-llm");
    }

    @Test
    void resolveSpanNameForTool() throws Exception {
        Method method = AnnotatedTarget.class.getMethod("namedTool");
        assertThat(TraceAdvice.resolveSpanName(method)).isEqualTo("my-tool");
    }

    @Test
    void resolveSpanNameForAgent() throws Exception {
        Method method = AnnotatedTarget.class.getMethod("namedAgent");
        assertThat(TraceAdvice.resolveSpanName(method)).isEqualTo("my-agent");
    }

    @Test
    void resolveSpanNameForGenericSpan() throws Exception {
        Method method = AnnotatedTarget.class.getMethod("genericSpan");
        assertThat(TraceAdvice.resolveSpanName(method)).isEqualTo("my-retriever");
    }

    // --- resolveSpanKind ---

    @Test
    void resolveSpanKindForChain() throws Exception {
        Method method = AnnotatedTarget.class.getMethod("namedChain");
        assertThat(TraceAdvice.resolveSpanKind(method)).isEqualTo(SemanticConventions.OpenInferenceSpanKind.CHAIN);
    }

    @Test
    void resolveSpanKindForLLM() throws Exception {
        Method method = AnnotatedTarget.class.getMethod("namedLLM");
        assertThat(TraceAdvice.resolveSpanKind(method)).isEqualTo(SemanticConventions.OpenInferenceSpanKind.LLM);
    }

    @Test
    void resolveSpanKindForTool() throws Exception {
        Method method = AnnotatedTarget.class.getMethod("namedTool");
        assertThat(TraceAdvice.resolveSpanKind(method)).isEqualTo(SemanticConventions.OpenInferenceSpanKind.TOOL);
    }

    @Test
    void resolveSpanKindForAgent() throws Exception {
        Method method = AnnotatedTarget.class.getMethod("namedAgent");
        assertThat(TraceAdvice.resolveSpanKind(method)).isEqualTo(SemanticConventions.OpenInferenceSpanKind.AGENT);
    }

    @Test
    void resolveSpanKindForGenericSpan() throws Exception {
        Method method = AnnotatedTarget.class.getMethod("genericSpan");
        assertThat(TraceAdvice.resolveSpanKind(method)).isEqualTo(SemanticConventions.OpenInferenceSpanKind.RETRIEVER);
    }

    // --- createTypedSpan ---

    @Test
    void createTypedSpanReturnsCorrectTypes() {
        try (TracedSpan span =
                TraceAdvice.createTypedSpan(tracer, "test", SemanticConventions.OpenInferenceSpanKind.LLM)) {
            assertThat(span).isInstanceOf(LLMSpan.class);
        }
        try (TracedSpan span =
                TraceAdvice.createTypedSpan(tracer, "test", SemanticConventions.OpenInferenceSpanKind.TOOL)) {
            assertThat(span).isInstanceOf(ToolSpan.class);
        }
        try (TracedSpan span =
                TraceAdvice.createTypedSpan(tracer, "test", SemanticConventions.OpenInferenceSpanKind.AGENT)) {
            assertThat(span).isInstanceOf(AgentSpan.class);
        }
        try (TracedSpan span =
                TraceAdvice.createTypedSpan(tracer, "test", SemanticConventions.OpenInferenceSpanKind.RETRIEVER)) {
            assertThat(span).isInstanceOf(RetrievalSpan.class);
        }
        try (TracedSpan span =
                TraceAdvice.createTypedSpan(tracer, "test", SemanticConventions.OpenInferenceSpanKind.EMBEDDING)) {
            assertThat(span).isInstanceOf(EmbeddingSpan.class);
        }
        try (TracedSpan span =
                TraceAdvice.createTypedSpan(tracer, "test", SemanticConventions.OpenInferenceSpanKind.CHAIN)) {
            assertThat(span).isInstanceOf(ChainSpan.class);
        }
    }

    // --- onEnter / onExit (direct invocation, not ByteBuddy) ---

    @Test
    void onEnterCreatesSpanWithAutoInput() throws Exception {
        Method method = AnnotatedTarget.class.getMethod("unnamedChain");
        Object[] args = {};

        TracedSpan span = TraceAdvice.onEnter(method, args);
        assertThat(span).isNotNull();
        span.close();

        SpanData data = exporter.getFinishedSpanItems().get(0);
        assertThat(data.getName()).isEqualTo("unnamedChain");
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.OPENINFERENCE_SPAN_KIND)))
                .isEqualTo("CHAIN");
    }

    @Test
    void onEnterReturnsNullWhenNoTracerRegistered() throws Exception {
        OpenInferenceAgent.unregister();
        Method method = AnnotatedTarget.class.getMethod("unnamedChain");
        TracedSpan span = TraceAdvice.onEnter(method, new Object[] {});
        assertThat(span).isNull();
    }

    @Test
    void onEnterCapturesMethodParams() throws Exception {
        Method method = AnnotatedTarget.class.getMethod("withParams", String.class, int.class);
        Object[] args = {"hello", 42};

        TracedSpan span = TraceAdvice.onEnter(method, args);
        assertThat(span).isNotNull();
        TraceAdvice.onExit(span, method, null, null);

        SpanData data = exporter.getFinishedSpanItems().get(0);
        String inputValue = data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.INPUT_VALUE));
        assertThat(inputValue).isNotNull();
        assertThat(inputValue).contains("hello");
        assertThat(inputValue).contains("42");
    }

    @Test
    void onEnterFiltersExcludeFromSpanParams() throws Exception {
        Method method = AnnotatedTarget.class.getMethod("withIgnored", String.class, String.class);
        Object[] args = {"visible", "secret"};

        TracedSpan span = TraceAdvice.onEnter(method, args);
        assertThat(span).isNotNull();
        TraceAdvice.onExit(span, method, null, null);

        SpanData data = exporter.getFinishedSpanItems().get(0);
        String inputValue = data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.INPUT_VALUE));
        // Single non-ignored param should be set directly (not as map)
        assertThat(inputValue).isEqualTo("visible");
    }

    @Test
    void onExitRecordsOutput() throws Exception {
        Method method = AnnotatedTarget.class.getMethod("unnamedChain");
        TracedSpan span = TraceAdvice.onEnter(method, new Object[] {});
        TraceAdvice.onExit(span, method, "result-value", null);

        SpanData data = exporter.getFinishedSpanItems().get(0);
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.OUTPUT_VALUE)))
                .isEqualTo("result-value");
    }

    @Test
    void onExitRecordsError() throws Exception {
        Method method = AnnotatedTarget.class.getMethod("unnamedChain");
        TracedSpan span = TraceAdvice.onEnter(method, new Object[] {});
        TraceAdvice.onExit(span, method, null, new RuntimeException("test error"));

        SpanData data = exporter.getFinishedSpanItems().get(0);
        assertThat(data.getStatus().getStatusCode()).isEqualTo(io.opentelemetry.api.trace.StatusCode.ERROR);
    }

    @Test
    void toolDescriptionIsApplied() throws Exception {
        Method method = AnnotatedTarget.class.getMethod("namedTool");
        TracedSpan span = TraceAdvice.onEnter(method, new Object[] {});
        TraceAdvice.onExit(span, method, null, null);

        SpanData data = exporter.getFinishedSpanItems().get(0);
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.TOOL_DESCRIPTION)))
                .isEqualTo("A test tool");
    }

    // --- Test target with annotations ---

    public static class AnnotatedTarget {
        @Chain(name = "my-chain")
        public void namedChain() {}

        @Chain
        public void unnamedChain() {}

        @LLM(name = "my-llm")
        public void namedLLM() {}

        @Tool(name = "my-tool", description = "A test tool")
        public void namedTool() {}

        @Agent(name = "my-agent")
        public void namedAgent() {}

        @Span(name = "my-retriever", kind = SemanticConventions.OpenInferenceSpanKind.RETRIEVER)
        public void genericSpan() {}

        @Chain
        public void withParams(String query, int count) {}

        @Chain
        public void withIgnored(String query, @ExcludeFromSpan String secret) {}
    }
}

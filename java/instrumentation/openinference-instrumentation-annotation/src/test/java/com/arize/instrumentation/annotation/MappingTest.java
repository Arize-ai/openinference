package com.arize.instrumentation.annotation;

import static org.assertj.core.api.Assertions.assertThat;

import com.arize.instrumentation.OITracer;
import com.arize.instrumentation.OpenInferenceAgent;
import com.arize.instrumentation.TraceConfig;
import com.arize.instrumentation.trace.TracedSpan;
import com.arize.semconv.trace.SemanticConventions;
import io.opentelemetry.api.common.AttributeKey;
import io.opentelemetry.sdk.testing.exporter.InMemorySpanExporter;
import io.opentelemetry.sdk.trace.SdkTracerProvider;
import io.opentelemetry.sdk.trace.data.SpanData;
import io.opentelemetry.sdk.trace.export.SimpleSpanProcessor;
import java.lang.reflect.Method;
import java.util.Map;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class MappingTest {

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

    @Test
    void inputMappingAppliesParamToAttribute() throws Exception {
        Method method = MappingTarget.class.getMethod("withInputMapping", String.class, String.class);
        Object[] args = {"the-prompt", "gpt-4o"};

        TracedSpan span = TraceAdvice.onEnter(method, args);
        TraceAdvice.onExit(span, method, null, null);

        SpanData data = exporter.getFinishedSpanItems().get(0);
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.LLM_MODEL_NAME)))
                .isEqualTo("gpt-4o");
    }

    @Test
    void outputMappingExtractsFieldToAttribute() throws Exception {
        Method method = MappingTarget.class.getMethod("withOutputMapping");
        Object[] args = {};

        TracedSpan span = TraceAdvice.onEnter(method, args);
        Map<String, Object> result = Map.of("usage", Map.of("totalTokens", 150));
        TraceAdvice.onExit(span, method, result, null);

        SpanData data = exporter.getFinishedSpanItems().get(0);
        assertThat(data.getAttributes().get(AttributeKey.longKey(SemanticConventions.LLM_TOKEN_COUNT_TOTAL)))
                .isEqualTo(150L);
    }

    @Test
    void getInputMappingsReturnsNullForUnannotated() throws Exception {
        Method method = MappingTarget.class.getMethod("unannotated");
        assertThat(TraceAdvice.getInputMappings(method)).isNull();
    }

    @Test
    void getOutputMappingsReturnsNullForUnannotated() throws Exception {
        Method method = MappingTarget.class.getMethod("unannotated");
        assertThat(TraceAdvice.getOutputMappings(method)).isNull();
    }

    @Test
    void getInputMappingsForAllAnnotationTypes() throws Exception {
        assertThat(TraceAdvice.getInputMappings(MappingTarget.class.getMethod("chainMethod")))
                .isNotNull();
        assertThat(TraceAdvice.getInputMappings(MappingTarget.class.getMethod("llmMethod")))
                .isNotNull();
        assertThat(TraceAdvice.getInputMappings(MappingTarget.class.getMethod("toolMethod")))
                .isNotNull();
        assertThat(TraceAdvice.getInputMappings(MappingTarget.class.getMethod("agentMethod")))
                .isNotNull();
        assertThat(TraceAdvice.getInputMappings(MappingTarget.class.getMethod("spanMethod")))
                .isNotNull();
    }

    // --- hide inputs suppresses input mappings ---

    @Test
    void hideInputsSuppressesInputMappings() throws Exception {
        OpenInferenceAgent.unregister();
        exporter.reset();

        TraceConfig config = TraceConfig.builder().hideInputs(true).build();
        SdkTracerProvider provider = SdkTracerProvider.builder()
                .addSpanProcessor(SimpleSpanProcessor.create(exporter))
                .build();
        OITracer hiddenTracer = new OITracer(provider.get("test-hidden"), config);
        OpenInferenceAgent.register(hiddenTracer);

        Method method = MappingTarget.class.getMethod("withInputMapping", String.class, String.class);
        Object[] args = {"secret-prompt", "gpt-4o"};

        TracedSpan span = TraceAdvice.onEnter(method, args);
        TraceAdvice.onExit(span, method, null, null);

        SpanData data = exporter.getFinishedSpanItems().get(0);
        // Input mapping attribute should be suppressed
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.LLM_MODEL_NAME)))
                .isNull();
        // Input value should also be suppressed
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.INPUT_VALUE)))
                .isNull();
    }

    @Test
    void hideOutputsSuppressesOutputMappings() throws Exception {
        OpenInferenceAgent.unregister();
        exporter.reset();

        TraceConfig config = TraceConfig.builder().hideOutputs(true).build();
        SdkTracerProvider provider = SdkTracerProvider.builder()
                .addSpanProcessor(SimpleSpanProcessor.create(exporter))
                .build();
        OITracer hiddenTracer = new OITracer(provider.get("test-hidden"), config);
        OpenInferenceAgent.register(hiddenTracer);

        Method method = MappingTarget.class.getMethod("withOutputMapping");
        Object[] args = {};

        TracedSpan span = TraceAdvice.onEnter(method, args);
        Map<String, Object> result = Map.of("usage", Map.of("totalTokens", 150));
        TraceAdvice.onExit(span, method, result, null);

        SpanData data = exporter.getFinishedSpanItems().get(0);
        // Output mapping attribute should be suppressed
        assertThat(data.getAttributes().get(AttributeKey.longKey(SemanticConventions.LLM_TOKEN_COUNT_TOTAL)))
                .isNull();
        // Output value should also be suppressed
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.OUTPUT_VALUE)))
                .isNull();
    }

    // --- Test target with mapping annotations ---

    public static class MappingTarget {
        @LLM(
                name = "mapped-llm",
                mapping = @SpanMapping(param = "model", attribute = SemanticConventions.LLM_MODEL_NAME))
        public void withInputMapping(String prompt, String model) {}

        @LLM(
                name = "output-mapped",
                outputMapping =
                        @SpanMapping(
                                field = "usage.totalTokens",
                                attribute = SemanticConventions.LLM_TOKEN_COUNT_TOTAL))
        public Map<String, Object> withOutputMapping() {
            return Map.of();
        }

        public void unannotated() {}

        @Chain
        public void chainMethod() {}

        @LLM
        public void llmMethod() {}

        @Tool
        public void toolMethod() {}

        @Agent
        public void agentMethod() {}

        @Span(kind = SemanticConventions.OpenInferenceSpanKind.EMBEDDING)
        public void spanMethod() {}
    }
}

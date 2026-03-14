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
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class AnnotationIntegrationTest {

    private InMemorySpanExporter exporter;
    private OITracer tracer;

    @BeforeEach
    void setUp() {
        exporter = InMemorySpanExporter.create();
        SdkTracerProvider provider = SdkTracerProvider.builder()
                .addSpanProcessor(SimpleSpanProcessor.create(exporter))
                .build();
        tracer = new OITracer(provider.get("integration-test"));
    }

    @AfterEach
    void tearDown() {
        exporter.reset();
    }

    @Test
    void nestedAgentWithChainAndLLMProducesCorrectTrace() {
        // Simulate: Agent -> Chain + LLM
        try (TracedAgentSpan agent = TracedAgentSpan.start(tracer, "qa-agent")) {
            agent.setInput("What is OpenInference?");
            agent.setAgentName("qa-agent");

            // Child: Chain
            String docs;
            try (TracedChainSpan chain = TracedChainSpan.start(tracer, "retriever")) {
                chain.setInput("What is OpenInference?");
                docs = "OpenInference is a tracing standard";
                chain.setOutput(docs);
            }

            // Child: LLM
            try (TracedLLMSpan llm = TracedLLMSpan.start(tracer, "generator")) {
                llm.setInput(docs);
                llm.setModelName("gpt-4o");
                llm.setTokenCountTotal(150);
                llm.setOutput("OpenInference provides tracing for AI apps");
            }

            agent.setOutput("OpenInference provides tracing for AI apps");
        }

        List<SpanData> spans = exporter.getFinishedSpanItems();
        assertThat(spans).hasSize(3);

        SpanData agentSpan = spans.stream()
                .filter(s -> s.getName().equals("qa-agent"))
                .findFirst()
                .orElseThrow();
        SpanData chainSpan = spans.stream()
                .filter(s -> s.getName().equals("retriever"))
                .findFirst()
                .orElseThrow();
        SpanData llmSpan = spans.stream()
                .filter(s -> s.getName().equals("generator"))
                .findFirst()
                .orElseThrow();

        // Verify nesting
        assertThat(chainSpan.getParentSpanId())
                .isEqualTo(agentSpan.getSpanContext().getSpanId());
        assertThat(llmSpan.getParentSpanId())
                .isEqualTo(agentSpan.getSpanContext().getSpanId());

        // Verify span kinds
        assertThat(agentSpan.getAttributes().get(AttributeKey.stringKey(SemanticConventions.OPENINFERENCE_SPAN_KIND)))
                .isEqualTo("AGENT");
        assertThat(chainSpan.getAttributes().get(AttributeKey.stringKey(SemanticConventions.OPENINFERENCE_SPAN_KIND)))
                .isEqualTo("CHAIN");
        assertThat(llmSpan.getAttributes().get(AttributeKey.stringKey(SemanticConventions.OPENINFERENCE_SPAN_KIND)))
                .isEqualTo("LLM");

        // Verify LLM attributes
        assertThat(llmSpan.getAttributes().get(AttributeKey.stringKey(SemanticConventions.LLM_MODEL_NAME)))
                .isEqualTo("gpt-4o");
        assertThat(llmSpan.getAttributes().get(AttributeKey.longKey(SemanticConventions.LLM_TOKEN_COUNT_TOTAL)))
                .isEqualTo(150L);
    }

    @Test
    void toolSpanWithDescription() {
        try (TracedToolSpan tool = TracedToolSpan.start(tracer, "weather-tool")) {
            tool.setInput(Map.of("location", "San Francisco"));
            tool.setToolName("weather");
            tool.setToolDescription("Gets current weather");
            tool.setOutput(Map.of("temp", 72, "condition", "sunny"));
        }

        SpanData data = exporter.getFinishedSpanItems().get(0);
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.OPENINFERENCE_SPAN_KIND)))
                .isEqualTo("TOOL");
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.TOOL_DESCRIPTION)))
                .isEqualTo("Gets current weather");
        // Input should be JSON since it's a Map
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.INPUT_MIME_TYPE)))
                .isEqualTo("application/json");
    }

    @Test
    void openInferenceAgentRegistrationAndUnregistration() {
        OpenInferenceAgent.register(tracer);
        assertThat(OpenInferenceAgent.getTracer()).isNotNull();

        OpenInferenceAgent.unregister();
        assertThat(OpenInferenceAgent.getTracer()).isNull();

        // Can re-register after unregister
        OpenInferenceAgent.register(tracer);
        assertThat(OpenInferenceAgent.getTracer()).isNotNull();

        // Clean up
        OpenInferenceAgent.unregister();
    }
}

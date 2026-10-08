package com.arize.examples.programmatic;

import static com.arize.semconv.trace.SemanticResourceAttributes.SEMRESATTRS_PROJECT_NAME;

import com.arize.instrumentation.OITracer;
import com.arize.instrumentation.trace.*;
import io.opentelemetry.api.common.AttributeKey;
import io.opentelemetry.api.common.Attributes;
import io.opentelemetry.api.trace.propagation.W3CTraceContextPropagator;
import io.opentelemetry.context.propagation.ContextPropagators;
import io.opentelemetry.exporter.logging.LoggingSpanExporter;
import io.opentelemetry.exporter.otlp.trace.OtlpGrpcSpanExporter;
import io.opentelemetry.sdk.OpenTelemetrySdk;
import io.opentelemetry.sdk.common.CompletableResultCode;
import io.opentelemetry.sdk.resources.Resource;
import io.opentelemetry.sdk.trace.SdkTracerProvider;
import io.opentelemetry.sdk.trace.export.BatchSpanProcessor;
import io.opentelemetry.sdk.trace.export.SimpleSpanProcessor;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;

/**
 * Example demonstrating the OpenInference programmatic trace API.
 *
 * <p>No annotations or ByteBuddy agent required — just typed span classes
 * with try-with-resources.
 *
 * <p>To run:
 * <ol>
 *   <li>Start Phoenix: {@code docker run -p 6006:6006 -p 4317:4317 arizephoenix/phoenix:latest}</li>
 *   <li>Run: {@code ./gradlew :examples:programmatic-example:run}</li>
 *   <li>View traces at http://localhost:6006</li>
 * </ol>
 */
public class ProgrammaticDemo {

    private static final Logger logger = Logger.getLogger(ProgrammaticDemo.class.getName());

    public static void main(String[] args) {
        SdkTracerProvider tracerProvider = initializeOpenTelemetry();
        OITracer tracer = new OITracer(tracerProvider.get("programmatic-example"));

        logger.info("=== Programmatic Tracing Demo ===");

        try (AgentSpan agent = AgentSpan.start(tracer, "qa-agent")) {
            agent.setInput("What is OpenInference?");
            agent.setAgentName("qa-agent");

            // Retrieval step
            String context;
            try (RetrievalSpan retrieval = RetrievalSpan.start(tracer, "search")) {
                retrieval.setInput("What is OpenInference?");
                context = "OpenInference is an open standard for AI tracing.";
                retrieval.setDocuments(
                        List.of(Map.of("document.content", context, "document.metadata", Map.of("source", "docs"))));
                retrieval.setOutput(context);
            }

            // Tool call
            try (ToolSpan tool = ToolSpan.start(tracer, "weather")) {
                tool.setToolName("weather");
                tool.setToolDescription("Gets current weather for a location");
                tool.setInput("San Francisco");
                tool.setOutput(Map.of("temp", 68, "condition", "foggy"));
            }

            // LLM call
            String answer;
            try (LLMSpan llm = LLMSpan.start(tracer, "generate")) {
                llm.setInput("What is OpenInference?");
                llm.setModelName("gpt-4o");
                llm.setSystem(com.arize.semconv.trace.SemanticConventions.LLMSystem.OPENAI);

                answer = "OpenInference provides tracing for AI apps.";

                llm.setOutput(answer);
                llm.setTokenCountPrompt(42);
                llm.setTokenCountCompletion(15);
                llm.setTokenCountTotal(57);
            }

            agent.setOutput(answer);
            logger.info("Agent answer: " + answer);
        }

        logger.info("=== Demo Complete ===");

        CompletableResultCode flushResult = tracerProvider.forceFlush();
        flushResult.join(10, TimeUnit.SECONDS);
        tracerProvider.shutdown().join(10, TimeUnit.SECONDS);

        System.out.println("\nTraces have been exported. Check Phoenix at http://localhost:6006");
    }

    private static SdkTracerProvider initializeOpenTelemetry() {
        String projectName = System.getenv().getOrDefault("PROJECT_NAME", "java-programmatic");

        Resource resource = Resource.getDefault()
                .merge(Resource.create(Attributes.of(
                        AttributeKey.stringKey("service.name"),
                        "programmatic-example",
                        AttributeKey.stringKey(SEMRESATTRS_PROJECT_NAME),
                        projectName)));

        OtlpGrpcSpanExporter otlpExporter = OtlpGrpcSpanExporter.builder()
                .setEndpoint("http://localhost:4317")
                .setTimeout(Duration.ofSeconds(2))
                .build();

        SdkTracerProvider tracerProvider = SdkTracerProvider.builder()
                .addSpanProcessor(BatchSpanProcessor.builder(otlpExporter)
                        .setScheduleDelay(Duration.ofSeconds(1))
                        .build())
                .addSpanProcessor(SimpleSpanProcessor.create(LoggingSpanExporter.create()))
                .setResource(resource)
                .build();

        OpenTelemetrySdk.builder()
                .setTracerProvider(tracerProvider)
                .setPropagators(ContextPropagators.create(W3CTraceContextPropagator.getInstance()))
                .buildAndRegisterGlobal();

        System.out.println("OpenTelemetry initialized. Traces will be sent to Phoenix at http://localhost:4317");
        return tracerProvider;
    }
}

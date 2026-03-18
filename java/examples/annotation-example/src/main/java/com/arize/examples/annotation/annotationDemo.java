package com.arize.examples.annotation;

import static com.arize.semconv.trace.SemanticResourceAttributes.SEMRESATTRS_PROJECT_NAME;

import com.arize.instrumentation.OITracer;
import com.arize.instrumentation.annotation.OpenInferenceAgent;
import com.arize.instrumentation.annotation.OpenInferenceAgentInstaller;
import io.opentelemetry.api.common.AttributeKey;
import io.opentelemetry.api.common.Attributes;
import io.opentelemetry.api.trace.propagation.W3CTraceContextPropagator;
import io.opentelemetry.context.propagation.ContextPropagators;
import io.opentelemetry.exporter.logging.LoggingSpanExporter;
import io.opentelemetry.exporter.otlp.trace.OtlpGrpcSpanExporter;
import io.opentelemetry.exporter.otlp.trace.OtlpGrpcSpanExporterBuilder;
import io.opentelemetry.sdk.OpenTelemetrySdk;
import io.opentelemetry.sdk.common.CompletableResultCode;
import io.opentelemetry.sdk.resources.Resource;
import io.opentelemetry.sdk.trace.SdkTracerProvider;
import io.opentelemetry.sdk.trace.export.BatchSpanProcessor;
import io.opentelemetry.sdk.trace.export.SimpleSpanProcessor;
import java.time.Duration;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Example demonstrating OpenInference annotation-based tracing.
 *
 * To run this example with Phoenix:
 * 1. Start Phoenix:
 * {@code docker run -p 6006:6006 -p 4317:4317 arizephoenix/phoenix:latest}
 * 2. (Optional) if you have auth enabled on Phoenix,
 * {@code export PHOENIX_API_KEY=your-key-here}
 * 3. Run the example: {@code ./gradlew :examples:annotation-example:run}
 * 4. View traces in Phoenix: http://localhost:6006
 */
public class annotationDemo {

    private static SdkTracerProvider tracerProvider;
    private static final Logger logger = Logger.getLogger(annotationDemo.class.getName());

    public static void main(String[] args) {
        // Install the ByteBuddy agent BEFORE loading any annotated classes
        OpenInferenceAgentInstaller.install();

        initializeOpenTelemetry();

        // Create OITracer and register it with the annotation agent
        OITracer tracer = new OITracer(tracerProvider.get("annotation-example"));
        OpenInferenceAgent.register(tracer);

        logger.info("=== Annotation-Based Tracing Demo ===");

        // Use the annotated service — spans are created automatically by @Agent,
        // @Chain, etc.
        QAService service = new QAService();
        String answer = service.answer("What is OpenInference and what's the weather?");
        logger.info("Agent answer: " + answer);

        logger.info("=== Demo Complete ===");

        // Flush and shutdown
        OpenInferenceAgent.unregister();

        if (tracerProvider != null) {
            logger.info("Flushing and shutting down trace provider...");

            CompletableResultCode flushResult = tracerProvider.forceFlush();
            flushResult.join(10, java.util.concurrent.TimeUnit.SECONDS);

            if (flushResult.isSuccess()) {
                logger.info("Successfully flushed all traces");
            } else {
                logger.warning("Failed to flush all traces");
            }

            CompletableResultCode shutdownResult = tracerProvider.shutdown();
            shutdownResult.join(10, java.util.concurrent.TimeUnit.SECONDS);

            if (!shutdownResult.isSuccess()) {
                logger.warning("Failed to shutdown trace provider cleanly");
            }
        }

        System.out.println("\nTraces have been exported. Check your Arize or Phoenix dashboard.");
    }

    private static void initializeOpenTelemetry() {
        String projectName = System.getenv().getOrDefault("PROJECT_NAME", "java-annotation");

        Resource resource = Resource.getDefault()
                .merge(Resource.create(Attributes.of(
                        AttributeKey.stringKey("service.name"), "annotation-example",
                        AttributeKey.stringKey(SEMRESATTRS_PROJECT_NAME), projectName,
                        AttributeKey.stringKey("service.version"), "0.1.0")));

        String arizeApiKey = System.getenv("ARIZE_API_KEY");
        String arizeSpaceId = System.getenv("ARIZE_SPACE_ID");
        String phoenixApiKey = System.getenv("PHOENIX_API_KEY");

        OtlpGrpcSpanExporterBuilder otlpExporterBuilder;
        String destination;

        if (arizeApiKey != null && !arizeApiKey.isEmpty()) {
            // Arize cloud
            otlpExporterBuilder = OtlpGrpcSpanExporter.builder()
                    .setEndpoint("https://otlp.arize.com")
                    .setTimeout(Duration.ofSeconds(5))
                    .setHeaders(
                            () -> Map.of("api_key", arizeApiKey, "space_id", arizeSpaceId != null ? arizeSpaceId : ""));
            destination = "Arize (project: " + projectName + ")";
        } else {
            // Phoenix local
            otlpExporterBuilder = OtlpGrpcSpanExporter.builder()
                    .setEndpoint("http://localhost:4317")
                    .setTimeout(Duration.ofSeconds(2));
            if (phoenixApiKey != null && !phoenixApiKey.isEmpty()) {
                otlpExporterBuilder.setHeaders(
                        () -> Map.of("Authorization", String.format("Bearer %s", phoenixApiKey)));
            } else {
                logger.log(Level.WARNING, "No ARIZE_API_KEY or PHOENIX_API_KEY set.");
            }
            destination = "Phoenix at http://localhost:6006";
        }

        OtlpGrpcSpanExporter otlpExporter = otlpExporterBuilder.build();

        tracerProvider = SdkTracerProvider.builder()
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

        System.out.println("OpenTelemetry initialized. Traces will be sent to " + destination);
    }
}

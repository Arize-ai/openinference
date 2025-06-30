package io.openinference.examples;

import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import io.opentelemetry.api.common.AttributeKey;
import io.opentelemetry.api.common.Attributes;
import io.opentelemetry.api.trace.propagation.W3CTraceContextPropagator;
import io.opentelemetry.context.propagation.ContextPropagators;
import io.opentelemetry.exporter.logging.LoggingSpanExporter;
import io.opentelemetry.exporter.otlp.trace.OtlpGrpcSpanExporter;
import io.opentelemetry.exporter.otlp.trace.OtlpGrpcSpanExporterBuilder;
import io.opentelemetry.sdk.OpenTelemetrySdk;
import io.opentelemetry.sdk.resources.Resource;
import io.opentelemetry.sdk.trace.SdkTracerProvider;
import io.opentelemetry.sdk.trace.export.BatchSpanProcessor;
import io.opentelemetry.sdk.trace.export.SimpleSpanProcessor;
import io.openinference.instrumentation.langchain4j.LangChain4jInstrumentor;
import io.openinference.instrumentation.langchain4j.LangChain4jModelListener;

import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Example demonstrating OpenInference instrumentation with LangChain4j.
 * 
 * To run this example with Phoenix:
 * 1. Start Phoenix: `docker run -p 6006:6006 -p 4317:4317 arizephoenix/phoenix:latest`
 * 2. Set your OpenAI API key: `export OPENAI_API_KEY=your-key-here`
 * 3. (Optional) if you have auth enabled on Phoenix, `export PHOENIX_API_KEY=your-key-here`
 * 4. Run the example: `./gradlew :examples:langchain4j-example:run`
 * 5. View traces in Phoenix: http://localhost:6006
 */
public class LangChain4jExample {
    
    private static SdkTracerProvider tracerProvider;
    private static final Logger logger = Logger.getLogger(LangChain4jExample.class.getName());
    
    public static void main(String[] args) {
        // 1. Initialize OpenTelemetry FIRST
        initializeOpenTelemetry();
        
        // 2. Instrument LangChain4j BEFORE creating models
        LangChain4jInstrumentor instrumentor = LangChain4jInstrumentor.instrument();
        
        // 3. Create the model listener
        LangChain4jModelListener listener = instrumentor.createModelListener();
        
        // 4. NOW create the model with the listener
        String apiKey = System.getenv("OPENAI_API_KEY");
        if (apiKey == null) {
            logger.log(Level.WARNING,"Please set OPENAI_API_KEY environment variable");
            System.exit(1);
        }
        
        ChatLanguageModel model = OpenAiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gpt-3.5-turbo")
                .temperature(0.7)
                .maxTokens(100)
                .listeners(List.of(listener))
                .timeout(Duration.ofSeconds(30))
                .build();
        
        // Use the model - traces will be automatically created
        logger.info("Sending request to OpenAI...");
        String response = model.generate("What is the capital of France? Answer in one sentence.");
        logger.info("Response: " + response);
        
        // Example with multiple messages to show conversation tracing
        logger.info("\nSending another request...");
        String response2 = model.generate("What about Germany?");
        logger.info("Response: " + response2);
        
        // Give time for spans to be exported
        try {
            logger.info("\nWaiting for traces to be exported to Phoenix...");
            Thread.sleep(5000);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        
        // Shutdown
        if (tracerProvider != null) {
            tracerProvider.shutdown();
        }
        
        System.out.println("\nTraces have been sent to Phoenix at http://localhost:6006");
    }
    
    private static void initializeOpenTelemetry() {
        // Create resource with service name
        Resource resource = Resource.getDefault()
                .merge(Resource.create(Attributes.of(
                        AttributeKey.stringKey("service.name"), "langchain4j-example",
                        AttributeKey.stringKey("service.version"), "0.1.0"
                )));

        String apiKey = System.getenv("PHOENIX_API_KEY");
        OtlpGrpcSpanExporterBuilder otlpExporterBuilder = OtlpGrpcSpanExporter.builder()
                .setEndpoint("http://localhost:4317")
                .setTimeout(Duration.ofSeconds(2));
        OtlpGrpcSpanExporter otlpExporter = null;
        if (apiKey != null && !apiKey.isEmpty()) {
            otlpExporter = otlpExporterBuilder
                    .setHeaders(() -> Map.of("Authorization", String.format("Bearer %s", apiKey)))
                    .build();
        } else {
            logger.log(Level.WARNING,"Please set PHOENIX_API_KEY environment variable if auth is enabled.");
            otlpExporter = otlpExporterBuilder.build();
        }



        // Create tracer provider with both OTLP (for Phoenix) and console exporters
        tracerProvider = SdkTracerProvider.builder()
                .addSpanProcessor(BatchSpanProcessor.builder(otlpExporter)
                        .setScheduleDelay(Duration.ofSeconds(1))
                        .build())
                .addSpanProcessor(SimpleSpanProcessor.create(LoggingSpanExporter.create()))
                .setResource(resource)
                .build();
        
        // Build OpenTelemetry SDK
        OpenTelemetrySdk.builder()
                .setTracerProvider(tracerProvider)
                .setPropagators(ContextPropagators.create(W3CTraceContextPropagator.getInstance()))
                .buildAndRegisterGlobal();
        
        // Add shutdown hook
        Runtime.getRuntime().addShutdownHook(new Thread(tracerProvider::shutdown));
        
        System.out.println("OpenTelemetry initialized. Traces will be sent to Phoenix at http://localhost:6006");
    }
} 
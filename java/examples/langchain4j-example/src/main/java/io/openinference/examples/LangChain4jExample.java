package io.openinference.examples;

import static com.arize.semconv.trace.SemanticResourceAttributes.SEMRESATTRS_PROJECT_NAME;

import com.arize.instrumentation.langchain4j.LangChain4jInstrumentor;
import com.arize.instrumentation.langchain4j.LangChain4jModelListener;
import dev.langchain4j.agent.tool.P;
import dev.langchain4j.agent.tool.Tool;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.agent.tool.ToolSpecifications;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.model.openai.OpenAiChatRequestParameters;
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
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;

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

    static class WeatherTools {

        @Tool("Returns the weather forecast for a given city")
        String getWeather(@P("The city for which the weather forecast should be returned") String city) {
            return "85 degrees";
        }
    }

    public static void main(String[] args) {
        initializeOpenTelemetry();

        LangChain4jInstrumentor instrumentor = LangChain4jInstrumentor.instrument();
        LangChain4jModelListener listener = instrumentor.createModelListener();

        String apiKey = System.getenv("OPENAI_API_KEY");
        if (apiKey == null) {
            logger.log(Level.SEVERE, "Please set OPENAI_API_KEY environment variable");
            System.exit(1);
        }

        List<ToolSpecification> toolSpecifications = ToolSpecifications.toolSpecificationsFrom(WeatherTools.class);

        OpenAiChatRequestParameters openAiChatRequestParameters = OpenAiChatRequestParameters.builder()
                .modelName("gpt-4.1")
                .temperature(0.7)
                .maxOutputTokens(100)
                .toolSpecifications(toolSpecifications)
                .build();

        ChatModel model = OpenAiChatModel.builder()
                .apiKey(apiKey)
                .defaultRequestParameters(openAiChatRequestParameters)
                .modelName("gpt-4.1")
                .temperature(0.7)
                .maxTokens(100)
                .listeners(List.of(listener))
                .timeout(Duration.ofSeconds(30))
                .build();
        // Use the model - traces will be automatically created
        logger.info("Sending request to OpenAI...");
        ChatResponse response = model.chat(UserMessage.from("What is the capital of France? Answer in one sentence."));
        logger.info("Response: " + response.aiMessage().text());

        // Example with multiple messages to show conversation tracing
        logger.info("\nSending another request...");
        ChatResponse response2 = model.chat(
                UserMessage.from("What is the capital of France? Answer in one sentence."),
                response.aiMessage(),
                UserMessage.from("What about Germany? also whats the weather like in germany"));
        logger.info("Response: " + response2);

        List<ToolExecutionResultMessage> toolExecutionResultMessages =
                response2.aiMessage().toolExecutionRequests().stream()
                        .map(t -> ToolExecutionResultMessage.from(t, "The weather will be 80 degrees."))
                        .collect(Collectors.toList());
        ArrayList<ChatMessage> messages = new ArrayList<>(List.of(
                UserMessage.from("What is the capital of France? Answer in one sentence."),
                response.aiMessage(),
                UserMessage.from("What about Germany? also whats the weather like in germany"),
                response2.aiMessage()));
        messages.addAll(toolExecutionResultMessages);

        model.chat(messages);

        if (tracerProvider != null) {
            logger.info("Flushing and shutting down trace provider...");

            // Force flush all pending spans
            CompletableResultCode flushResult = tracerProvider.forceFlush();
            flushResult.join(10, java.util.concurrent.TimeUnit.SECONDS);

            if (flushResult.isSuccess()) {
                logger.info("Successfully flushed all traces");
            } else {
                logger.warning("Failed to flush all traces");
            }

            // Shutdown the trace provider
            CompletableResultCode shutdownResult = tracerProvider.shutdown();
            shutdownResult.join(10, java.util.concurrent.TimeUnit.SECONDS);

            if (!shutdownResult.isSuccess()) {
                logger.warning("Failed to shutdown trace provider cleanly");
            }
        }

        System.out.println("\nTraces have been sent to Phoenix at http://localhost:6006");
    }

    private static void initializeOpenTelemetry() {
        // Create resource with service name
        Resource resource = Resource.getDefault()
                .merge(Resource.create(Attributes.of(
                        AttributeKey.stringKey("service.name"), "langchain4j",
                        AttributeKey.stringKey(SEMRESATTRS_PROJECT_NAME), "langchain4j-project",
                        AttributeKey.stringKey("service.version"), "0.1.0")));

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
            logger.log(Level.WARNING, "Please set PHOENIX_API_KEY environment variable if auth is enabled.");
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

        System.out.println("OpenTelemetry initialized. Traces will be sent to Phoenix at http://localhost:6006");
    }
}

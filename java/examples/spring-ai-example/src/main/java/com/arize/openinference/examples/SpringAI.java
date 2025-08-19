package com.arize.openinference.examples;

import static com.arize.semconv.trace.SemanticResourceAttributes.SEMRESATTRS_PROJECT_NAME;

import com.arize.instrumentation.OITracer;
import com.arize.instrumentation.TraceConfig;
import com.arize.instrumentation.springAI.SpringAIInstrumentor;
import io.micrometer.observation.ObservationRegistry;
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
import java.util.function.Function;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.openai.OpenAiChatModel;
import org.springframework.ai.openai.OpenAiChatOptions;
import org.springframework.ai.openai.api.OpenAiApi;
import org.springframework.ai.tool.ToolCallback;
import org.springframework.ai.tool.function.FunctionToolCallback;

/**
 * Example demonstrating OpenInference instrumentation with Spring AI.
 *
 * To run this example with Phoenix:
 * 1. Start Phoenix: `docker run -p 6006:6006 -p 4317:4317 arizephoenix/phoenix:latest`
 * 2. Set your OpenAI API key: `export OPENAI_API_KEY=your-key-here`
 * 3. (Optional) if you have auth enabled on Phoenix, `export PHOENIX_API_KEY=your-key-here`
 * 4. Run the example: `./gradlew :examples:spring-ai-example:run`
 * 5. View traces in Phoenix: http://localhost:6006
 */
public class SpringAI {
    public enum Unit {
        C,
        F
    }

    public record WeatherRequest(String location, Unit unit) {}

    public record WeatherResponse(double temp, Unit unit) {}

    public record MusicRequest(String location) {}

    public record MusicResponse(String song, String description) {}

    static class WeatherService implements Function<WeatherRequest, WeatherResponse> {
        public WeatherResponse apply(WeatherRequest request) {
            return new WeatherResponse(30.0, Unit.C);
        }
    }

    static class MusicService implements Function<MusicRequest, MusicResponse> {
        public MusicResponse apply(MusicRequest request) {
            return new MusicResponse("hips dont lie.", "I dont deny.");
        }
    }

    private static SdkTracerProvider tracerProvider;
    private static final Logger logger = Logger.getLogger(SpringAI.class.getName());

    public static void main(String[] args) {
        initializeOpenTelemetry();

        String apiKey = System.getenv("OPENAI_API_KEY");
        if (apiKey == null) {
            logger.log(Level.SEVERE, "Please set OPENAI_API_KEY environment variable");
            System.exit(1);
        }

        ToolCallback weatherToolCallBack = FunctionToolCallback.builder("currentWeather", new WeatherService())
                .description("Get the weather in location")
                .inputType(WeatherRequest.class)
                .build();

        ToolCallback musicToolCallBack = FunctionToolCallback.builder("topSong", new MusicService())
                .description("Gets the stop song in a location")
                .inputType(MusicRequest.class)
                .build();

        OpenAiApi openAiApi = OpenAiApi.builder().apiKey(apiKey).build();
        OpenAiChatOptions openAiChatOptions = OpenAiChatOptions.builder()
                .model("gpt-4")
                .temperature(0.4)
                .maxTokens(200)
                .toolCallbacks(weatherToolCallBack, musicToolCallBack)
                .parallelToolCalls(true)
//                .toolChoice("required")
                .build();

        // Create OITracer using the initialized tracer provider
        OITracer tracer = new OITracer(tracerProvider.get("com.arize.spring-ai"), TraceConfig.getDefault());

        ObservationRegistry registry = ObservationRegistry.create();
        registry.observationConfig().observationHandler(new SpringAIInstrumentor(tracer));
        //        registry.observationConfig().observationHandler()

        OpenAiChatModel chatModel = OpenAiChatModel.builder()
                .openAiApi(openAiApi)
                .defaultOptions(openAiChatOptions)
                .observationRegistry(registry)
                .build();

        // Use the model - traces will be automatically created
        logger.info("Sending request to OpenAI...");
        ChatResponse response = chatModel.call(new Prompt("Generate the names of 5 famous pirates."));
        logger.info("Response: " + response.getResult().getOutput().toString());
        // Send another request to show multiple spans
        logger.info("\\nSending another request...");

        ChatResponse response2 = chatModel.call(Prompt.builder()
                .messages(
                        new UserMessage("Generate the names of 5 famous pirates."),
                        response.getResult().getOutput(),
                        new UserMessage(
                                "What is the current weather in miami in Fahrenheit? Whats the current trending song there"))
                .build());
        logger.info("Response: " + response2.getResult().getOutput().toString());

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

        System.out.println("\\nTraces have been sent to Phoenix at http://localhost:6006");
    }

    private static void initializeOpenTelemetry() {
        // Create resource with service name
        Resource resource = Resource.getDefault()
                .merge(Resource.create(Attributes.of(
                        AttributeKey.stringKey("service.name"), "spring-ai",
                        AttributeKey.stringKey(SEMRESATTRS_PROJECT_NAME), "spring-ai-project",
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

package io.openinference.examples;

import com.arize.instrumentation.langchain4j.*;
import dev.langchain4j.agent.tool.Tool;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.service.AiServices;
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
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

public class ArizeAXExample {

    private static SdkTracerProvider tracerProvider;
    private static final Logger logger = Logger.getLogger(ArizeAXExample.class.getName());

    /**
     * AI Service interface - LangChain4j will implement this and handle tool calling automatically
     */
    interface MathAssistant {
        String chat(String userMessage);
    }

    /**
     * Math tools that the LLM can call to perform calculations.
     * These tools will be automatically discovered and made available to the LLM.
     */
    static class MathTools {

        @Tool("Adds two numbers together and returns the sum")
        public double add(double a, double b) {
            double result = a + b;
            logger.info(String.format("Tool executed: add(%.2f, %.2f) = %.2f", a, b, result));
            return result;
        }

        @Tool("Subtracts the second number from the first number")
        public double subtract(double a, double b) {
            double result = a - b;
            logger.info(String.format("Tool executed: subtract(%.2f, %.2f) = %.2f", a, b, result));
            return result;
        }

        @Tool("Multiplies two numbers together and returns the product")
        public double multiply(double a, double b) {
            double result = a * b;
            logger.info(String.format("Tool executed: multiply(%.2f, %.2f) = %.2f", a, b, result));
            return result;
        }

        @Tool("Divides the first number by the second number")
        public double divide(double a, double b) {
            if (b == 0) {
                logger.warning("Tool executed: divide - attempted division by zero");
                throw new IllegalArgumentException("Cannot divide by zero");
            }
            double result = a / b;
            logger.info(String.format("Tool executed: divide(%.2f, %.2f) = %.2f", a, b, result));
            return result;
        }
    }

    public static void main(String[] args) {
        // Initialize OpenTelemetry with Arize AX endpoint
        initializeOpenTelemetry();

        // Set up LangChain4j instrumentation
        LangChain4jInstrumentor instrumentor = LangChain4jInstrumentor.instrument();
        LangChain4jModelListener listener = instrumentor.createModelListener();
        LangChain4jAiServiceStartedListener aiServiceListener = instrumentor.createAiServiceStartedListener();
        LangChain4jServiceCompletedListener aiServiceCompletedListener =
                instrumentor.createAiServiceCompletedListener();
        LangChain4jToolExecutedEventListener toolExecutedListener = instrumentor.createToolExecutedListener();
        LangChain4jServiceResponseReceivedListener chatListener = instrumentor.createServiceResponseReceivedListener();
        LangChain4jServiceRequestIssuedListener requestIssuedListener = instrumentor.createServiceRequestIssuedListener();
        LangChain4jAiServiceErrorListener errorListener = instrumentor.createServiceErrorListener();

        // Get OpenAI API key
        String apiKey = System.getenv("OPENAI_API_KEY");
        if (apiKey == null) {
            logger.log(Level.SEVERE, "Please set OPENAI_API_KEY environment variable");
            System.exit(1);
        }

        // Create OpenAI chat model with instrumentation
        OpenAiChatModel model = OpenAiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gpt-4.1-nano") // Use a smaller model for faster responses in this example
                .temperature(1.0) // Lower temperature for more consistent tool usage
                .maxCompletionTokens(300)
                .listeners(List.of(listener))
                .timeout(Duration.ofSeconds(30))
                .build();

        OpenAiChatModel newModel = OpenAiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gpt-4.1-nano") // Use a smaller model for faster responses in this example
                .temperature(1.0) // Lower temperature for more consistent tool usage
                .maxCompletionTokens(300)
                .timeout(Duration.ofSeconds(30))
                .build();

        // Create math tools instance
        MathTools mathTools = new MathTools();

        // Build AI Service with tools - this automatically handles tool calling
        MathAssistant assistant = AiServices.builder(MathAssistant.class)
                .chatModel(newModel)
                .systemMessage("You are a helpful assistant that can perform basic math calculations. Use the provided tools to answer user questions accurately.")
                .tools(mathTools)
                .registerListeners(
                        aiServiceListener, aiServiceCompletedListener, toolExecutedListener,
                        chatListener, requestIssuedListener, errorListener)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .build();

        logger.info("Created Math Assistant with 4 tools: add, subtract, multiply, divide");
        logger.info("The LLM will automatically select and execute the appropriate tool\n");

        model.chat("What is the Sum of 45 and 78? ");

        // Example 1: Simple Addition
        logger.info("=== Example 1: Simple Addition ===");
        String question1 = "What is 45 plus 78?";
        logger.info("Question: " + question1);
        String answer1 = assistant.chat(question1);
        logger.info("Answer: " + answer1 + "\n");

        // Example 2: Multiplication
        logger.info("=== Example 2: Multiplication ===");
        String question2 = "Calculate 23 multiplied by 17";
        logger.info("Question: " + question2);
        String answer2 = assistant.chat(question2);
        logger.info("Answer: " + answer2 + "\n");

        // Example 3: Division
        logger.info("=== Example 3: Division ===");
        String question3 = "What is 144 divided by 12?";
        logger.info("Question: " + question3);
        String answer3 = assistant.chat(question3);
        logger.info("Answer: " + answer3 + "\n");

        shutdownTracing();
    }

    private static void initializeOpenTelemetry() {
        // Create resource with service name
        Resource resource = Resource.getDefault();

        OtlpGrpcSpanExporterBuilder otlpExporterBuilder = OtlpGrpcSpanExporter.builder()
                .setEndpoint("http://localhost:4317")
                .setTimeout(Duration.ofSeconds(2));
        OtlpGrpcSpanExporter otlpExporter = null;
        otlpExporter = otlpExporterBuilder.build();

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

    private static void shutdownTracing() {
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
    }
}

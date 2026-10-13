package com.arize.examples.adk;

import static com.arize.semconv.trace.SemanticResourceAttributes.SEMRESATTRS_PROJECT_NAME;

import com.google.adk.agents.LlmAgent;
import com.google.adk.runner.InMemoryRunner;
import com.google.adk.sessions.Session;
import com.google.adk.tools.Annotations.Schema;
import com.google.adk.tools.FunctionTool;
import com.google.genai.types.Content;
import com.google.genai.types.Part;
import io.opentelemetry.api.common.AttributeKey;
import io.opentelemetry.api.common.Attributes;
import io.opentelemetry.api.trace.propagation.W3CTraceContextPropagator;
import io.opentelemetry.context.propagation.ContextPropagators;
import io.opentelemetry.exporter.otlp.trace.OtlpGrpcSpanExporter;
import io.opentelemetry.sdk.OpenTelemetrySdk;
import io.opentelemetry.sdk.resources.Resource;
import io.opentelemetry.sdk.trace.SdkTracerProvider;
import io.opentelemetry.sdk.trace.export.BatchSpanProcessor;
import java.time.Duration;
import java.util.Map;
import java.util.concurrent.TimeUnit;

/**
 * Google ADK for Java agent with one function tool, traced by the OpenInference ADK Java agent.
 *
 * <p>ADK creates its own OpenTelemetry spans ({@code invocation}, {@code agent_run [...]},
 * {@code call_llm}, {@code tool_call [...]}) through {@code GlobalOpenTelemetry}. The
 * {@code -javaagent} adds the OpenInference span kinds, messages, tool arguments, token counts,
 * session and user ids to those spans, so the only application-side setup is registering a global
 * OpenTelemetry SDK before the first ADK call.
 *
 * <p>To run:
 * <ol>
 *   <li>Start Phoenix: {@code docker run -p 6006:6006 -p 4317:4317 arizephoenix/phoenix:latest}</li>
 *   <li>Export {@code GOOGLE_API_KEY} (or {@code GEMINI_API_KEY})</li>
 *   <li>Run: {@code ./gradlew :examples:adk-java-example:run}</li>
 *   <li>View traces at http://localhost:6006 in the project named by {@code PROJECT_NAME}
 *       (default {@code adk-java-example})</li>
 * </ol>
 */
public class WeatherToolExample {

    /** Canned weather lookup exposed to the model as a function tool. */
    public static Map<String, Object> getWeather(
            @Schema(name = "city", description = "The city to look up the weather for") String city) {
        return Map.of("city", city, "forecast", "sunny", "temperature_celsius", 21);
    }

    public static void main(String[] args) {
        SdkTracerProvider tracerProvider = initializeOpenTelemetry();

        LlmAgent agent = LlmAgent.builder()
                .name("weather_agent")
                .model(System.getenv().getOrDefault("GEMINI_MODEL", "gemini-2.5-flash"))
                .description("Answers weather questions with the getWeather tool.")
                .instruction("You are a weather assistant. Always call getWeather before answering "
                        + "and reply in one short sentence.")
                .tools(FunctionTool.create(WeatherToolExample.class, "getWeather"))
                .build();

        InMemoryRunner runner = new InMemoryRunner(agent);
        String userId = "user-123";
        Session session =
                runner.sessionService().createSession(runner.appName(), userId).blockingGet();

        Content message = Content.fromParts(Part.fromText("What is the weather in Paris right now?"));
        runner.runAsync(userId, session.id(), message).blockingForEach(event -> {
            if (event.finalResponse()) {
                System.out.println("Agent: " + event.stringifyContent());
            }
        });

        tracerProvider.forceFlush().join(10, TimeUnit.SECONDS);
        tracerProvider.shutdown().join(10, TimeUnit.SECONDS);
        System.out.println("Traces exported. Check Phoenix at http://localhost:6006");
    }

    private static SdkTracerProvider initializeOpenTelemetry() {
        String projectName = System.getenv().getOrDefault("PROJECT_NAME", "adk-java-example");
        String endpoint = System.getenv().getOrDefault("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317");

        Resource resource = Resource.getDefault()
                .merge(Resource.create(Attributes.of(
                        AttributeKey.stringKey("service.name"),
                        "adk-java-example",
                        AttributeKey.stringKey(SEMRESATTRS_PROJECT_NAME),
                        projectName)));

        OtlpGrpcSpanExporter exporter = OtlpGrpcSpanExporter.builder()
                .setEndpoint(endpoint)
                .setTimeout(Duration.ofSeconds(5))
                .build();

        SdkTracerProvider tracerProvider = SdkTracerProvider.builder()
                .addSpanProcessor(BatchSpanProcessor.builder(exporter)
                        .setScheduleDelay(Duration.ofSeconds(1))
                        .build())
                .setResource(resource)
                .build();

        // ADK's Telemetry class captures GlobalOpenTelemetry when it is first loaded, so the SDK
        // must be registered before any ADK class is touched.
        OpenTelemetrySdk.builder()
                .setTracerProvider(tracerProvider)
                .setPropagators(ContextPropagators.create(W3CTraceContextPropagator.getInstance()))
                .buildAndRegisterGlobal();
        return tracerProvider;
    }
}

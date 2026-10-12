package com.arize.instrumentation.adk;

import static com.arize.instrumentation.adk.SpanAssertions.allNamed;
import static com.arize.instrumentation.adk.SpanAssertions.awaitSpans;
import static com.arize.instrumentation.adk.SpanAssertions.keysStartingWith;
import static com.arize.instrumentation.adk.SpanAssertions.lng;
import static com.arize.instrumentation.adk.SpanAssertions.named;
import static com.arize.instrumentation.adk.SpanAssertions.names;
import static com.arize.instrumentation.adk.SpanAssertions.str;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.arize.instrumentation.OITracer;
import com.arize.instrumentation.OpenInferenceAgent;
import com.arize.instrumentation.SuppressTracing;
import com.arize.instrumentation.TraceConfig;
import com.google.adk.Telemetry;
import com.google.adk.agents.LlmAgent;
import com.google.adk.agents.RunConfig;
import com.google.adk.events.Event;
import com.google.adk.models.BaseLlm;
import com.google.adk.models.BaseLlmConnection;
import com.google.adk.models.LlmRequest;
import com.google.adk.models.LlmResponse;
import com.google.adk.runner.InMemoryRunner;
import com.google.adk.sessions.Session;
import com.google.adk.tools.Annotations.Schema;
import com.google.adk.tools.FunctionTool;
import com.google.genai.types.Content;
import com.google.genai.types.FunctionCall;
import com.google.genai.types.GenerateContentResponseUsageMetadata;
import com.google.genai.types.Part;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.StatusCode;
import io.opentelemetry.context.Scope;
import io.opentelemetry.sdk.testing.exporter.InMemorySpanExporter;
import io.opentelemetry.sdk.trace.SdkTracerProvider;
import io.opentelemetry.sdk.trace.data.SpanData;
import io.opentelemetry.sdk.trace.export.SimpleSpanProcessor;
import io.reactivex.rxjava3.core.Flowable;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;
import net.bytebuddy.agent.ByteBuddyAgent;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/**
 * Installs the agent into the test JVM and runs a real ADK {@code Runner} with a scripted model, so
 * every advice is exercised through ADK's own call sequence rather than by calling it directly.
 */
class TelemetryAgentIntegrationTest {

    private static final String MODEL = "scripted-model";
    private static final String USER_ID = "user-123";

    private static InMemorySpanExporter exporter;
    private static SdkTracerProvider provider;

    /** The tool under test; public so ADK's reflective FunctionTool can invoke it. */
    public static final class WeatherTools {
        private WeatherTools() {}

        public static Map<String, Object> getWeather(
                @Schema(name = "city", description = "City to look up") String city) {
            if ("Atlantis".equals(city)) {
                throw new IllegalArgumentException("unknown city: " + city);
            }
            return Map.of("city", city, "forecast", "sunny", "temperature_celsius", 21);
        }
    }

    @BeforeAll
    static void installAgent() {
        exporter = InMemorySpanExporter.create();
        provider = SdkTracerProvider.builder()
                .addSpanProcessor(SimpleSpanProcessor.create(exporter))
                .build();
        // ADK reads GlobalOpenTelemetry once when Telemetry loads; the test hook avoids the global.
        Telemetry.setTracerForTesting(provider.get("gcp.vertex.agent"));
        TelemetryAgent.install(ByteBuddyAgent.install());
    }

    @BeforeEach
    void reset() {
        exporter.reset();
        OpenInferenceAgent.unregister();
        // every test starts from a clean thread context; a leak here would nest the next run
        assertThat(Span.current().getSpanContext().isValid())
                .as("no span leaked onto the test thread")
                .isFalse();
    }

    @AfterEach
    void tearDown() {
        OpenInferenceAgent.unregister();
    }

    // --- scripted model ---

    /** Replays queued responses; each call to the model consumes the next one. */
    static final class ScriptedLlm extends BaseLlm {
        final List<LlmRequest> requests = new ArrayList<>();
        private final Deque<Supplier<Flowable<LlmResponse>>> script = new ArrayDeque<>();

        ScriptedLlm() {
            super(MODEL);
        }

        ScriptedLlm reply(LlmResponse response) {
            script.add(() -> Flowable.just(response));
            return this;
        }

        ScriptedLlm fail(Throwable error) {
            script.add(() -> Flowable.error(error));
            return this;
        }

        @Override
        public Flowable<LlmResponse> generateContent(LlmRequest request, boolean stream) {
            requests.add(request);
            Supplier<Flowable<LlmResponse>> next = script.poll();
            return next == null ? Flowable.error(new IllegalStateException("no scripted response left")) : next.get();
        }

        @Override
        public BaseLlmConnection connect(LlmRequest request) {
            throw new UnsupportedOperationException("live connections are not scripted");
        }
    }

    private static LlmResponse toolCallResponse(String city) {
        return LlmResponse.builder()
                .content(Content.builder()
                        .role("model")
                        .parts(List.of(Part.builder()
                                .functionCall(FunctionCall.builder()
                                        .name("getWeather")
                                        .args(Map.of("city", city))
                                        .build())
                                .build()))
                        .build())
                .usageMetadata(GenerateContentResponseUsageMetadata.builder()
                        .promptTokenCount(50)
                        .candidatesTokenCount(10)
                        .totalTokenCount(60)
                        .build())
                .build();
    }

    private static LlmResponse textResponse(String text) {
        return LlmResponse.builder()
                .content(Content.builder()
                        .role("model")
                        .parts(List.of(Part.fromText(text)))
                        .build())
                .usageMetadata(GenerateContentResponseUsageMetadata.builder()
                        .promptTokenCount(80)
                        .candidatesTokenCount(12)
                        .totalTokenCount(92)
                        .build())
                .build();
    }

    private static LlmAgent weatherAgent(ScriptedLlm model) {
        return LlmAgent.builder()
                .name("weather_agent")
                .model(model)
                .instruction("Always call getWeather.")
                .tools(FunctionTool.create(WeatherTools.class, "getWeather"))
                .build();
    }

    private static Session newSession(InMemoryRunner runner) {
        return runner.sessionService().createSession(runner.appName(), USER_ID).blockingGet();
    }

    private static Content userMessage(String text) {
        return Content.fromParts(Part.fromText(text));
    }

    private static List<Event> run(InMemoryRunner runner, Session session, String text) {
        return runner.runAsync(USER_ID, session.id(), userMessage(text))
                .toList()
                .blockingGet();
    }

    // --- tests ---

    @Test
    void toolCallRunProducesOpenInferenceSpans() throws Exception {
        ScriptedLlm model = new ScriptedLlm().reply(toolCallResponse("Paris")).reply(textResponse("Sunny in Paris."));
        InMemoryRunner runner = new InMemoryRunner(weatherAgent(model));
        Session session = newSession(runner);

        List<Event> events = run(runner, session, "Weather in Paris?");
        assertThat(events).isNotEmpty();
        assertThat(model.requests).hasSize(2);

        List<SpanData> spans = awaitSpans(exporter, 6);
        assertThat(names(spans))
                .containsExactlyInAnyOrder(
                        "invocation",
                        "agent_run [weather_agent]",
                        "call_llm",
                        "tool_call [getWeather]",
                        "tool_response [getWeather]",
                        "call_llm");

        // invocation: CHAIN with session, user, agent and the new message
        SpanData invocation = named(spans, "invocation");
        assertThat(invocation.getParentSpanContext().isValid()).isFalse();
        assertThat(str(invocation, "openinference.span.kind")).isEqualTo("CHAIN");
        assertThat(str(invocation, "session.id")).isEqualTo(session.id());
        assertThat(str(invocation, "user.id")).isEqualTo(USER_ID);
        assertThat(str(invocation, "agent.name")).isEqualTo("weather_agent");
        assertThat(str(invocation, "input.value")).contains("Weather in Paris?");
        assertThat(str(invocation, "input.mime_type")).isEqualTo("application/json");
        assertThat(str(invocation, "output.value")).contains("Sunny in Paris.");
        assertThat(str(invocation, "metadata")).isNull();
        assertThat(invocation.getStatus().getStatusCode()).isEqualTo(StatusCode.OK);

        // agent_run: AGENT, child of invocation, with the final response as output
        SpanData agentRun = named(spans, "agent_run [weather_agent]");
        assertThat(agentRun.getParentSpanId()).isEqualTo(invocation.getSpanId());
        assertThat(str(agentRun, "openinference.span.kind")).isEqualTo("AGENT");
        assertThat(str(agentRun, "output.value")).contains("Sunny in Paris.");
        assertThat(agentRun.getStatus().getStatusCode()).isEqualTo(StatusCode.OK);

        // call_llm: first call returns a tool call, second call sees the tool result
        List<SpanData> llmCalls = allNamed(spans, "call_llm");
        assertThat(llmCalls).hasSize(2);
        llmCalls.sort((a, b) -> Long.compare(a.getStartEpochNanos(), b.getStartEpochNanos()));
        SpanData first = llmCalls.get(0);
        SpanData second = llmCalls.get(1);
        for (SpanData llm : llmCalls) {
            assertThat(llm.getParentSpanId()).isEqualTo(agentRun.getSpanId());
            assertThat(str(llm, "openinference.span.kind")).isEqualTo("LLM");
            assertThat(str(llm, "llm.model_name")).isEqualTo(MODEL);
            assertThat(str(llm, "llm.provider")).isEqualTo("google");
            assertThat(str(llm, "llm.tools.0.tool.json_schema")).contains("getWeather");
            assertThat(str(llm, "llm.input_messages.0.message.role")).isEqualTo("system");
            assertThat(str(llm, "llm.input_messages.0.message.contents.0.message_content.text"))
                    .contains("Always call getWeather.");
            assertThat(str(llm, "llm.input_messages.1.message.role")).isEqualTo("user");
            assertThat(str(llm, "llm.input_messages.1.message.contents.0.message_content.text"))
                    .isEqualTo("Weather in Paris?");
            // ADK's own attributes are left intact
            assertThat(str(llm, "gen_ai.request.model")).isEqualTo(MODEL);
            assertThat(llm.getStatus().getStatusCode()).isEqualTo(StatusCode.UNSET);
        }
        assertThat(str(first, "llm.output_messages.0.message.tool_calls.0.tool_call.function.name"))
                .isEqualTo("getWeather");
        assertThat(str(first, "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments"))
                .isEqualTo("{\"city\":\"Paris\"}");
        assertThat(lng(first, "llm.token_count.total")).isEqualTo(60);

        String callId = str(second, "llm.input_messages.2.message.tool_calls.0.tool_call.id");
        assertThat(callId).isNotBlank();
        assertThat(str(second, "llm.input_messages.2.message.role")).isEqualTo("model");
        assertThat(str(second, "llm.input_messages.3.message.role")).isEqualTo("tool");
        assertThat(str(second, "llm.input_messages.3.message.tool_call_id")).isEqualTo(callId);
        assertThat(str(second, "llm.input_messages.3.message.name")).isEqualTo("getWeather");
        assertThat(str(second, "llm.input_messages.3.message.content")).contains("\"forecast\":\"sunny\"");
        assertThat(str(second, "llm.output_messages.0.message.contents.0.message_content.text"))
                .isEqualTo("Sunny in Paris.");
        assertThat(lng(second, "llm.token_count.prompt")).isEqualTo(80);
        assertThat(lng(second, "llm.token_count.completion")).isEqualTo(12);

        // tool_call: TOOL with arguments and, from the runAsync advice, the tool result
        SpanData toolCall = named(spans, "tool_call [getWeather]");
        assertThat(toolCall.getParentSpanId()).isEqualTo(agentRun.getSpanId());
        assertThat(str(toolCall, "openinference.span.kind")).isEqualTo("TOOL");
        assertThat(str(toolCall, "tool.name")).isEqualTo("getWeather");
        assertThat(str(toolCall, "tool.parameters")).isEqualTo("{\"city\":\"Paris\"}");
        assertThat(str(toolCall, "input.value")).isEqualTo("{\"city\":\"Paris\"}");
        assertThat(str(toolCall, "output.value"))
                .contains("\"forecast\":\"sunny\"")
                .contains("\"temperature_celsius\":21");
        assertThat(str(toolCall, "output.mime_type")).isEqualTo("application/json");

        // tool_response: ADK's separate span is classified and carries the response body
        SpanData toolResponse = named(spans, "tool_response [getWeather]");
        assertThat(str(toolResponse, "openinference.span.kind")).isEqualTo("CHAIN");
        assertThat(str(toolResponse, "output.value")).contains("\"forecast\":\"sunny\"");
    }

    @Test
    void stateDeltaBecomesMetadata() throws Exception {
        ScriptedLlm model = new ScriptedLlm().reply(textResponse("Hello."));
        InMemoryRunner runner = new InMemoryRunner(weatherAgent(model));
        Session session = newSession(runner);

        runner.runAsync(
                        USER_ID,
                        session.id(),
                        userMessage("Hi"),
                        RunConfig.builder().build(),
                        Map.of("tenant", "acme"))
                .toList()
                .blockingGet();

        SpanData invocation = named(awaitSpans(exporter, 3), "invocation");
        assertThat(str(invocation, "metadata")).isEqualTo("{\"tenant\":\"acme\"}");
        assertThat(str(invocation, "openinference.span.kind")).isEqualTo("CHAIN");
    }

    @Test
    void suppressedRunsLeaveAdkSpansUndecorated() throws Exception {
        ScriptedLlm model = new ScriptedLlm().reply(toolCallResponse("Paris")).reply(textResponse("Sunny."));
        InMemoryRunner runner = new InMemoryRunner(weatherAgent(model));
        Session session = newSession(runner);

        try (Scope ignored = SuppressTracing.begin()) {
            run(runner, session, "Weather in Paris?");
        }
        assertThat(SuppressTracing.isSuppressed())
                .as("suppression scope closed cleanly")
                .isFalse();

        List<SpanData> spans = awaitSpans(exporter, 6);
        for (SpanData span : spans) {
            assertThat(str(span, "openinference.span.kind"))
                    .as("kind on %s", span.getName())
                    .isNull();
            assertThat(keysStartingWith(span, "llm."))
                    .as("llm.* on %s", span.getName())
                    .isEmpty();
            assertThat(keysStartingWith(span, "input."))
                    .as("input.* on %s", span.getName())
                    .isEmpty();
            assertThat(keysStartingWith(span, "output."))
                    .as("output.* on %s", span.getName())
                    .isEmpty();
        }
        // ADK's own telemetry is untouched
        assertThat(str(named(spans, "call_llm"), "gen_ai.request.model")).isEqualTo(MODEL);
    }

    @Test
    void traceConfigHidesInputsAcrossAllSpans() throws Exception {
        OpenInferenceAgent.register(new OITracer(
                provider.get("app"),
                TraceConfig.builder().hideInputs(true).hideToolParameters(true).build()));
        ScriptedLlm model = new ScriptedLlm().reply(toolCallResponse("Paris")).reply(textResponse("Sunny."));
        InMemoryRunner runner = new InMemoryRunner(weatherAgent(model));
        Session session = newSession(runner);

        run(runner, session, "Weather in Paris?");

        List<SpanData> spans = awaitSpans(exporter, 6);
        SpanData invocation = named(spans, "invocation");
        assertThat(str(invocation, "openinference.span.kind")).isEqualTo("CHAIN");
        assertThat(str(invocation, "session.id")).isEqualTo(session.id());
        assertThat(keysStartingWith(invocation, "input.")).isEmpty();
        assertThat(str(invocation, "output.value")).isNotNull();

        for (SpanData llm : allNamed(spans, "call_llm")) {
            assertThat(str(llm, "openinference.span.kind")).isEqualTo("LLM");
            assertThat(keysStartingWith(llm, "input.")).isEmpty();
            assertThat(keysStartingWith(llm, "llm.input_messages")).isEmpty();
            assertThat(str(llm, "llm.model_name")).isEqualTo(MODEL);
            assertThat(keysStartingWith(llm, "llm.output_messages")).isNotEmpty();
        }

        SpanData toolCall = named(spans, "tool_call [getWeather]");
        assertThat(str(toolCall, "openinference.span.kind")).isEqualTo("TOOL");
        assertThat(str(toolCall, "tool.parameters")).isNull();
        assertThat(keysStartingWith(toolCall, "input.")).isEmpty();
        assertThat(str(toolCall, "output.value")).contains("sunny");
    }

    @Test
    void modelFailureMarksInvocationAndAgentSpansAsErrors() throws Exception {
        ScriptedLlm model = new ScriptedLlm().fail(new IllegalStateException("quota exceeded"));
        InMemoryRunner runner = new InMemoryRunner(weatherAgent(model));
        Session session = newSession(runner);

        assertThatThrownBy(() -> run(runner, session, "Weather in Paris?")).hasMessageContaining("quota exceeded");

        List<SpanData> spans = awaitSpans(exporter, 3);
        for (String name : List.of("invocation", "agent_run [weather_agent]")) {
            SpanData span = named(spans, name);
            assertThat(span.getStatus().getStatusCode()).as(name).isEqualTo(StatusCode.ERROR);
            assertThat(span.getStatus().getDescription()).as(name).isNotBlank();
            assertThat(span.getEvents()).as(name).anySatisfy(e -> assertThat(e.getName())
                    .isEqualTo("exception"));
            assertThat(keysStartingWith(span, "output.")).as(name).isEmpty();
        }
        // the agent stream fails with the model's error; ADK wraps it once more at the Runner level
        assertThat(named(spans, "agent_run [weather_agent]").getStatus().getDescription())
                .isEqualTo("quota exceeded");
        SpanData llm = named(spans, "call_llm");
        // ADK records the failure itself; the advice must not have overwritten it with OK
        assertThat(llm.getStatus().getStatusCode()).isNotEqualTo(StatusCode.OK);
        assertThat(str(llm, "openinference.span.kind")).isNull();
    }

    @Test
    void toolFailureIsReportedAsToolOutput() throws Exception {
        // ADK's FunctionTool catches the exception and hands the model an error result instead
        ScriptedLlm model =
                new ScriptedLlm().reply(toolCallResponse("Atlantis")).reply(textResponse("Sorry."));
        InMemoryRunner runner = new InMemoryRunner(weatherAgent(model));
        Session session = newSession(runner);

        run(runner, session, "Weather in Atlantis?");

        List<SpanData> spans = awaitSpans(exporter, 6);
        SpanData toolCall = named(spans, "tool_call [getWeather]");
        assertThat(str(toolCall, "openinference.span.kind")).isEqualTo("TOOL");
        assertThat(str(toolCall, "tool.parameters")).isEqualTo("{\"city\":\"Atlantis\"}");
        assertThat(str(toolCall, "output.value")).containsIgnoringCase("error");
        assertThat(toolCall.getStatus().getStatusCode()).isNotEqualTo(StatusCode.OK);
        // the error result is what the model sees on the next turn
        List<SpanData> llmCalls = allNamed(spans, "call_llm");
        llmCalls.sort((a, b) -> Long.compare(a.getStartEpochNanos(), b.getStartEpochNanos()));
        assertThat(str(llmCalls.get(1), "llm.input_messages.3.message.content")).containsIgnoringCase("error");
        assertThat(named(spans, "invocation").getStatus().getStatusCode()).isEqualTo(StatusCode.OK);
    }

    @Test
    void multiTurnSessionKeepsSessionIdAcrossInvocations() throws Exception {
        ScriptedLlm model = new ScriptedLlm().reply(textResponse("Hi!")).reply(textResponse("Still here."));
        InMemoryRunner runner = new InMemoryRunner(weatherAgent(model));
        Session session = newSession(runner);

        run(runner, session, "Hello");
        run(runner, session, "Are you there?");

        List<SpanData> spans = awaitSpans(exporter, 6);
        List<SpanData> invocations = allNamed(spans, "invocation");
        assertThat(invocations).hasSize(2);
        assertThat(invocations)
                .allSatisfy(inv -> assertThat(str(inv, "session.id")).isEqualTo(session.id()));
        // each run is its own root trace: the previous invocation must not stay current on the thread
        assertThat(invocations)
                .allSatisfy(
                        inv -> assertThat(inv.getParentSpanContext().isValid()).isFalse());
        assertThat(invocations.stream().map(SpanData::getTraceId).distinct()).hasSize(2);
        assertThat(Span.current().getSpanContext().isValid()).isFalse();

        // the second LLM call sees the first turn in its history
        List<SpanData> llmCalls = allNamed(spans, "call_llm");
        llmCalls.sort((a, b) -> Long.compare(a.getStartEpochNanos(), b.getStartEpochNanos()));
        SpanData second = llmCalls.get(1);
        assertThat(str(second, "llm.input_messages.1.message.contents.0.message_content.text"))
                .isEqualTo("Hello");
        assertThat(str(second, "llm.input_messages.2.message.role")).isEqualTo("model");
        assertThat(str(second, "llm.input_messages.3.message.contents.0.message_content.text"))
                .isEqualTo("Are you there?");
    }
}

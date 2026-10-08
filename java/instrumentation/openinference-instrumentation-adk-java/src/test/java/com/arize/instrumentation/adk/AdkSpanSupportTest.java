package com.arize.instrumentation.adk;

import static com.arize.instrumentation.adk.SpanAssertions.keys;
import static com.arize.instrumentation.adk.SpanAssertions.keysStartingWith;
import static com.arize.instrumentation.adk.SpanAssertions.lng;
import static com.arize.instrumentation.adk.SpanAssertions.single;
import static com.arize.instrumentation.adk.SpanAssertions.str;
import static org.assertj.core.api.Assertions.assertThat;

import com.arize.instrumentation.OITracer;
import com.arize.instrumentation.OpenInferenceAgent;
import com.arize.instrumentation.SuppressTracing;
import com.arize.instrumentation.TraceConfig;
import com.google.adk.events.Event;
import com.google.adk.models.LlmRequest;
import com.google.adk.models.LlmResponse;
import com.google.adk.tools.Annotations.Schema;
import com.google.adk.tools.FunctionTool;
import com.google.genai.types.Content;
import com.google.genai.types.FunctionCall;
import com.google.genai.types.FunctionResponse;
import com.google.genai.types.GenerateContentConfig;
import com.google.genai.types.GenerateContentResponseUsageMetadata;
import com.google.genai.types.Part;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.StatusCode;
import io.opentelemetry.api.trace.Tracer;
import io.opentelemetry.context.Scope;
import io.opentelemetry.sdk.testing.exporter.InMemorySpanExporter;
import io.opentelemetry.sdk.trace.SdkTracerProvider;
import io.opentelemetry.sdk.trace.data.SpanData;
import io.opentelemetry.sdk.trace.export.SimpleSpanProcessor;
import io.reactivex.rxjava3.core.Flowable;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/** Unit tests for the ADK/genai to OpenInference attribute mapping and its TraceConfig handling. */
class AdkSpanSupportTest {

    private InMemorySpanExporter exporter;
    private Tracer tracer;

    /** Public so ADK's reflective FunctionTool can see the method from a package-private test. */
    public static final class WeatherTools {
        private WeatherTools() {}

        public static Map<String, Object> getWeather(@Schema(name = "city", description = "City name") String city) {
            return Map.of("city", city, "forecast", "sunny");
        }
    }

    @BeforeEach
    void setUp() {
        exporter = InMemorySpanExporter.create();
        tracer = SdkTracerProvider.builder()
                .addSpanProcessor(SimpleSpanProcessor.create(exporter))
                .build()
                .get("test");
    }

    @AfterEach
    void tearDown() {
        OpenInferenceAgent.unregister();
    }

    // --- helpers ---

    private SpanData record(String name, Consumer<Span> body) {
        Span span = tracer.spanBuilder(name).startSpan();
        try {
            body.accept(span);
        } finally {
            span.end();
        }
        return single(exporter);
    }

    private void withConfig(TraceConfig config) {
        OpenInferenceAgent.register(new OITracer(tracer, config));
    }

    private static Part functionCallPart(String id, String name, Map<String, Object> args) {
        return Part.builder()
                .functionCall(
                        FunctionCall.builder().id(id).name(name).args(args).build())
                .build();
    }

    private static Part functionResponsePart(String id, String name, Map<String, Object> response) {
        return Part.builder()
                .functionResponse(FunctionResponse.builder()
                        .id(id)
                        .name(name)
                        .response(response)
                        .build())
                .build();
    }

    private static Content content(String role, Part... parts) {
        return Content.builder().role(role).parts(List.of(parts)).build();
    }

    /** A second-turn request: system prompt, user question, model tool call, two parallel tool results. */
    private static LlmRequest toolTurnRequest() {
        return LlmRequest.builder()
                .model("gemini-2.5-flash")
                .config(GenerateContentConfig.builder()
                        .systemInstruction(Content.fromParts(Part.fromText("You are a weather bot.")))
                        .temperature(0.2f)
                        .build())
                .contents(List.of(
                        content("user", Part.fromText("Weather in Paris and Rome?")),
                        content(
                                "model",
                                functionCallPart("call-1", "getWeather", Map.of("city", "Paris")),
                                functionCallPart("call-2", "getWeather", Map.of("city", "Rome"))),
                        content(
                                "user",
                                functionResponsePart("call-1", "getWeather", Map.of("forecast", "sunny")),
                                functionResponsePart("call-2", "getWeather", Map.of("forecast", "rain")))))
                .appendTools(List.of(FunctionTool.create(WeatherTools.class, "getWeather")))
                .build();
    }

    private static LlmResponse textResponse(String text) {
        return LlmResponse.builder()
                .content(content("model", Part.fromText(text)))
                .usageMetadata(GenerateContentResponseUsageMetadata.builder()
                        .promptTokenCount(100)
                        .candidatesTokenCount(20)
                        .totalTokenCount(120)
                        .build())
                .build();
    }

    // --- LLM calls ---

    @Test
    void writeLlmCallMapsRequestAndResponse() {
        SpanData span = record(
                "call_llm", s -> AdkSpanSupport.writeLlmCall(s, toolTurnRequest(), textResponse("Sunny and rainy.")));

        assertThat(str(span, "openinference.span.kind")).isEqualTo("LLM");
        assertThat(str(span, "llm.provider")).isEqualTo("google");
        assertThat(str(span, "llm.model_name")).isEqualTo("gemini-2.5-flash");
        assertThat(str(span, "input.mime_type")).isEqualTo("application/json");
        assertThat(str(span, "input.value")).contains("Weather in Paris and Rome?");
        assertThat(str(span, "output.mime_type")).isEqualTo("application/json");
        assertThat(str(span, "output.value")).contains("Sunny and rainy.");
        assertThat(str(span, "llm.invocation_parameters")).contains("0.2");
        assertThat(str(span, "llm.tools.0.tool.json_schema"))
                .contains("\"name\":\"getWeather\"")
                .contains("city");

        // system instruction first, then the conversation
        assertThat(str(span, "llm.input_messages.0.message.role")).isEqualTo("system");
        assertThat(str(span, "llm.input_messages.0.message.contents.0.message_content.type"))
                .isEqualTo("text");
        assertThat(str(span, "llm.input_messages.0.message.contents.0.message_content.text"))
                .isEqualTo("You are a weather bot.");
        assertThat(str(span, "llm.input_messages.1.message.role")).isEqualTo("user");
        assertThat(str(span, "llm.input_messages.1.message.contents.0.message_content.text"))
                .isEqualTo("Weather in Paris and Rome?");

        // two tool calls in one model message with contiguous indices
        assertThat(str(span, "llm.input_messages.2.message.role")).isEqualTo("model");
        assertThat(str(span, "llm.input_messages.2.message.tool_calls.0.tool_call.id"))
                .isEqualTo("call-1");
        assertThat(str(span, "llm.input_messages.2.message.tool_calls.0.tool_call.function.name"))
                .isEqualTo("getWeather");
        assertThat(str(span, "llm.input_messages.2.message.tool_calls.0.tool_call.function.arguments"))
                .isEqualTo("{\"city\":\"Paris\"}");
        assertThat(str(span, "llm.input_messages.2.message.tool_calls.1.tool_call.id"))
                .isEqualTo("call-2");
        assertThat(str(span, "llm.input_messages.2.message.tool_calls.1.tool_call.function.arguments"))
                .isEqualTo("{\"city\":\"Rome\"}");

        // parallel tool results become separate tool messages, none overwritten
        assertThat(str(span, "llm.input_messages.3.message.role")).isEqualTo("tool");
        assertThat(str(span, "llm.input_messages.3.message.tool_call_id")).isEqualTo("call-1");
        assertThat(str(span, "llm.input_messages.3.message.name")).isEqualTo("getWeather");
        assertThat(str(span, "llm.input_messages.3.message.content")).isEqualTo("{\"forecast\":\"sunny\"}");
        assertThat(str(span, "llm.input_messages.4.message.role")).isEqualTo("tool");
        assertThat(str(span, "llm.input_messages.4.message.tool_call_id")).isEqualTo("call-2");
        assertThat(str(span, "llm.input_messages.4.message.content")).isEqualTo("{\"forecast\":\"rain\"}");
        assertThat(keysStartingWith(span, "llm.input_messages.5.")).isEmpty();

        assertThat(str(span, "llm.output_messages.0.message.role")).isEqualTo("model");
        assertThat(str(span, "llm.output_messages.0.message.contents.0.message_content.text"))
                .isEqualTo("Sunny and rainy.");

        assertThat(lng(span, "llm.token_count.prompt")).isEqualTo(100);
        assertThat(lng(span, "llm.token_count.completion")).isEqualTo(20);
        assertThat(lng(span, "llm.token_count.total")).isEqualTo(120);
    }

    @Test
    void writeLlmCallMapsToolCallResponse() {
        LlmResponse response = LlmResponse.builder()
                .content(content("model", functionCallPart("call-9", "getWeather", Map.of("city", "Oslo"))))
                .build();
        SpanData span = record("call_llm", s -> AdkSpanSupport.writeLlmCall(s, toolTurnRequest(), response));

        assertThat(str(span, "llm.output_messages.0.message.role")).isEqualTo("model");
        assertThat(str(span, "llm.output_messages.0.message.tool_calls.0.tool_call.id"))
                .isEqualTo("call-9");
        assertThat(str(span, "llm.output_messages.0.message.tool_calls.0.tool_call.function.name"))
                .isEqualTo("getWeather");
        assertThat(str(span, "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments"))
                .isEqualTo("{\"city\":\"Oslo\"}");
        assertThat(keysStartingWith(span, "llm.output_messages.0.message.contents"))
                .isEmpty();
        assertThat(keysStartingWith(span, "llm.token_count")).isEmpty();
    }

    @Test
    void writeLlmCallIgnoresPartialStreamChunks() {
        LlmResponse chunk = LlmResponse.builder()
                .content(content("model", Part.fromText("Sun")))
                .partial(true)
                .build();
        SpanData span = record("call_llm", s -> AdkSpanSupport.writeLlmCall(s, toolTurnRequest(), chunk));

        assertThat(keys(span)).isEmpty();
    }

    @Test
    void writeLlmCallWithoutContentWritesNoOutput() {
        LlmResponse empty = LlmResponse.builder().content(content("model")).build();
        SpanData span = record("call_llm", s -> AdkSpanSupport.writeLlmCall(s, toolTurnRequest(), empty));

        assertThat(str(span, "openinference.span.kind")).isEqualTo("LLM");
        assertThat(str(span, "input.value")).isNotNull();
        assertThat(keysStartingWith(span, "output.")).isEmpty();
        assertThat(keysStartingWith(span, "llm.output_messages")).isEmpty();
    }

    @Test
    void writeLlmCallWithoutToolsOrConfig() {
        LlmRequest request = LlmRequest.builder()
                .model("gemini-2.5-flash")
                .contents(List.of(content("user", Part.fromText("hi"))))
                .build();
        SpanData span = record("call_llm", s -> AdkSpanSupport.writeLlmCall(s, request, textResponse("hello")));

        assertThat(keysStartingWith(span, "llm.tools")).isEmpty();
        assertThat(str(span, "llm.invocation_parameters")).isNull();
        assertThat(str(span, "llm.input_messages.0.message.role")).isEqualTo("user");
        assertThat(keysStartingWith(span, "llm.input_messages.1.")).isEmpty();
    }

    // --- token usage ---

    @Test
    void writeUsageAddsThoughtsWhenNotFoldedIntoCandidates() {
        GenerateContentResponseUsageMetadata usage = GenerateContentResponseUsageMetadata.builder()
                .promptTokenCount(95)
                .candidatesTokenCount(14)
                .thoughtsTokenCount(69)
                .totalTokenCount(178)
                .build();
        SpanData span = record("call_llm", s -> AdkSpanSupport.writeUsage(s, usage));

        assertThat(lng(span, "llm.token_count.prompt")).isEqualTo(95);
        assertThat(lng(span, "llm.token_count.completion")).isEqualTo(83);
        assertThat(lng(span, "llm.token_count.completion_details.reasoning")).isEqualTo(69);
        assertThat(lng(span, "llm.token_count.total")).isEqualTo(178);
    }

    @Test
    void writeUsageDoesNotDoubleCountFoldedThoughts() {
        // prompt + candidates == total means candidates already include the thoughts
        GenerateContentResponseUsageMetadata usage = GenerateContentResponseUsageMetadata.builder()
                .promptTokenCount(10)
                .candidatesTokenCount(30)
                .thoughtsTokenCount(20)
                .totalTokenCount(40)
                .build();
        SpanData span = record("call_llm", s -> AdkSpanSupport.writeUsage(s, usage));

        assertThat(lng(span, "llm.token_count.completion")).isEqualTo(30);
        assertThat(lng(span, "llm.token_count.completion_details.reasoning")).isEqualTo(20);
    }

    @Test
    void writeUsageAddsToolUsePromptTokens() {
        GenerateContentResponseUsageMetadata usage = GenerateContentResponseUsageMetadata.builder()
                .promptTokenCount(10)
                .toolUsePromptTokenCount(5)
                .candidatesTokenCount(3)
                .totalTokenCount(18)
                .build();
        SpanData span = record("call_llm", s -> AdkSpanSupport.writeUsage(s, usage));

        assertThat(lng(span, "llm.token_count.prompt")).isEqualTo(15);
        assertThat(lng(span, "llm.token_count.completion")).isEqualTo(3);
    }

    @Test
    void writeUsageOmitsAbsentCounts() {
        SpanData span = record(
                "call_llm",
                s -> AdkSpanSupport.writeUsage(
                        s, GenerateContentResponseUsageMetadata.builder().build()));

        assertThat(keysStartingWith(span, "llm.token_count")).isEmpty();
    }

    // --- tools ---

    @Test
    void writeToolCallRecordsArguments() {
        SpanData span = record("tool_call [getWeather]", s -> AdkSpanSupport.writeToolCall(s, Map.of("city", "Paris")));

        assertThat(str(span, "openinference.span.kind")).isEqualTo("TOOL");
        assertThat(str(span, "tool.parameters")).isEqualTo("{\"city\":\"Paris\"}");
        assertThat(str(span, "input.value")).isEqualTo("{\"city\":\"Paris\"}");
        assertThat(str(span, "input.mime_type")).isEqualTo("application/json");
    }

    @Test
    void writeToolCallWithoutArgumentsOnlySetsKind() {
        SpanData span = record("tool_call [noop]", s -> AdkSpanSupport.writeToolCall(s, Map.of()));

        assertThat(keys(span)).containsExactly("openinference.span.kind");
    }

    @Test
    void writeToolInfoAndResult() {
        FunctionTool tool = FunctionTool.create(WeatherTools.class, "getWeather");
        SpanData span = record("tool_call [getWeather]", s -> {
            AdkSpanSupport.writeToolInfo(s, tool);
            AdkSpanSupport.writeToolResult(s, Map.of("forecast", "sunny"));
        });

        assertThat(str(span, "tool.name")).isEqualTo("getWeather");
        assertThat(str(span, "output.value")).isEqualTo("{\"forecast\":\"sunny\"}");
        assertThat(str(span, "output.mime_type")).isEqualTo("application/json");
    }

    @Test
    void writeToolResponseUsesSingleFunctionResponseBody() {
        Event event = Event.builder()
                .id("evt-1")
                .invocationId("inv-1")
                .author("agent")
                .content(content("user", functionResponsePart("call-1", "getWeather", Map.of("forecast", "sunny"))))
                .build();
        SpanData span = record("tool_response [getWeather]", s -> AdkSpanSupport.writeToolResponse(s, event));

        assertThat(str(span, "openinference.span.kind")).isEqualTo("CHAIN");
        assertThat(str(span, "output.value")).isEqualTo("{\"forecast\":\"sunny\"}");
    }

    @Test
    void writeToolResponseFallsBackToEventForMultipleResponses() {
        Event event = Event.builder()
                .id("evt-2")
                .invocationId("inv-1")
                .author("agent")
                .content(content(
                        "user",
                        functionResponsePart("call-1", "getWeather", Map.of("forecast", "sunny")),
                        functionResponsePart("call-2", "getWeather", Map.of("forecast", "rain"))))
                .build();
        SpanData span = record("tool_response [getWeather]", s -> AdkSpanSupport.writeToolResponse(s, event));

        assertThat(str(span, "output.value"))
                .contains("call-1")
                .contains("call-2")
                .contains("rain");
    }

    // --- flowable decoration ---

    @Test
    void decoratedFlowableRecordsFinalResponseAndAgentKind() {
        Event finalEvent = Event.builder()
                .id("evt-3")
                .invocationId("inv-1")
                .author("weather_agent")
                .content(content("model", Part.fromText("Sunny.")))
                .build();
        Span span = tracer.spanBuilder("agent_run [weather_agent]").startSpan();
        TraceFlowableAdvice.decorate(span, Flowable.just(finalEvent)).blockingSubscribe();
        span.end();

        SpanData data = single(exporter);
        assertThat(str(data, "openinference.span.kind")).isEqualTo("AGENT");
        assertThat(str(data, "output.value")).contains("Sunny.");
        assertThat(data.getStatus().getStatusCode()).isEqualTo(StatusCode.OK);
    }

    @Test
    void decoratedFlowableRecordsErrors() {
        Span span = tracer.spanBuilder("invocation").startSpan();
        Flowable<?> failing = TraceFlowableAdvice.decorate(span, Flowable.error(new IllegalStateException("boom")));
        try {
            failing.blockingSubscribe();
        } catch (RuntimeException expected) {
            // propagated to the caller as ADK would
        }
        span.end();

        SpanData data = single(exporter);
        assertThat(str(data, "openinference.span.kind")).isNull();
        assertThat(data.getStatus().getStatusCode()).isEqualTo(StatusCode.ERROR);
        assertThat(data.getStatus().getDescription()).isEqualTo("boom");
        assertThat(data.getEvents()).anySatisfy(e -> assertThat(e.getName()).isEqualTo("exception"));
    }

    // --- TraceConfig masking ---

    @Test
    void hideInputsMasksInputValueAndMessagesOnly() {
        withConfig(TraceConfig.builder().hideInputs(true).build());
        SpanData span =
                record("call_llm", s -> AdkSpanSupport.writeLlmCall(s, toolTurnRequest(), textResponse("Sunny.")));

        assertThat(keysStartingWith(span, "input.")).isEmpty();
        assertThat(keysStartingWith(span, "llm.input_messages")).isEmpty();
        assertThat(str(span, "output.value")).isNotNull();
        assertThat(str(span, "llm.output_messages.0.message.role")).isEqualTo("model");
        assertThat(str(span, "llm.model_name")).isEqualTo("gemini-2.5-flash");
    }

    @Test
    void hideOutputsMasksOutputValueAndMessagesOnly() {
        withConfig(TraceConfig.builder().hideOutputs(true).build());
        SpanData span =
                record("call_llm", s -> AdkSpanSupport.writeLlmCall(s, toolTurnRequest(), textResponse("Sunny.")));

        assertThat(keysStartingWith(span, "output.")).isEmpty();
        assertThat(keysStartingWith(span, "llm.output_messages")).isEmpty();
        assertThat(str(span, "input.value")).isNotNull();
        assertThat(lng(span, "llm.token_count.total")).isEqualTo(120);
    }

    @Test
    void hideInputTextMirrorsTracedSpanRules() {
        // TracedSpan.shouldHide masks the plain message.content but keeps structured message.contents
        withConfig(TraceConfig.builder().hideInputText(true).build());
        SpanData span =
                record("call_llm", s -> AdkSpanSupport.writeLlmCall(s, toolTurnRequest(), textResponse("Sunny.")));

        assertThat(str(span, "llm.input_messages.3.message.role")).isEqualTo("tool");
        assertThat(str(span, "llm.input_messages.3.message.content")).isNull();
        assertThat(str(span, "llm.input_messages.3.message.tool_call_id")).isEqualTo("call-1");
        assertThat(str(span, "llm.input_messages.1.message.contents.0.message_content.text"))
                .isEqualTo("Weather in Paris and Rome?");
        assertThat(str(span, "llm.output_messages.0.message.contents.0.message_content.text"))
                .isEqualTo("Sunny.");
    }

    @Test
    void hideInputMessagesKeepsInputValue() {
        withConfig(TraceConfig.builder().hideInputMessages(true).build());
        SpanData span =
                record("call_llm", s -> AdkSpanSupport.writeLlmCall(s, toolTurnRequest(), textResponse("Sunny.")));

        assertThat(keysStartingWith(span, "llm.input_messages")).isEmpty();
        assertThat(str(span, "input.value")).isNotNull();
        assertThat(str(span, "llm.output_messages.0.message.role")).isEqualTo("model");
    }

    @Test
    void hideToolParametersKeepsInputValue() {
        withConfig(TraceConfig.builder().hideToolParameters(true).build());
        SpanData span = record("tool_call [getWeather]", s -> AdkSpanSupport.writeToolCall(s, Map.of("city", "Paris")));

        assertThat(str(span, "tool.parameters")).isNull();
        assertThat(str(span, "input.value")).isEqualTo("{\"city\":\"Paris\"}");
    }

    // --- activation ---

    @Test
    void isActiveRejectsInvalidSuppressedOrDisabledSpans() {
        assertThat(AdkSpanSupport.isActive(null)).isFalse();
        assertThat(AdkSpanSupport.isActive(Span.getInvalid())).isFalse();

        Span span = tracer.spanBuilder("invocation").startSpan();
        try {
            assertThat(AdkSpanSupport.isActive(span)).isTrue();
            try (Scope ignored = SuppressTracing.begin()) {
                assertThat(AdkSpanSupport.isActive(span)).isFalse();
            }
            withConfig(TraceConfig.builder().suppressTracing(true).build());
            assertThat(AdkSpanSupport.isActive(span)).isFalse();
        } finally {
            span.end();
        }
    }

    @Test
    void toJsonNeverThrows() {
        assertThat(AdkSpanSupport.toJson(null)).isNull();
        assertThat(AdkSpanSupport.toJson(new Object())).isNull();
        assertThat(AdkSpanSupport.toJson(Map.of("a", 1))).isEqualTo("{\"a\":1}");
    }
}

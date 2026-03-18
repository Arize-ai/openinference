package com.arize.instrumentation.langchain4j;

import static com.github.tomakehurst.wiremock.client.WireMock.*;
import static org.assertj.core.api.Assertions.assertThat;

import com.arize.instrumentation.OITracer;
import com.arize.instrumentation.TraceConfig;
import com.arize.semconv.trace.SemanticConventions;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.github.tomakehurst.wiremock.WireMockServer;
import com.github.tomakehurst.wiremock.core.WireMockConfiguration;
import dev.langchain4j.model.openai.OpenAiChatModel;
import io.opentelemetry.api.common.AttributeKey;
import io.opentelemetry.api.trace.SpanKind;
import io.opentelemetry.api.trace.StatusCode;
import io.opentelemetry.sdk.testing.exporter.InMemorySpanExporter;
import io.opentelemetry.sdk.trace.SdkTracerProvider;
import io.opentelemetry.sdk.trace.data.SpanData;
import io.opentelemetry.sdk.trace.export.SimpleSpanProcessor;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class LangChain4jModelListenerTest {

    private static final ObjectMapper JSON = new ObjectMapper();

    private static WireMockServer wireMock;
    private InMemorySpanExporter spanExporter;
    private SdkTracerProvider tracerProvider;
    private OITracer oiTracer;

    @BeforeAll
    static void startWireMock() {
        wireMock = new WireMockServer(WireMockConfiguration.options().dynamicPort());
        wireMock.start();
    }

    @AfterAll
    static void stopWireMock() {
        wireMock.stop();
    }

    @BeforeEach
    void setUp() {
        wireMock.resetAll();
        spanExporter = InMemorySpanExporter.create();
        tracerProvider = SdkTracerProvider.builder()
                .addSpanProcessor(SimpleSpanProcessor.create(spanExporter))
                .build();
        oiTracer = new OITracer(tracerProvider.get("test"), TraceConfig.getDefault());
    }

    @AfterEach
    void tearDown() {
        tracerProvider.shutdown();
    }

    private OpenAiChatModel buildModel(LangChain4jModelListener listener) {
        return OpenAiChatModel.builder()
                .baseUrl("http://localhost:" + wireMock.port() + "/v1")
                .apiKey("test-key")
                .modelName("gpt-4")
                .listeners(List.of(listener))
                .timeout(Duration.ofSeconds(5))
                .build();
    }

    private void stubChatCompletion(String content) {
        wireMock.stubFor(post(urlPathEqualTo("/v1/chat/completions"))
                .willReturn(okJson(
                        """
                {
                  "id": "chatcmpl-test",
                  "object": "chat.completion",
                  "model": "gpt-4",
                  "choices": [{
                    "index": 0,
                    "message": {
                      "role": "assistant",
                      "content": "%s"
                    },
                    "finish_reason": "stop"
                  }],
                  "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30
                  }
                }
                """
                                .formatted(content))));
    }

    private void stubChatCompletionNoUsage(String content) {
        wireMock.stubFor(post(urlPathEqualTo("/v1/chat/completions"))
                .willReturn(okJson(
                        """
                {
                  "id": "chatcmpl-test",
                  "object": "chat.completion",
                  "model": "gpt-4",
                  "choices": [{
                    "index": 0,
                    "message": {
                      "role": "assistant",
                      "content": "%s"
                    },
                    "finish_reason": "stop"
                  }]
                }
                """
                                .formatted(content))));
    }

    private void stubToolCallResponse() {
        wireMock.stubFor(
                post(urlPathEqualTo("/v1/chat/completions"))
                        .willReturn(
                                okJson(
                                        """
                {
                  "id": "chatcmpl-tool",
                  "object": "chat.completion",
                  "model": "gpt-4",
                  "choices": [{
                    "index": 0,
                    "message": {
                      "role": "assistant",
                      "content": null,
                      "tool_calls": [{
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                          "name": "get_weather",
                          "arguments": "{\\"city\\": \\"London\\"}"
                        }
                      }]
                    },
                    "finish_reason": "tool_calls"
                  }],
                  "usage": {
                    "prompt_tokens": 15,
                    "completion_tokens": 25,
                    "total_tokens": 40
                  }
                }
                """)));
    }

    // ── Span and model attributes ──────────────────────────────────────

    @Test
    void setsCorrectSpanAndModelAttributes() {
        stubChatCompletion("Hello!");
        LangChain4jModelListener listener = new LangChain4jModelListener(oiTracer);
        OpenAiChatModel model = buildModel(listener);

        model.chat("Hi");

        List<SpanData> spans = spanExporter.getFinishedSpanItems();
        assertThat(spans).hasSize(1);

        SpanData span = spans.get(0);
        // Span basics
        assertThat(span.getName()).isEqualTo("generate");
        assertThat(span.getKind()).isEqualTo(SpanKind.CLIENT);
        assertThat(span.getStatus().getStatusCode()).isEqualTo(StatusCode.OK);
        // OpenInference span kind
        assertThat(span.getAttributes().get(AttributeKey.stringKey(SemanticConventions.OPENINFERENCE_SPAN_KIND)))
                .isEqualTo("LLM");
        // Model attributes
        assertThat(span.getAttributes().get(AttributeKey.stringKey(SemanticConventions.LLM_MODEL_NAME)))
                .isEqualTo("gpt-4");
        assertThat(span.getAttributes().get(AttributeKey.stringKey(SemanticConventions.LLM_SYSTEM)))
                .isEqualTo("langchain4j");
        assertThat(span.getAttributes().get(AttributeKey.stringKey(SemanticConventions.LLM_PROVIDER)))
                .isEqualTo("openai");
    }

    // ── Token counts ────────────────────────────────────────────────────

    @Test
    void capturesTokenUsage() {
        stubChatCompletion("Hello!");
        LangChain4jModelListener listener = new LangChain4jModelListener(oiTracer);
        buildModel(listener).chat("Hi");

        SpanData span = spanExporter.getFinishedSpanItems().get(0);
        assertThat(span.getAttributes().get(AttributeKey.longKey(SemanticConventions.LLM_TOKEN_COUNT_PROMPT)))
                .isEqualTo(10L);
        assertThat(span.getAttributes().get(AttributeKey.longKey(SemanticConventions.LLM_TOKEN_COUNT_COMPLETION)))
                .isEqualTo(20L);
        assertThat(span.getAttributes().get(AttributeKey.longKey(SemanticConventions.LLM_TOKEN_COUNT_TOTAL)))
                .isEqualTo(30L);
    }

    @Test
    void handlesNullTokenUsageGracefully() {
        stubChatCompletionNoUsage("Hello!");
        LangChain4jModelListener listener = new LangChain4jModelListener(oiTracer);
        buildModel(listener).chat("Hi");

        SpanData span = spanExporter.getFinishedSpanItems().get(0);
        assertThat(span.getStatus().getStatusCode()).isEqualTo(StatusCode.OK);
        // Token counts should not be set when usage is absent
        assertThat(span.getAttributes().get(AttributeKey.longKey(SemanticConventions.LLM_TOKEN_COUNT_PROMPT)))
                .isNull();
        assertThat(span.getAttributes().get(AttributeKey.longKey(SemanticConventions.LLM_TOKEN_COUNT_COMPLETION)))
                .isNull();
        assertThat(span.getAttributes().get(AttributeKey.longKey(SemanticConventions.LLM_TOKEN_COUNT_TOTAL)))
                .isNull();
    }

    // ── Input messages ──────────────────────────────────────────────────

    @Test
    void capturesInputMessagesWithValidJson() throws Exception {
        stubChatCompletion("Hello!");
        LangChain4jModelListener listener = new LangChain4jModelListener(oiTracer);
        buildModel(listener).chat("Hi there");

        SpanData span = spanExporter.getFinishedSpanItems().get(0);

        // Individual message attributes
        assertThat(span.getAttributes().get(AttributeKey.stringKey("llm.input_messages.0.message.role")))
                .isEqualTo("user");
        assertThat(span.getAttributes().get(AttributeKey.stringKey("llm.input_messages.0.message.content")))
                .contains("Hi there");

        // input.value should be valid JSON with the message
        String inputValue = span.getAttributes().get(AttributeKey.stringKey(SemanticConventions.INPUT_VALUE));
        assertThat(inputValue).isNotNull();
        List<Map<String, Object>> messages = JSON.readValue(inputValue, new TypeReference<>() {});
        assertThat(messages).isNotEmpty();
        assertThat(messages.get(0).get("role")).isEqualTo("user");
        assertThat((String) messages.get(0).get("content")).contains("Hi there");

        assertThat(span.getAttributes().get(AttributeKey.stringKey(SemanticConventions.INPUT_MIME_TYPE)))
                .isEqualTo("application/json");
    }

    @Test
    void capturesSystemAndUserInputMessages() {
        stubChatCompletion("Hello!");
        LangChain4jModelListener listener = new LangChain4jModelListener(oiTracer);
        OpenAiChatModel model = buildModel(listener);

        // Use chat with explicit messages to include a system message
        dev.langchain4j.model.chat.request.ChatRequest request =
                dev.langchain4j.model.chat.request.ChatRequest.builder()
                        .messages(
                                new dev.langchain4j.data.message.SystemMessage("You are a helpful assistant"),
                                new dev.langchain4j.data.message.UserMessage("Hello"))
                        .build();
        model.chat(request);

        SpanData span = spanExporter.getFinishedSpanItems().get(0);

        // System message at index 0
        assertThat(span.getAttributes().get(AttributeKey.stringKey("llm.input_messages.0.message.role")))
                .isEqualTo("system");
        assertThat(span.getAttributes().get(AttributeKey.stringKey("llm.input_messages.0.message.content")))
                .contains("helpful assistant");

        // User message at index 1
        assertThat(span.getAttributes().get(AttributeKey.stringKey("llm.input_messages.1.message.role")))
                .isEqualTo("user");
        assertThat(span.getAttributes().get(AttributeKey.stringKey("llm.input_messages.1.message.content")))
                .contains("Hello");
    }

    // ── Output messages ─────────────────────────────────────────────────

    @Test
    void capturesOutputMessages() {
        stubChatCompletion("The answer is 42.");
        LangChain4jModelListener listener = new LangChain4jModelListener(oiTracer);
        buildModel(listener).chat("What is the answer?");

        SpanData span = spanExporter.getFinishedSpanItems().get(0);
        assertThat(span.getAttributes().get(AttributeKey.stringKey("llm.output_messages.0.message.role")))
                .isEqualTo("assistant");
        assertThat(span.getAttributes().get(AttributeKey.stringKey("llm.output_messages.0.message.content")))
                .isEqualTo("The answer is 42.");
        assertThat(span.getAttributes().get(AttributeKey.stringKey(SemanticConventions.OUTPUT_VALUE)))
                .isEqualTo("The answer is 42.");
        assertThat(span.getAttributes().get(AttributeKey.stringKey(SemanticConventions.OUTPUT_MIME_TYPE)))
                .isEqualTo("text/plain");
    }

    // ── Tool call output ────────────────────────────────────────────────

    @Test
    void capturesToolCallsInOutputMessage() {
        stubToolCallResponse();
        LangChain4jModelListener listener = new LangChain4jModelListener(oiTracer);
        buildModel(listener).chat("What's the weather in London?");

        SpanData span = spanExporter.getFinishedSpanItems().get(0);
        String prefix = "llm.output_messages.0.";
        assertThat(span.getAttributes().get(AttributeKey.stringKey(prefix + "message.role")))
                .isEqualTo("assistant");
        assertThat(span.getAttributes()
                        .get(AttributeKey.stringKey(prefix + "message.tool_calls.0.tool_call.function.name")))
                .isEqualTo("get_weather");
        assertThat(span.getAttributes()
                        .get(AttributeKey.stringKey(prefix + "message.tool_calls.0.tool_call.function.arguments")))
                .contains("London");
        assertThat(span.getAttributes().get(AttributeKey.stringKey(prefix + "message.tool_calls.0.tool_call.id")))
                .isEqualTo("call_abc123");
    }

    @Test
    void capturesToolCallsFinishReason() {
        stubToolCallResponse();
        LangChain4jModelListener listener = new LangChain4jModelListener(oiTracer);
        buildModel(listener).chat("What's the weather?");

        SpanData span = spanExporter.getFinishedSpanItems().get(0);
        List<String> finishReasons =
                span.getAttributes().get(AttributeKey.stringArrayKey("llm.response.finish_reasons"));
        assertThat(finishReasons).isNotNull().contains("TOOL_EXECUTION");
    }

    // ── Finish reason ───────────────────────────────────────────────────

    @Test
    void capturesStopFinishReason() {
        stubChatCompletion("Done.");
        LangChain4jModelListener listener = new LangChain4jModelListener(oiTracer);
        buildModel(listener).chat("Hi");

        SpanData span = spanExporter.getFinishedSpanItems().get(0);
        List<String> finishReasons =
                span.getAttributes().get(AttributeKey.stringArrayKey("llm.response.finish_reasons"));
        assertThat(finishReasons).isNotNull().contains("STOP");
    }

    // ── Error handling ──────────────────────────────────────────────────

    @Test
    void recordsErrorOnApiFailure() {
        wireMock.stubFor(
                post(urlPathEqualTo("/v1/chat/completions"))
                        .willReturn(
                                aResponse()
                                        .withStatus(401)
                                        .withHeader("Content-Type", "application/json")
                                        .withBody(
                                                """
                        {
                          "error": {
                            "message": "Invalid API key",
                            "type": "invalid_request_error",
                            "code": "invalid_api_key"
                          }
                        }
                        """)));

        LangChain4jModelListener listener = new LangChain4jModelListener(oiTracer);
        OpenAiChatModel model = buildModel(listener);

        try {
            model.chat("Hi");
        } catch (Exception ignored) {
        }

        List<SpanData> spans = spanExporter.getFinishedSpanItems();
        assertThat(spans).hasSize(1);

        SpanData span = spans.get(0);
        assertThat(span.getStatus().getStatusCode()).isEqualTo(StatusCode.ERROR);
        assertThat(span.getStatus().getDescription()).isNotEmpty();
        assertThat(span.getEvents()).isNotEmpty(); // exception event recorded
    }

    // ── Message hiding ──────────────────────────────────────────────────

    @Test
    void hidesInputMessagesWhenConfigured() {
        stubChatCompletion("Response");
        TraceConfig config = TraceConfig.builder().hideInputMessages(true).build();
        OITracer hiddenTracer = new OITracer(tracerProvider.get("test-hidden"), config);
        OpenAiChatModel model = buildModel(new LangChain4jModelListener(hiddenTracer));
        model.chat("Secret input");

        SpanData span = spanExporter.getFinishedSpanItems().get(0);
        // Individual input message attributes should be absent
        assertThat(span.getAttributes().get(AttributeKey.stringKey("llm.input_messages.0.message.role")))
                .isNull();
        assertThat(span.getAttributes().get(AttributeKey.stringKey("llm.input_messages.0.message.content")))
                .isNull();
        // input.value should be absent
        assertThat(span.getAttributes().get(AttributeKey.stringKey(SemanticConventions.INPUT_VALUE)))
                .isNull();
        assertThat(span.getAttributes().get(AttributeKey.stringKey(SemanticConventions.INPUT_MIME_TYPE)))
                .isNull();

        // But model attributes should still be present
        assertThat(span.getAttributes().get(AttributeKey.stringKey(SemanticConventions.LLM_MODEL_NAME)))
                .isEqualTo("gpt-4");
        assertThat(span.getAttributes().get(AttributeKey.stringKey(SemanticConventions.OPENINFERENCE_SPAN_KIND)))
                .isEqualTo("LLM");
    }

    @Test
    void hidesOutputMessagesWhenConfigured() {
        stubChatCompletion("Secret response");
        TraceConfig config = TraceConfig.builder().hideOutputMessages(true).build();
        OITracer hiddenTracer = new OITracer(tracerProvider.get("test-hidden"), config);
        OpenAiChatModel model = buildModel(new LangChain4jModelListener(hiddenTracer));
        model.chat("Hi");

        SpanData span = spanExporter.getFinishedSpanItems().get(0);
        // Individual output message attributes should be absent
        assertThat(span.getAttributes().get(AttributeKey.stringKey("llm.output_messages.0.message.role")))
                .isNull();
        assertThat(span.getAttributes().get(AttributeKey.stringKey("llm.output_messages.0.message.content")))
                .isNull();
        // output.value should be absent
        assertThat(span.getAttributes().get(AttributeKey.stringKey(SemanticConventions.OUTPUT_VALUE)))
                .isNull();
        assertThat(span.getAttributes().get(AttributeKey.stringKey(SemanticConventions.OUTPUT_MIME_TYPE)))
                .isNull();

        // But token counts should still be present
        assertThat(span.getAttributes().get(AttributeKey.longKey(SemanticConventions.LLM_TOKEN_COUNT_PROMPT)))
                .isEqualTo(10L);
    }

    @Test
    void hidesBothInputAndOutputWhenConfigured() {
        stubChatCompletion("Secret");
        TraceConfig config = TraceConfig.builder()
                .hideInputMessages(true)
                .hideOutputMessages(true)
                .build();
        OITracer hiddenTracer = new OITracer(tracerProvider.get("test-hidden-both"), config);
        OpenAiChatModel model = buildModel(new LangChain4jModelListener(hiddenTracer));
        model.chat("Secret");

        SpanData span = spanExporter.getFinishedSpanItems().get(0);
        // Input hidden
        assertThat(span.getAttributes().get(AttributeKey.stringKey(SemanticConventions.INPUT_VALUE)))
                .isNull();
        // Output hidden
        assertThat(span.getAttributes().get(AttributeKey.stringKey(SemanticConventions.OUTPUT_VALUE)))
                .isNull();
        // Span is still OK
        assertThat(span.getStatus().getStatusCode()).isEqualTo(StatusCode.OK);
        // Core attributes still present
        assertThat(span.getAttributes().get(AttributeKey.stringKey(SemanticConventions.OPENINFERENCE_SPAN_KIND)))
                .isEqualTo("LLM");
        assertThat(span.getAttributes().get(AttributeKey.stringKey(SemanticConventions.LLM_MODEL_NAME)))
                .isEqualTo("gpt-4");
    }

    // ── Multiple calls ──────────────────────────────────────────────────

    @Test
    void multipleCallsCreateIndependentSpans() {
        stubChatCompletion("First");
        LangChain4jModelListener listener = new LangChain4jModelListener(oiTracer);
        OpenAiChatModel model = buildModel(listener);

        model.chat("One");

        wireMock.resetAll();
        stubChatCompletion("Second");
        model.chat("Two");

        List<SpanData> spans = spanExporter.getFinishedSpanItems();
        assertThat(spans).hasSize(2);
        assertThat(spans.get(0).getTraceId()).isNotEqualTo(spans.get(1).getTraceId());
    }

    // ── Invocation parameters ───────────────────────────────────────────

    @Test
    void capturesInvocationParameters() throws Exception {
        stubChatCompletion("Hello!");
        LangChain4jModelListener listener = new LangChain4jModelListener(oiTracer);
        OpenAiChatModel model = OpenAiChatModel.builder()
                .baseUrl("http://localhost:" + wireMock.port() + "/v1")
                .apiKey("test-key")
                .modelName("gpt-4")
                .temperature(0.7)
                .topP(0.9)
                .listeners(List.of(listener))
                .timeout(Duration.ofSeconds(5))
                .build();
        model.chat("Hi");

        SpanData span = spanExporter.getFinishedSpanItems().get(0);
        String params = span.getAttributes().get(AttributeKey.stringKey(SemanticConventions.LLM_INVOCATION_PARAMETERS));
        assertThat(params).isNotNull();

        // Parse and verify actual values
        Map<String, Object> paramsMap = JSON.readValue(params, new TypeReference<>() {});
        assertThat(((Number) paramsMap.get("temperature")).doubleValue()).isEqualTo(0.7);
        assertThat(((Number) paramsMap.get("top_p")).doubleValue()).isEqualTo(0.9);
    }

    @Test
    void invocationParametersOmitsNullValues() throws Exception {
        stubChatCompletion("Hello!");
        LangChain4jModelListener listener = new LangChain4jModelListener(oiTracer);
        // Default model — no temperature/topP/maxTokens set
        buildModel(listener).chat("Hi");

        SpanData span = spanExporter.getFinishedSpanItems().get(0);
        String params = span.getAttributes().get(AttributeKey.stringKey(SemanticConventions.LLM_INVOCATION_PARAMETERS));
        assertThat(params).isNotNull();

        Map<String, Object> paramsMap = JSON.readValue(params, new TypeReference<>() {});
        assertThat(paramsMap).doesNotContainKey("temperature");
        assertThat(paramsMap).doesNotContainKey("top_p");
        assertThat(paramsMap).doesNotContainKey("max_tokens");
    }
}

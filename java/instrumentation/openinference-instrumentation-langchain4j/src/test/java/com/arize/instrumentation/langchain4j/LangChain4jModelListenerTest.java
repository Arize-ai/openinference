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
import com.github.tomakehurst.wiremock.stubbing.Scenario;
import dev.langchain4j.agent.tool.ToolSpecification;
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

    // ── All attributes in one pass ─────────────────────────────────────

    @Test
    void capturesAllSpanAttributesForChatCompletion() throws Exception {
        stubChatCompletion("The answer is 42.");
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

        // Send system + user messages
        dev.langchain4j.model.chat.request.ChatRequest request =
                dev.langchain4j.model.chat.request.ChatRequest.builder()
                        .messages(
                                new dev.langchain4j.data.message.SystemMessage("You are a helpful assistant"),
                                new dev.langchain4j.data.message.UserMessage("What is the answer?"))
                        .build();
        model.chat(request);

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

        // Token counts
        assertThat(span.getAttributes().get(AttributeKey.longKey(SemanticConventions.LLM_TOKEN_COUNT_PROMPT)))
                .isEqualTo(10L);
        assertThat(span.getAttributes().get(AttributeKey.longKey(SemanticConventions.LLM_TOKEN_COUNT_COMPLETION)))
                .isEqualTo(20L);
        assertThat(span.getAttributes().get(AttributeKey.longKey(SemanticConventions.LLM_TOKEN_COUNT_TOTAL)))
                .isEqualTo(30L);

        // Finish reason
        List<String> finishReasons =
                span.getAttributes().get(AttributeKey.stringArrayKey("llm.response.finish_reasons"));
        assertThat(finishReasons).containsExactly("STOP");

        // Input messages: system at index 0, user at index 1
        assertThat(span.getAttributes().get(AttributeKey.stringKey("llm.input_messages.0.message.role")))
                .isEqualTo("system");
        assertThat(span.getAttributes().get(AttributeKey.stringKey("llm.input_messages.0.message.content")))
                .contains("helpful assistant");
        assertThat(span.getAttributes().get(AttributeKey.stringKey("llm.input_messages.1.message.role")))
                .isEqualTo("user");
        assertThat(span.getAttributes().get(AttributeKey.stringKey("llm.input_messages.1.message.content")))
                .contains("What is the answer?");

        // input.value is valid JSON
        String inputValue = span.getAttributes().get(AttributeKey.stringKey(SemanticConventions.INPUT_VALUE));
        assertThat(inputValue).isNotNull();
        List<Map<String, Object>> inputMessages = JSON.readValue(inputValue, new TypeReference<>() {});
        assertThat(inputMessages).hasSize(2);
        assertThat(inputMessages.get(0).get("role")).isEqualTo("system");
        assertThat(inputMessages.get(1).get("role")).isEqualTo("user");
        assertThat(span.getAttributes().get(AttributeKey.stringKey(SemanticConventions.INPUT_MIME_TYPE)))
                .isEqualTo("application/json");

        // Output messages
        assertThat(span.getAttributes().get(AttributeKey.stringKey("llm.output_messages.0.message.role")))
                .isEqualTo("assistant");
        assertThat(span.getAttributes().get(AttributeKey.stringKey("llm.output_messages.0.message.content")))
                .isEqualTo("The answer is 42.");
        assertThat(span.getAttributes().get(AttributeKey.stringKey(SemanticConventions.OUTPUT_VALUE)))
                .isEqualTo("The answer is 42.");
        assertThat(span.getAttributes().get(AttributeKey.stringKey(SemanticConventions.OUTPUT_MIME_TYPE)))
                .isEqualTo("text/plain");

        // Invocation parameters
        String params = span.getAttributes().get(AttributeKey.stringKey(SemanticConventions.LLM_INVOCATION_PARAMETERS));
        assertThat(params).isNotNull();
        Map<String, Object> paramsMap = JSON.readValue(params, new TypeReference<>() {});
        assertThat(((Number) paramsMap.get("temperature")).doubleValue()).isEqualTo(0.7);
        assertThat(((Number) paramsMap.get("top_p")).doubleValue()).isEqualTo(0.9);
    }

    // ── Tool specifications in request ─────────────────────────────────

    @Test
    void capturesToolSpecificationsInRequest() {
        stubToolCallResponse();
        LangChain4jModelListener listener = new LangChain4jModelListener(oiTracer);
        OpenAiChatModel model = buildModel(listener);

        // Send request with tool specifications
        ToolSpecification toolSpec = ToolSpecification.builder()
                .name("get_weather")
                .description("Get the current weather for a city")
                .build();

        dev.langchain4j.model.chat.request.ChatRequest request =
                dev.langchain4j.model.chat.request.ChatRequest.builder()
                        .messages(new dev.langchain4j.data.message.UserMessage("What's the weather in London?"))
                        .toolSpecifications(toolSpec)
                        .build();
        model.chat(request);

        SpanData span = spanExporter.getFinishedSpanItems().get(0);

        // Tool spec JSON schema should be set
        String toolSchema = span.getAttributes().get(AttributeKey.stringKey("llm.tools.0.tool.json_schema"));
        assertThat(toolSchema).isNotNull();
        assertThat(toolSchema).contains("get_weather");
        assertThat(toolSchema).contains("function");
        assertThat(toolSchema).contains("Get the current weather");

        // Tool call in output
        String prefix = "llm.output_messages.0.";
        assertThat(span.getAttributes()
                        .get(AttributeKey.stringKey(prefix + "message.tool_calls.0.tool_call.function.name")))
                .isEqualTo("get_weather");
        assertThat(span.getAttributes().get(AttributeKey.stringKey(prefix + "message.tool_calls.0.tool_call.id")))
                .isEqualTo("call_abc123");

        // Finish reason
        List<String> finishReasons =
                span.getAttributes().get(AttributeKey.stringArrayKey("llm.response.finish_reasons"));
        assertThat(finishReasons).containsExactly("TOOL_EXECUTION");
    }

    // ── Multi-turn tool execution ───────────────────────────────────────

    @Test
    void capturesMultiTurnToolExecution() {
        // First call: LLM requests a tool call
        wireMock.stubFor(post(urlPathEqualTo("/v1/chat/completions"))
                .inScenario("multi-turn")
                .whenScenarioStateIs(Scenario.STARTED)
                .willReturn(
                        okJson(
                                """
                {
                  "id": "chatcmpl-1",
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
                    "completion_tokens": 10,
                    "total_tokens": 25
                  }
                }
                """))
                .willSetStateTo("tool-executed"));

        // Second call: after tool result, LLM gives final answer
        wireMock.stubFor(
                post(urlPathEqualTo("/v1/chat/completions"))
                        .inScenario("multi-turn")
                        .whenScenarioStateIs("tool-executed")
                        .willReturn(
                                okJson(
                                        """
                {
                  "id": "chatcmpl-2",
                  "object": "chat.completion",
                  "model": "gpt-4",
                  "choices": [{
                    "index": 0,
                    "message": {
                      "role": "assistant",
                      "content": "The weather in London is sunny."
                    },
                    "finish_reason": "stop"
                  }],
                  "usage": {
                    "prompt_tokens": 30,
                    "completion_tokens": 10,
                    "total_tokens": 40
                  }
                }
                """)));

        LangChain4jModelListener listener = new LangChain4jModelListener(oiTracer);
        OpenAiChatModel model = buildModel(listener);

        // First turn: user message -> tool call response
        ToolSpecification toolSpec = ToolSpecification.builder()
                .name("get_weather")
                .description("Get weather")
                .build();

        dev.langchain4j.model.chat.request.ChatRequest firstRequest =
                dev.langchain4j.model.chat.request.ChatRequest.builder()
                        .messages(new dev.langchain4j.data.message.UserMessage("What's the weather in London?"))
                        .toolSpecifications(toolSpec)
                        .build();
        var firstResponse = model.chat(firstRequest);

        // Second turn: include AI tool call message + tool result
        dev.langchain4j.model.chat.request.ChatRequest secondRequest =
                dev.langchain4j.model.chat.request.ChatRequest.builder()
                        .messages(
                                new dev.langchain4j.data.message.UserMessage("What's the weather in London?"),
                                firstResponse.aiMessage(),
                                dev.langchain4j.data.message.ToolExecutionResultMessage.from(
                                        "call_abc123", "get_weather", "Sunny, 22°C"))
                        .build();
        model.chat(secondRequest);

        List<SpanData> spans = spanExporter.getFinishedSpanItems();
        assertThat(spans).hasSize(2);

        // First span: tool call response
        SpanData firstSpan = spans.get(0);
        assertThat(firstSpan
                        .getAttributes()
                        .get(AttributeKey.stringKey(
                                "llm.output_messages.0.message.tool_calls.0.tool_call.function.name")))
                .isEqualTo("get_weather");

        // Second span: input should contain AI message with tool calls and tool result
        SpanData secondSpan = spans.get(1);

        // User message at index 0
        assertThat(secondSpan.getAttributes().get(AttributeKey.stringKey("llm.input_messages.0.message.role")))
                .isEqualTo("user");

        // AI message with tool call at index 1
        assertThat(secondSpan.getAttributes().get(AttributeKey.stringKey("llm.input_messages.1.message.role")))
                .isEqualTo("assistant");
        assertThat(secondSpan
                        .getAttributes()
                        .get(AttributeKey.stringKey(
                                "llm.input_messages.1.message.tool_calls.0.tool_call.function.name")))
                .isEqualTo("get_weather");
        assertThat(secondSpan
                        .getAttributes()
                        .get(AttributeKey.stringKey("llm.input_messages.1.message.tool_calls.0.tool_call.id")))
                .isEqualTo("call_abc123");

        // Tool execution result at index 2
        assertThat(secondSpan.getAttributes().get(AttributeKey.stringKey("llm.input_messages.2.message.role")))
                .isEqualTo("tool");
        assertThat(secondSpan.getAttributes().get(AttributeKey.stringKey("llm.input_messages.2.message.tool_call_id")))
                .isEqualTo("call_abc123");

        // Final output is text
        assertThat(secondSpan.getAttributes().get(AttributeKey.stringKey("llm.output_messages.0.message.content")))
                .isEqualTo("The weather in London is sunny.");
        assertThat(secondSpan.getAttributes().get(AttributeKey.stringKey(SemanticConventions.OUTPUT_VALUE)))
                .isEqualTo("The weather in London is sunny.");
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
        assertThat(span.getEvents()).isNotEmpty();
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
        assertThat(span.getAttributes().get(AttributeKey.stringKey("llm.input_messages.0.message.role")))
                .isNull();
        assertThat(span.getAttributes().get(AttributeKey.stringKey(SemanticConventions.INPUT_VALUE)))
                .isNull();
        assertThat(span.getAttributes().get(AttributeKey.stringKey(SemanticConventions.INPUT_MIME_TYPE)))
                .isNull();

        // Model attributes still present
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
        assertThat(span.getAttributes().get(AttributeKey.stringKey("llm.output_messages.0.message.role")))
                .isNull();
        assertThat(span.getAttributes().get(AttributeKey.stringKey(SemanticConventions.OUTPUT_VALUE)))
                .isNull();
        assertThat(span.getAttributes().get(AttributeKey.stringKey(SemanticConventions.OUTPUT_MIME_TYPE)))
                .isNull();

        // Token counts still present
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
        assertThat(span.getAttributes().get(AttributeKey.stringKey(SemanticConventions.INPUT_VALUE)))
                .isNull();
        assertThat(span.getAttributes().get(AttributeKey.stringKey(SemanticConventions.OUTPUT_VALUE)))
                .isNull();
        assertThat(span.getStatus().getStatusCode()).isEqualTo(StatusCode.OK);
        assertThat(span.getAttributes().get(AttributeKey.stringKey(SemanticConventions.OPENINFERENCE_SPAN_KIND)))
                .isEqualTo("LLM");
    }
}

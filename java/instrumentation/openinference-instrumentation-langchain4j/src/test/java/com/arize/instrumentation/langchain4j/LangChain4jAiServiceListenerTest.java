package com.arize.instrumentation.langchain4j;

import static com.github.tomakehurst.wiremock.client.WireMock.okJson;
import static org.assertj.core.api.Assertions.assertThat;

import com.arize.instrumentation.TraceConfig;
import com.arize.semconv.trace.SemanticConventions;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.github.tomakehurst.wiremock.client.WireMock;
import com.github.tomakehurst.wiremock.stubbing.Scenario;
import dev.langchain4j.agent.tool.Tool;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.service.AiServices;
import io.opentelemetry.api.common.AttributeKey;
import io.opentelemetry.api.trace.SpanKind;
import io.opentelemetry.api.trace.StatusCode;
import io.opentelemetry.sdk.trace.data.SpanData;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.time.Duration;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInfo;

class LangChain4jAiServiceListenerTest extends BaseInstrumentationSetup {
    ObjectMapper objectMapper = new ObjectMapper();

    // Utility function for constructing tool schema attribute keys
    private static String getToolSchemaAttributeKey(int toolIndex) {
        return "llm.tools." + toolIndex + ".tool.json_schema";
    }

    public void stubForOpenAiChatModelCalls() throws Exception {
        String cassettePath = CASSETTES_ROOT + "LangChain4jAiServiceListenerTest/";
        String initialCallContent = Files.readString(Paths.get(cassettePath + "v1_chat_completions-1st-call.json"));
        String toolResultCallContent = Files.readString(Paths.get(cassettePath + "v1_chat_completions-2nd-call.json"));
        wireMock.stubFor(
                com.github.tomakehurst.wiremock.client.WireMock.post(WireMock.urlEqualTo("/v1/chat/completions"))
                        .atPriority(1)
                        .inScenario("multiple tool calls")
                        .whenScenarioStateIs(Scenario.STARTED)
                        .willReturn(okJson(initialCallContent))
                        .willSetStateTo("SECOND_CALL"));
        wireMock.stubFor(
                com.github.tomakehurst.wiremock.client.WireMock.post(WireMock.urlEqualTo("/v1/chat/completions"))
                        .atPriority(1)
                        .inScenario("multiple tool calls")
                        .whenScenarioStateIs("SECOND_CALL")
                        .willReturn(okJson(toolResultCallContent)));
    }

    public void stubForOpenAiChatErrorCalls() throws Exception {
        wireMock.stubFor(
                com.github.tomakehurst.wiremock.client.WireMock.post(WireMock.urlEqualTo("/v1/chat/completions"))
                        .atPriority(1)
                        .willReturn(
                                WireMock.aResponse()
                                        .withStatus(401)
                                        .withHeader("Content-Type", "application/json")
                                        .withBody(
                                                """
                {
                  "error": {
                    "message": "Incorrect API key provided: test-api-key. You can find your API key at https://platform.openai.com/account/api-keys.",
                    "type": "invalid_request_error",
                    "param": null,
                    "code": "invalid_api_key"
                  }
                }
            """)));
    }

    public MathAssistant createMathAssistant(TestInfo testInfo, LangChain4jInstrumentor instrumentor) {
        // Each test method gets its own cassette named after the method
        String apiKey = "test-api-key";
        OpenAiChatModel chatModel = OpenAiChatModel.builder()
                .apiKey(apiKey)
                .baseUrl("http://localhost:" + wireMock.port() + "/v1")
                .modelName("gpt-4.1-nano")
                .temperature(1.0)
                .maxCompletionTokens(300)
                .timeout(Duration.ofSeconds(30))
                .build();

        return AiServices.builder(MathAssistant.class)
                .chatModel(chatModel)
                .systemMessage("You are a helpful assistant that can perform basic math calculations. "
                        + "Use the provided tools to answer user questions accurately.")
                .tools(new MathTools())
                .registerListeners(instrumentor.createAiServiceListeners())
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .build();
    }

    @SuppressWarnings("unchecked")
    public String getToolName(String toolJsonSchema) throws JsonProcessingException {
        Map<String, Object> root = objectMapper.readValue(toolJsonSchema, Map.class);
        Map<String, Object> function = (Map<String, Object>) root.get("function");
        return (String) function.get("name");
    }

    @Test
    void aiServicesWithToolsTest(TestInfo testInfo) throws Exception {
        stubForOpenAiChatModelCalls();
        LangChain4jInstrumentor instrumentor =
                LangChain4jInstrumentor.instrument(tracerProvider, TraceConfig.getDefault());
        try {
            MathAssistant assistant = createMathAssistant(testInfo, instrumentor);
            String answer = assistant.chat("What is 45 plus 79?");
            assertThat(answer).containsIgnoringCase("45 plus 79 is 124.");
        } finally {
            instrumentor.uninstrument();
        }

        List<SpanData> spans = spanExporter.getFinishedSpanItems();
        assertThat(spans).hasSize(4);

        // Verify First LLM Span Attributes
        SpanData llmSpan = spans.get(0);
        assertThat(llmSpan.getName()).isEqualTo("LLM");
        assertThat(llmSpan.getKind()).isEqualTo(SpanKind.CLIENT);
        assertThat(llmSpan.getStatus().getStatusCode()).isEqualTo(StatusCode.OK);

        Map<AttributeKey<?>, Object> attrs =
                new HashMap<>(llmSpan.getAttributes().asMap());

        assertThat(attrs.remove(AttributeKey.stringKey(SemanticConventions.OPENINFERENCE_SPAN_KIND)))
                .isEqualTo(SemanticConventions.OpenInferenceSpanKind.LLM.toString());
        assertThat((String) attrs.remove(AttributeKey.stringKey(SemanticConventions.INPUT_VALUE)))
                .containsIgnoringCase("What is 45 plus 79?");
        assertThat(attrs.remove(AttributeKey.stringKey(SemanticConventions.INPUT_MIME_TYPE)))
                .isEqualTo("application/json");
        assertThat(attrs.remove(AttributeKey.stringKey(SemanticConventions.OUTPUT_MIME_TYPE)))
                .isEqualTo("text/plain");
        assertThat(Objects.isNull(
                        llmSpan.getAttributes().get(AttributeKey.stringKey(SemanticConventions.OUTPUT_VALUE))))
                .isTrue(); // LLM span should not contain the final answer since tool calls are involved
        // Model attributes
        assertThat(attrs.remove(AttributeKey.stringKey(SemanticConventions.LLM_MODEL_NAME)))
                .isEqualTo("unknown");
        assertThat(attrs.remove(AttributeKey.stringKey(SemanticConventions.LLM_SYSTEM)))
                .isEqualTo("langchain4j");
        assertThat(attrs.remove(AttributeKey.stringKey(SemanticConventions.LLM_PROVIDER)))
                .isEqualTo("openai");

        // Token counts
        assertThat(attrs.remove(AttributeKey.longKey(SemanticConventions.LLM_TOKEN_COUNT_PROMPT)))
                .isEqualTo(162L);
        assertThat(attrs.remove(AttributeKey.longKey(SemanticConventions.LLM_TOKEN_COUNT_COMPLETION)))
                .isEqualTo(17L);
        assertThat(attrs.remove(AttributeKey.longKey(SemanticConventions.LLM_TOKEN_COUNT_TOTAL)))
                .isEqualTo(179L);

        // Finish reason
        Object value = attrs.remove(AttributeKey.stringArrayKey("llm.response.finish_reasons"));

        assertThat(value).isInstanceOf(List.class);

        // Input messages: system at index 0, user at index 1
        assertThat(attrs.remove(AttributeKey.stringKey("llm.input_messages.0.message.role")))
                .isEqualTo("system");
        assertThat((String) attrs.remove(AttributeKey.stringKey("llm.input_messages.0.message.content")))
                .contains("helpful assistant");
        assertThat((String) attrs.remove(AttributeKey.stringKey("llm.input_messages.1.message.role")))
                .isEqualTo("user");
        assertThat((String) attrs.remove(AttributeKey.stringKey("llm.input_messages.1.message.content")))
                .contains("What is 45 plus 79?");

        // Output messages
        assertThat(attrs.remove(AttributeKey.stringKey("llm.output_messages.0.message.role")))
                .isEqualTo("assistant");
        assertThat(attrs.remove(AttributeKey.stringKey("llm.output_messages.0.message.tool_calls.0.tool_call.id")))
                .isEqualTo("call_Qm51xgE57uzoj2f6qkZAWwb4");
        assertThat(attrs.remove(
                        AttributeKey.stringKey("llm.output_messages.0.message.tool_calls.0.tool_call.function.name")))
                .isEqualTo("add");
        assertThat(attrs.remove(AttributeKey.stringKey(
                        "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments")))
                .isEqualTo("{\"a\":45,\"b\":79}");

        // Tool schemas
        List<String> expectedTools = List.of("add", "multiply", "divide", "subtract");
        List<String> actualTools = List.of(
                getToolName((String) attrs.remove(AttributeKey.stringKey(getToolSchemaAttributeKey(0)))),
                getToolName((String) attrs.remove(AttributeKey.stringKey(getToolSchemaAttributeKey(1)))),
                getToolName((String) attrs.remove(AttributeKey.stringKey(getToolSchemaAttributeKey(2)))),
                getToolName((String) attrs.remove(AttributeKey.stringKey(getToolSchemaAttributeKey(3)))));
        assertThat(actualTools).containsExactlyInAnyOrderElementsOf(expectedTools);
        assertThat((String) attrs.remove(AttributeKey.stringKey(SemanticConventions.LLM_INVOCATION_PARAMETERS)))
                .contains("{}");
        assertThat(attrs).isEmpty();

        // Verify TooL Call Span Attributes
        SpanData toolSpan = spans.get(1);
        assertThat(toolSpan.getName()).isEqualTo("AiService.tool");
        assertThat(toolSpan.getKind()).isEqualTo(SpanKind.CLIENT);
        assertThat(toolSpan.getStatus().getStatusCode()).isEqualTo(StatusCode.OK);

        Map<AttributeKey<?>, Object> toolAttrs =
                new HashMap<>(toolSpan.getAttributes().asMap());

        assertThat(toolAttrs.remove(AttributeKey.stringKey(SemanticConventions.OPENINFERENCE_SPAN_KIND)))
                .isEqualTo(SemanticConventions.OpenInferenceSpanKind.TOOL.toString());
        assertThat((String) toolAttrs.remove(AttributeKey.stringKey(SemanticConventions.INPUT_VALUE)))
                .containsIgnoringCase("{\"a\":45,\"b\":79}");
        assertThat(toolAttrs.remove(AttributeKey.stringKey(SemanticConventions.INPUT_MIME_TYPE)))
                .isEqualTo("application/json");
        assertThat(toolAttrs.remove(AttributeKey.stringKey(SemanticConventions.OUTPUT_MIME_TYPE)))
                .isEqualTo("text/plain");
        assertThat(toolAttrs.remove(AttributeKey.stringKey(SemanticConventions.OUTPUT_VALUE)))
                .isEqualTo("124.0");
        assertThat(toolAttrs.remove(AttributeKey.stringKey(SemanticConventions.TOOL_CALL_ID)))
                .isEqualTo("call_Qm51xgE57uzoj2f6qkZAWwb4");
        assertThat(toolAttrs.remove(AttributeKey.stringKey(SemanticConventions.TOOL_NAME)))
                .isEqualTo("add");
        assertThat(toolAttrs.remove(AttributeKey.stringKey(SemanticConventions.TOOL_PARAMETERS)))
                .isEqualTo("{\"a\":45,\"b\":79}");
        assertThat(toolAttrs).isEmpty();

        // Verify Final LLM Span Attributes
        SpanData finalLlmSpan = spans.get(2);
        assertThat(finalLlmSpan.getName()).isEqualTo("LLM");
        assertThat(finalLlmSpan.getKind()).isEqualTo(SpanKind.CLIENT);
        assertThat(finalLlmSpan.getStatus().getStatusCode()).isEqualTo(StatusCode.OK);

        Map<AttributeKey<?>, Object> finalLlmAttrs =
                new HashMap<>(finalLlmSpan.getAttributes().asMap());

        assertThat(finalLlmAttrs.remove(AttributeKey.stringKey(SemanticConventions.OPENINFERENCE_SPAN_KIND)))
                .isEqualTo(SemanticConventions.OpenInferenceSpanKind.LLM.toString());
        String inputValue = (String) finalLlmAttrs.remove(AttributeKey.stringKey(SemanticConventions.INPUT_VALUE));
        assertThat(inputValue).containsIgnoringCase("What is 45 plus 79?");
        assertThat(inputValue)
                .containsIgnoringCase(
                        "\"role\":\"tool\",\"content\":\"124.0\",\"message.tool_call_id\":\"call_Qm51xgE57uzoj2f6qkZAWwb4\"");
        assertThat(finalLlmAttrs.remove(AttributeKey.stringKey(SemanticConventions.INPUT_MIME_TYPE)))
                .isEqualTo("application/json");
        assertThat(finalLlmAttrs.remove(AttributeKey.stringKey(SemanticConventions.OUTPUT_MIME_TYPE)))
                .isEqualTo("text/plain");
        assertThat(finalLlmAttrs.remove(AttributeKey.stringKey(SemanticConventions.OUTPUT_VALUE)))
                .isEqualTo("The sum of 45 plus 79 is 124.");
        // Model attributes
        assertThat(finalLlmAttrs.remove(AttributeKey.stringKey(SemanticConventions.LLM_MODEL_NAME)))
                .isEqualTo("unknown");
        assertThat(finalLlmAttrs.remove(AttributeKey.stringKey(SemanticConventions.LLM_SYSTEM)))
                .isEqualTo("langchain4j");
        assertThat(finalLlmAttrs.remove(AttributeKey.stringKey(SemanticConventions.LLM_PROVIDER)))
                .isEqualTo("openai");

        // Token counts
        assertThat(finalLlmAttrs.remove(AttributeKey.longKey(SemanticConventions.LLM_TOKEN_COUNT_PROMPT)))
                .isEqualTo(189L);
        assertThat(finalLlmAttrs.remove(AttributeKey.longKey(SemanticConventions.LLM_TOKEN_COUNT_COMPLETION)))
                .isEqualTo(13L);
        assertThat(finalLlmAttrs.remove(AttributeKey.longKey(SemanticConventions.LLM_TOKEN_COUNT_TOTAL)))
                .isEqualTo(202L);

        // Finish reason
        List<String> finishReasons =
                (List<String>) finalLlmAttrs.remove(AttributeKey.stringArrayKey("llm.response.finish_reasons"));
        assertThat(finishReasons).containsExactly("STOP");

        // Input messages: system at index 0, user at index 1
        assertThat(finalLlmAttrs.remove(AttributeKey.stringKey("llm.input_messages.0.message.role")))
                .isEqualTo("system");
        assertThat((String) finalLlmAttrs.remove(AttributeKey.stringKey("llm.input_messages.0.message.content")))
                .contains("helpful assistant");
        assertThat(finalLlmAttrs.remove(AttributeKey.stringKey("llm.input_messages.1.message.role")))
                .isEqualTo("user");
        assertThat((String) finalLlmAttrs.remove(AttributeKey.stringKey("llm.input_messages.1.message.content")))
                .contains("What is 45 plus 79?");

        assertThat(finalLlmAttrs.remove(AttributeKey.stringKey("llm.input_messages.2.message.role")))
                .isEqualTo("assistant");
        assertThat((String) finalLlmAttrs.remove(AttributeKey.stringKey("llm.input_messages.2.message.content")))
                .contains("ToolExecutionRequest");
        assertThat(finalLlmAttrs.remove(
                        AttributeKey.stringKey("llm.input_messages.2.message.tool_calls.0.tool_call.id")))
                .isEqualTo("call_Qm51xgE57uzoj2f6qkZAWwb4");
        assertThat(finalLlmAttrs.remove(
                        AttributeKey.stringKey("llm.input_messages.2.message.tool_calls.0.tool_call.function.name")))
                .isEqualTo("add");
        assertThat(finalLlmAttrs.remove(AttributeKey.stringKey(
                        "llm.input_messages.2.message.tool_calls.0.tool_call.function.arguments")))
                .isEqualTo("{\"a\":45,\"b\":79}");

        assertThat(finalLlmAttrs.remove(AttributeKey.stringKey("llm.input_messages.3.message.role")))
                .isEqualTo("tool");
        assertThat(finalLlmAttrs.remove(AttributeKey.stringKey("llm.input_messages.3.message.tool_call_id")))
                .isEqualTo("call_Qm51xgE57uzoj2f6qkZAWwb4");
        assertThat((String) finalLlmAttrs.remove(AttributeKey.stringKey("llm.input_messages.3.message.content")))
                .containsIgnoringCase("call_Qm51xgE57uzoj2f6qkZAWwb4");

        // Output messages
        assertThat(finalLlmAttrs.remove(AttributeKey.stringKey("llm.output_messages.0.message.role")))
                .isEqualTo("assistant");
        assertThat(finalLlmAttrs.remove(AttributeKey.stringKey("llm.output_messages.0.message.content")))
                .isEqualTo("The sum of 45 plus 79 is 124.");

        // Tool schemas
        actualTools = List.of(
                getToolName((String) finalLlmAttrs.remove(AttributeKey.stringKey(getToolSchemaAttributeKey(0)))),
                getToolName((String) finalLlmAttrs.remove(AttributeKey.stringKey(getToolSchemaAttributeKey(1)))),
                getToolName((String) finalLlmAttrs.remove(AttributeKey.stringKey(getToolSchemaAttributeKey(2)))),
                getToolName((String) finalLlmAttrs.remove(AttributeKey.stringKey(getToolSchemaAttributeKey(3)))));
        assertThat(actualTools).containsExactlyInAnyOrderElementsOf(expectedTools);
        assertThat((String) finalLlmAttrs.remove(AttributeKey.stringKey(SemanticConventions.LLM_INVOCATION_PARAMETERS)))
                .contains("{}");
        assertThat(finalLlmAttrs).isEmpty();

        // Verify Agent Span Attributes
        SpanData agentSpan = spans.get(3);
        assertThat(agentSpan.getName()).isEqualTo("AiService.chat");
        assertThat(agentSpan.getKind()).isEqualTo(SpanKind.CLIENT);
        assertThat(agentSpan.getStatus().getStatusCode()).isEqualTo(StatusCode.OK);

        Map<AttributeKey<?>, Object> agentAttrs =
                new HashMap<>(agentSpan.getAttributes().asMap());

        assertThat(agentAttrs.remove(AttributeKey.stringKey(SemanticConventions.OPENINFERENCE_SPAN_KIND)))
                .isEqualTo(SemanticConventions.OpenInferenceSpanKind.AGENT.toString());
        assertThat((String) agentAttrs.remove(AttributeKey.stringKey(SemanticConventions.INPUT_VALUE)))
                .containsIgnoringCase("What is 45 plus 79?");
        assertThat(agentAttrs.remove(AttributeKey.stringKey(SemanticConventions.INPUT_MIME_TYPE)))
                .isEqualTo("application/json");
        assertThat(agentAttrs.remove(AttributeKey.stringKey(SemanticConventions.OUTPUT_MIME_TYPE)))
                .isEqualTo("text/plain");
        assertThat(agentAttrs.remove(AttributeKey.stringKey(SemanticConventions.OUTPUT_VALUE)))
                .isEqualTo("The sum of 45 plus 79 is 124.");
        assertThat(agentAttrs).isEmpty();
    }

    @Test
    void traceConfigInputOutputMessagesDisabled(TestInfo testInfo) throws Exception {
        stubForOpenAiChatModelCalls();
        TraceConfig config = TraceConfig.builder()
                .hideInputMessages(true)
                .hideOutputMessages(true)
                .build();
        LangChain4jInstrumentor instrumentor = LangChain4jInstrumentor.instrument(tracerProvider, config);
        try {
            MathAssistant assistant = createMathAssistant(testInfo, instrumentor);
            String answer = assistant.chat("What is 45 plus 79?");
            assertThat(answer).containsIgnoringCase("45 plus 79 is 124.");
        } finally {
            instrumentor.uninstrument();
        }
        List<SpanData> spans = spanExporter.getFinishedSpanItems();
        instrumentor.uninstrument();
        assertThat(spans).hasSize(4);

        // Verify First LLM Span Attributes
        SpanData llmSpan = spans.get(0);
        assertThat(Objects.isNull(llmSpan.getAttributes().get(AttributeKey.stringKey(SemanticConventions.INPUT_VALUE))))
                .isTrue();
        assertThat(Objects.isNull(
                        llmSpan.getAttributes().get(AttributeKey.stringKey(SemanticConventions.INPUT_MIME_TYPE))))
                .isTrue();

        assertThat(Objects.isNull(
                        llmSpan.getAttributes().get(AttributeKey.stringKey(SemanticConventions.OUTPUT_VALUE))))
                .isTrue();
        assertThat(Objects.isNull(
                        llmSpan.getAttributes().get(AttributeKey.stringKey(SemanticConventions.OUTPUT_MIME_TYPE))))
                .isTrue();

        // Verify Final LLM Span Attributes
        SpanData finalLlmSpan = spans.get(2);
        assertThat(Objects.isNull(
                        finalLlmSpan.getAttributes().get(AttributeKey.stringKey(SemanticConventions.INPUT_VALUE))))
                .isTrue();
        assertThat(Objects.isNull(
                        finalLlmSpan.getAttributes().get(AttributeKey.stringKey(SemanticConventions.INPUT_MIME_TYPE))))
                .isTrue();

        assertThat(Objects.isNull(
                        finalLlmSpan.getAttributes().get(AttributeKey.stringKey(SemanticConventions.OUTPUT_VALUE))))
                .isTrue();
        assertThat(Objects.isNull(
                        finalLlmSpan.getAttributes().get(AttributeKey.stringKey(SemanticConventions.OUTPUT_MIME_TYPE))))
                .isTrue();

        // Verify Agent Span Attributes
        SpanData agentSpan = spans.get(3);
        assertThat(Objects.isNull(
                        agentSpan.getAttributes().get(AttributeKey.stringKey(SemanticConventions.INPUT_VALUE))))
                .isTrue();
        assertThat(Objects.isNull(
                        agentSpan.getAttributes().get(AttributeKey.stringKey(SemanticConventions.INPUT_MIME_TYPE))))
                .isTrue();
        assertThat(Objects.isNull(
                        agentSpan.getAttributes().get(AttributeKey.stringKey(SemanticConventions.OUTPUT_VALUE))))
                .isTrue();
        assertThat(Objects.isNull(
                        agentSpan.getAttributes().get(AttributeKey.stringKey(SemanticConventions.OUTPUT_MIME_TYPE))))
                .isTrue();
    }

    @Test
    void hidesInputMessagesWhenConfigured(TestInfo testInfo) throws Exception {
        stubForOpenAiChatModelCalls();
        TraceConfig config = TraceConfig.builder().hideInputMessages(true).build();
        LangChain4jInstrumentor instrumentor = LangChain4jInstrumentor.instrument(tracerProvider, config);
        MathAssistant assistant = createMathAssistant(testInfo, instrumentor);
        String answer = assistant.chat("What is 45 plus 79?");
        assertThat(answer).containsIgnoringCase("The sum of 45 plus 79 is 124.");
        List<SpanData> spans = spanExporter.getFinishedSpanItems();
        assertThat(spans).hasSize(4);
        instrumentor.uninstrument();

        // Verify First LLM Span Attributes
        SpanData llmSpan = spans.get(0);
        assertThat(llmSpan.getName()).isEqualTo("LLM");
        assertThat(llmSpan.getKind()).isEqualTo(SpanKind.CLIENT);
        assertThat(llmSpan.getStatus().getStatusCode()).isEqualTo(StatusCode.OK);

        Map<AttributeKey<?>, Object> llmAttrs =
                new HashMap<>(llmSpan.getAttributes().asMap());

        assertThat(llmAttrs.remove(AttributeKey.stringKey(SemanticConventions.OPENINFERENCE_SPAN_KIND)))
                .isEqualTo(SemanticConventions.OpenInferenceSpanKind.LLM.toString());
        assertThat(Objects.isNull(llmAttrs.remove(AttributeKey.stringKey(SemanticConventions.INPUT_VALUE))))
                .isTrue();
        assertThat(Objects.isNull(llmAttrs.remove(AttributeKey.stringKey(SemanticConventions.INPUT_MIME_TYPE))))
                .isTrue();
        assertThat(llmAttrs.remove(AttributeKey.stringKey(SemanticConventions.OUTPUT_MIME_TYPE)))
                .isEqualTo("text/plain");
        assertThat(Objects.isNull(llmAttrs.remove(AttributeKey.stringKey(SemanticConventions.OUTPUT_VALUE))))
                .isTrue(); // LLM span should not contain the final answer since tool calls are involved
        // Model attributes
        assertThat(llmAttrs.remove(AttributeKey.stringKey(SemanticConventions.LLM_MODEL_NAME)))
                .isEqualTo("unknown");
        assertThat(llmAttrs.remove(AttributeKey.stringKey(SemanticConventions.LLM_SYSTEM)))
                .isEqualTo("langchain4j");
        assertThat(llmAttrs.remove(AttributeKey.stringKey(SemanticConventions.LLM_PROVIDER)))
                .isEqualTo("openai");

        // Token counts
        assertThat(llmAttrs.remove(AttributeKey.longKey(SemanticConventions.LLM_TOKEN_COUNT_PROMPT)))
                .isEqualTo(162L);
        assertThat(llmAttrs.remove(AttributeKey.longKey(SemanticConventions.LLM_TOKEN_COUNT_COMPLETION)))
                .isEqualTo(17L);
        assertThat(llmAttrs.remove(AttributeKey.longKey(SemanticConventions.LLM_TOKEN_COUNT_TOTAL)))
                .isEqualTo(179L);

        // Finish reason
        List<String> finishReasons =
                (List<String>) llmAttrs.remove(AttributeKey.stringArrayKey("llm.response.finish_reasons"));
        assertThat(finishReasons).containsExactly("TOOL_EXECUTION");

        // Output messages
        assertThat(llmAttrs.remove(AttributeKey.stringKey("llm.output_messages.0.message.role")))
                .isEqualTo("assistant");
        assertThat(llmAttrs.remove(AttributeKey.stringKey("llm.output_messages.0.message.tool_calls.0.tool_call.id")))
                .isEqualTo("call_Qm51xgE57uzoj2f6qkZAWwb4");
        assertThat(llmAttrs.remove(
                        AttributeKey.stringKey("llm.output_messages.0.message.tool_calls.0.tool_call.function.name")))
                .isEqualTo("add");
        assertThat(llmAttrs.remove(AttributeKey.stringKey(
                        "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments")))
                .isEqualTo("{\"a\":45,\"b\":79}");

        // Tool schemas
        List<String> expectedTools = List.of("add", "multiply", "divide", "subtract");
        List<String> actualTools = List.of(
                getToolName((String) llmAttrs.remove(AttributeKey.stringKey(getToolSchemaAttributeKey(0)))),
                getToolName((String) llmAttrs.remove(AttributeKey.stringKey(getToolSchemaAttributeKey(1)))),
                getToolName((String) llmAttrs.remove(AttributeKey.stringKey(getToolSchemaAttributeKey(2)))),
                getToolName((String) llmAttrs.remove(AttributeKey.stringKey(getToolSchemaAttributeKey(3)))));
        assertThat(actualTools).containsExactlyInAnyOrderElementsOf(expectedTools);
        assertThat((String) llmAttrs.remove(AttributeKey.stringKey(SemanticConventions.LLM_INVOCATION_PARAMETERS)))
                .contains("{}");
        assertThat(llmAttrs).isEmpty();

        // Verify TooL Call Span Attributes
        SpanData toolSpan = spans.get(1);
        assertThat(toolSpan.getName()).isEqualTo("AiService.tool");
        assertThat(toolSpan.getKind()).isEqualTo(SpanKind.CLIENT);
        assertThat(toolSpan.getStatus().getStatusCode()).isEqualTo(StatusCode.OK);

        Map<AttributeKey<?>, Object> toolAttrs =
                new HashMap<>(toolSpan.getAttributes().asMap());

        assertThat(toolAttrs.remove(AttributeKey.stringKey(SemanticConventions.OPENINFERENCE_SPAN_KIND)))
                .isEqualTo(SemanticConventions.OpenInferenceSpanKind.TOOL.toString());
        assertThat((String) toolAttrs.remove(AttributeKey.stringKey(SemanticConventions.INPUT_VALUE)))
                .containsIgnoringCase("{\"a\":45,\"b\":79}");
        assertThat(toolAttrs.remove(AttributeKey.stringKey(SemanticConventions.INPUT_MIME_TYPE)))
                .isEqualTo("application/json");
        assertThat(toolAttrs.remove(AttributeKey.stringKey(SemanticConventions.OUTPUT_MIME_TYPE)))
                .isEqualTo("text/plain");
        assertThat(toolAttrs.remove(AttributeKey.stringKey(SemanticConventions.OUTPUT_VALUE)))
                .isEqualTo("124.0");
        assertThat(toolAttrs.remove(AttributeKey.stringKey(SemanticConventions.TOOL_CALL_ID)))
                .isEqualTo("call_Qm51xgE57uzoj2f6qkZAWwb4");
        assertThat(toolAttrs.remove(AttributeKey.stringKey(SemanticConventions.TOOL_NAME)))
                .isEqualTo("add");
        assertThat(toolAttrs.remove(AttributeKey.stringKey(SemanticConventions.TOOL_PARAMETERS)))
                .isEqualTo("{\"a\":45,\"b\":79}");
        assertThat(toolAttrs).isEmpty();

        // Verify Final LLM Span Attributes
        SpanData finalLlmSpan = spans.get(2);
        assertThat(finalLlmSpan.getName()).isEqualTo("LLM");
        assertThat(finalLlmSpan.getKind()).isEqualTo(SpanKind.CLIENT);
        assertThat(finalLlmSpan.getStatus().getStatusCode()).isEqualTo(StatusCode.OK);

        Map<AttributeKey<?>, Object> finalLlmAttrs =
                new HashMap<>(finalLlmSpan.getAttributes().asMap());

        assertThat(finalLlmAttrs.remove(AttributeKey.stringKey(SemanticConventions.OPENINFERENCE_SPAN_KIND)))
                .isEqualTo(SemanticConventions.OpenInferenceSpanKind.LLM.toString());
        assertThat(Objects.isNull(finalLlmAttrs.remove(AttributeKey.stringKey(SemanticConventions.INPUT_VALUE))))
                .isTrue();
        assertThat(Objects.isNull(finalLlmAttrs.remove(AttributeKey.stringKey(SemanticConventions.INPUT_MIME_TYPE))))
                .isTrue();
        assertThat(finalLlmAttrs.remove(AttributeKey.stringKey(SemanticConventions.OUTPUT_MIME_TYPE)))
                .isEqualTo("text/plain");
        assertThat(finalLlmAttrs.remove(AttributeKey.stringKey(SemanticConventions.OUTPUT_VALUE)))
                .isEqualTo("The sum of 45 plus 79 is 124.");
        // Model attributes
        assertThat(finalLlmAttrs.remove(AttributeKey.stringKey(SemanticConventions.LLM_MODEL_NAME)))
                .isEqualTo("unknown");
        assertThat(finalLlmAttrs.remove(AttributeKey.stringKey(SemanticConventions.LLM_SYSTEM)))
                .isEqualTo("langchain4j");
        assertThat(finalLlmAttrs.remove(AttributeKey.stringKey(SemanticConventions.LLM_PROVIDER)))
                .isEqualTo("openai");

        // Token counts
        assertThat(finalLlmAttrs.remove(AttributeKey.longKey(SemanticConventions.LLM_TOKEN_COUNT_PROMPT)))
                .isEqualTo(189L);
        assertThat(finalLlmAttrs.remove(AttributeKey.longKey(SemanticConventions.LLM_TOKEN_COUNT_COMPLETION)))
                .isEqualTo(13L);
        assertThat(finalLlmAttrs.remove(AttributeKey.longKey(SemanticConventions.LLM_TOKEN_COUNT_TOTAL)))
                .isEqualTo(202L);

        // Finish reason
        List<String> finalFinishReasons =
                (List<String>) finalLlmAttrs.remove(AttributeKey.stringArrayKey("llm.response.finish_reasons"));
        assertThat(finalFinishReasons).containsExactly("STOP");

        // Output messages
        assertThat(finalLlmAttrs.remove(AttributeKey.stringKey("llm.output_messages.0.message.role")))
                .isEqualTo("assistant");
        assertThat(finalLlmAttrs.remove(AttributeKey.stringKey("llm.output_messages.0.message.content")))
                .isEqualTo("The sum of 45 plus 79 is 124.");

        // Tool schemas
        actualTools = List.of(
                getToolName((String) finalLlmAttrs.remove(AttributeKey.stringKey(getToolSchemaAttributeKey(0)))),
                getToolName((String) finalLlmAttrs.remove(AttributeKey.stringKey(getToolSchemaAttributeKey(1)))),
                getToolName((String) finalLlmAttrs.remove(AttributeKey.stringKey(getToolSchemaAttributeKey(2)))),
                getToolName((String) finalLlmAttrs.remove(AttributeKey.stringKey(getToolSchemaAttributeKey(3)))));
        assertThat(actualTools).containsExactlyInAnyOrderElementsOf(expectedTools);
        assertThat((String) finalLlmAttrs.remove(AttributeKey.stringKey(SemanticConventions.LLM_INVOCATION_PARAMETERS)))
                .contains("{}");
        assertThat(finalLlmAttrs).isEmpty();

        // Verify Agent Span Attributes
        SpanData agentSpan = spans.get(3);
        assertThat(agentSpan.getName()).isEqualTo("AiService.chat");
        assertThat(agentSpan.getKind()).isEqualTo(SpanKind.CLIENT);
        assertThat(agentSpan.getStatus().getStatusCode()).isEqualTo(StatusCode.OK);

        Map<AttributeKey<?>, Object> agentAttrs =
                new HashMap<>(agentSpan.getAttributes().asMap());

        assertThat(agentAttrs.remove(AttributeKey.stringKey(SemanticConventions.OPENINFERENCE_SPAN_KIND)))
                .isEqualTo(SemanticConventions.OpenInferenceSpanKind.AGENT.toString());
        assertThat(Objects.isNull(agentAttrs.remove(AttributeKey.stringKey(SemanticConventions.INPUT_VALUE))))
                .isTrue();
        assertThat(Objects.isNull(agentAttrs.remove(AttributeKey.stringKey(SemanticConventions.INPUT_MIME_TYPE))))
                .isTrue();
        assertThat(agentAttrs.remove(AttributeKey.stringKey(SemanticConventions.OUTPUT_MIME_TYPE)))
                .isEqualTo("text/plain");
        assertThat(agentAttrs.remove(AttributeKey.stringKey(SemanticConventions.OUTPUT_VALUE)))
                .isEqualTo("The sum of 45 plus 79 is 124.");
        assertThat(agentAttrs).isEmpty();
    }

    @Test
    void aiServiceErrorHandling(TestInfo testInfo) throws Exception {
        stubForOpenAiChatErrorCalls();
        LangChain4jInstrumentor instrumentor =
                LangChain4jInstrumentor.instrument(tracerProvider, TraceConfig.getDefault());
        try {
            MathAssistant assistant = createMathAssistant(testInfo, instrumentor);
            assistant.chat("What is 100 divided by 0?");
        } catch (Exception e) {
            assertThat(e.getMessage()).containsIgnoringCase("Incorrect API key provided");
        } finally {
            instrumentor.uninstrument();
        }

        List<SpanData> spans = spanExporter.getFinishedSpanItems();
        assertThat(spans).isNotEmpty();
        boolean errorSpanFound = false;
        for (SpanData span : spans) {
            if (span.getStatus().getStatusCode() == StatusCode.ERROR) {
                errorSpanFound = true;
                assertThat(span.getStatus().getStatusCode()).isEqualTo(StatusCode.ERROR);
                assertThat(span.getStatus().getDescription()).contains("Incorrect API key provided");
                break;
            }
        }
        assertThat(errorSpanFound)
                .as("Expected to find an error span with StatusCode.ERROR")
                .isTrue();
    }

    interface MathAssistant {
        String chat(String userMessage);
    }

    static class MathTools {

        @Tool("Adds two numbers together and returns the sum")
        public double add(double a, double b) {
            return a + b;
        }

        @Tool("Subtracts the second number from the first number")
        public double subtract(double a, double b) {
            return a - b;
        }

        @Tool("Multiplies two numbers together and returns the product")
        public double multiply(double a, double b) {
            return a * b;
        }

        @Tool("Divides the first number by the second number")
        public double divide(double a, double b) {
            if (b == 0) throw new IllegalArgumentException("Cannot divide by zero");
            return a / b;
        }
    }
}

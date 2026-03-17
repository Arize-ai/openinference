package com.arize.instrumentation.langchain4j.utils;

import com.arize.instrumentation.OITracer;
import com.arize.instrumentation.langchain4j.LangChain4jModelListener;
import com.arize.semconv.trace.SemanticConventions;
import com.fasterxml.jackson.annotation.JsonAutoDetect;
import com.fasterxml.jackson.annotation.PropertyAccessor;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.message.*;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import io.opentelemetry.api.common.AttributeKey;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.StatusCode;

import java.util.*;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;

/**
 * Utility class for extracting and setting chat message attributes on OpenTelemetry spans.
 * <p>
 * Provides static methods for handling tool specifications, input/output messages, and tool call extraction for tracing.
 * </p>
 */
public class ChatMessageAttributeUtils {

    private static final Logger logger = Logger.getLogger(LangChain4jModelListener.class.getName());

    private static final ObjectMapper objectMapper = new ObjectMapper();

    private static final ObjectMapper toolParamMapper = new ObjectMapper().setVisibility(
            PropertyAccessor.FIELD,
            JsonAutoDetect.Visibility.ANY
    );

    /**
     * Sets LLM tool attributes on the span for the provided tool specifications.
     *
     * @param span              the span to set attributes on
     * @param toolSpecifications the list of tool specifications
     */
    public static void setLlmToolAttributes(Span span, List<ToolSpecification> toolSpecifications) {
        for (int idx = 0; idx < toolSpecifications.size(); idx++) {
            ToolSpecification toolSpec = toolSpecifications.get(idx);
            try {
                Map<String, Object> functionMap = new LinkedHashMap<>();
                functionMap.put(SemanticConventions.ToolAttributePostfixes.NAME, toolSpec.name());
                if (toolSpec.description() != null) {
                    functionMap.put(SemanticConventions.ToolAttributePostfixes.DESCRIPTION, toolSpec.description());
                }
                functionMap.put(
                        SemanticConventions.ToolAttributePostfixes.PARAMETERS,
                        toolSpec.parameters() != null
                                ? toolParamMapper.convertValue(toolSpec.parameters(), Map.class)
                                : new LinkedHashMap<>()
                );
                Map<String, Object> toolSchemaMap = new LinkedHashMap<>();
                toolSchemaMap.put("type", "function");
                toolSchemaMap.put("function", functionMap);

                span.setAttribute(
                        AttributeKey.stringKey(SemanticConventions.LLM_TOOLS + "." + idx + "."
                                + SemanticConventions.TOOL_JSON_SCHEMA),
                        objectMapper.writeValueAsString(toolSchemaMap));
            } catch (JsonProcessingException e) {
                logger.log(Level.WARNING, "Failed to serialize tool specification at index " + idx, e);
            }
        }
    }

    /**
     * Sets input message attributes on the span for the provided chat messages.
     *
     * @param span     the span to set attributes on
     * @param messages the list of chat messages
     */
    public static void setInputMessageAttributes(Span span, List<ChatMessage> messages) {
        for (int i = 0; i < messages.size(); i++) {
            ChatMessage message = messages.get(i);
            String prefix = String.format("%s.%d.", SemanticConventions.LLM_INPUT_MESSAGES, i);

            // Set role
            String role = mapMessageRole(message.type());
            span.setAttribute(AttributeKey.stringKey(prefix + SemanticConventions.MESSAGE_ROLE), role);

            // Set content
            span.setAttribute(AttributeKey.stringKey(prefix + SemanticConventions.MESSAGE_CONTENT), message.toString());

            // Set ToolCall
            if (message.type().equals(ChatMessageType.AI)) {
                AiMessage aiMessage = (AiMessage) message;
                if (aiMessage.toolExecutionRequests() != null
                        && !aiMessage.toolExecutionRequests().isEmpty()) toolCallExtraction(span, prefix, aiMessage);
            }

            // Set Tool Response
            if (message.type().equals(ChatMessageType.TOOL_EXECUTION_RESULT)) {
                ToolExecutionResultMessage toolExecutionResultMessage = (ToolExecutionResultMessage) message;

                span.setAttribute(
                        AttributeKey.stringKey(prefix + SemanticConventions.MESSAGE_TOOL_CALL_ID),
                        toolExecutionResultMessage.id());
            }
        }
    }

    /**
     * Extracts tool call information from an AI message and sets attributes on the span.
     *
     * @param span      the span to set attributes on
     * @param prefix    the attribute prefix
     * @param aiMessage the AI message containing tool execution requests
     */
    public static void toolCallExtraction(Span span, String prefix, AiMessage aiMessage) {
        for (int i = 0; i < aiMessage.toolExecutionRequests().size(); i++) {
            // Add tool call attributes here if needed
            span.setAttribute(
                    AttributeKey.stringKey(prefix + SemanticConventions.MESSAGE_TOOL_CALLS + "." + i + "."
                            + SemanticConventions.TOOL_CALL_ID),
                    aiMessage.toolExecutionRequests().get(i).id());
            span.setAttribute(
                    AttributeKey.stringKey(prefix + SemanticConventions.MESSAGE_TOOL_CALLS + "." + i + "."
                            + SemanticConventions.TOOL_CALL_FUNCTION_ARGUMENTS_JSON),
                    aiMessage.toolExecutionRequests().get(i).arguments());
            span.setAttribute(
                    AttributeKey.stringKey(prefix + SemanticConventions.MESSAGE_TOOL_CALLS + "." + i + "."
                            + SemanticConventions.TOOL_CALL_FUNCTION_NAME),
                    aiMessage.toolExecutionRequests().get(i).name());
        }
    }

    public static void setOutputMessageAttributes(Span span, AiMessage aiMessage) {
        String prefix = String.format("%s.%d.", SemanticConventions.LLM_OUTPUT_MESSAGES, 0);

        // Set role
        span.setAttribute(AttributeKey.stringKey(prefix + SemanticConventions.MESSAGE_ROLE), "assistant");

        // Set content
        span.setAttribute(AttributeKey.stringKey(prefix + SemanticConventions.MESSAGE_CONTENT), aiMessage.text());

        // Set tool calls if present

        if (aiMessage.toolExecutionRequests() != null
                && !aiMessage.toolExecutionRequests().isEmpty()) {
            toolCallExtraction(span, prefix, aiMessage);
        }
    }

    public static String mapMessageRole(ChatMessageType type) {
        switch (type) {
            case SYSTEM:
                return "system";
            case USER:
                return "user";
            case AI:
                return "assistant";
            case TOOL_EXECUTION_RESULT:
                return "tool";
            default:
                return type.toString().toLowerCase();
        }
    }

    public static List<Map<String, Object>> convertMessages(List<ChatMessage> messages) {
        List<Map<String, Object>> result = new ArrayList<>();

        for (ChatMessage message : messages) {
            Map<String, Object> messageMap = new HashMap<>();

            switch (message.type()) {
                case SYSTEM -> {
                    messageMap.put("role", "system");
                    if (message instanceof SystemMessage systemMessage) {
                        messageMap.put("content", systemMessage.text());
                    }
                }
                case USER -> {
                    messageMap.put("role", "user");
                    if (message instanceof UserMessage userMessage) {
                        messageMap.put("content", userMessage.singleText());
                    }
                }
                case AI -> {
                    messageMap.put("role", "assistant");
                    if (message instanceof AiMessage aiMessage) {
                        messageMap.put("content", aiMessage.text());
                        if (aiMessage.toolExecutionRequests() != null
                                && !aiMessage.toolExecutionRequests().isEmpty()) {
                            messageMap.put(
                                    "content",
                                    Map.of(
                                            "tool_calls",
                                            aiMessage.toolExecutionRequests().stream()
                                                    .map(t -> Map.of(
                                                            "id",
                                                            t.id(),
                                                            "function",
                                                            Map.of("arguments", t.arguments(), "name", t.name())))
                                                    .collect(Collectors.toList())));
                        }
                    }
                }
                case TOOL_EXECUTION_RESULT -> {
                    messageMap.put("role", "tool");
                    if (message.type().equals(ChatMessageType.TOOL_EXECUTION_RESULT)) {
                        ToolExecutionResultMessage toolExecutionResultMessage = (ToolExecutionResultMessage) message;
                        messageMap.put("content", toolExecutionResultMessage.text());
                        messageMap.put(SemanticConventions.MESSAGE_TOOL_CALL_ID, toolExecutionResultMessage.id());
                    }
                }
            }

            result.add(messageMap);
        }

        return result;
    }

    public static void handleChatRequest(OITracer tracer, Span span, ChatRequest request) {
        String modelName = request.modelName() != null ? request.modelName() : "unknown";

        // Set basic attributes
        span.setAttribute(
                SemanticConventions.OPENINFERENCE_SPAN_KIND,
                SemanticConventions.OpenInferenceSpanKind.LLM.getValue()
        );
        span.setAttribute(SemanticConventions.LLM_MODEL_NAME, modelName);
        span.setAttribute(SemanticConventions.LLM_SYSTEM, "langchain4j");

        // OpenAI Client only supported for now
        span.setAttribute(SemanticConventions.LLM_PROVIDER, SemanticConventions.LLMProvider.OPENAI.getValue());

        // Set invocation parameters as a single JSON object
        Map<String, Object> invocationParams = new HashMap<>();
        if (request.temperature() != null) {
            invocationParams.put("temperature", request.temperature());
        }
        if (request.maxOutputTokens() != null) {
            invocationParams.put("max_tokens", request.maxOutputTokens());
        }
        if (request.topP() != null) {
            invocationParams.put("top_p", request.topP());
        }

        try {
            span.setAttribute(
                    SemanticConventions.LLM_INVOCATION_PARAMETERS, objectMapper.writeValueAsString(invocationParams));
        } catch (JsonProcessingException e) {
            logger.log(Level.WARNING, "Failed to serialize invocation parameters", e);
        }

        // Set tool attributes
        List<ToolSpecification> toolSpecifications = request.toolSpecifications();
        if (Objects.nonNull(toolSpecifications) && !toolSpecifications.isEmpty()) {
            setLlmToolAttributes(span, toolSpecifications);
        }

        // Set input attributes
        if (!tracer.getConfig().isHideInputMessages()) {
            try {
                // Set input messages with proper structure
                setInputMessageAttributes(span, request.messages());

                // Also set input.value and input.mime_type for compatibility
                List<Map<String, Object>> messagesList = convertMessages(request.messages());
                String messagesJson = objectMapper.writeValueAsString(messagesList);
                span.setAttribute(SemanticConventions.INPUT_VALUE, messagesJson);
                span.setAttribute(SemanticConventions.INPUT_MIME_TYPE, "application/json");
            } catch (JsonProcessingException e) {
                logger.log(Level.WARNING, "Failed to serialize input messages", e);
            }
        }
    }

    public static void handleChatResponse(OITracer tracer, Span span, ChatResponse response) {
        // Set response attributes
        if (response.finishReason() != null) {
            // Set finish reasons as an array attribute
            span.setAttribute(
                    AttributeKey.stringArrayKey("llm.response.finish_reasons"),
                    List.of(response.finishReason().name()));
        }

        // Set output message with proper structure
        if (!tracer.getConfig().isHideOutputMessages() && response.aiMessage() != null) {
            AiMessage aiMessage = response.aiMessage();

            // Set output messages with proper structure
            setOutputMessageAttributes(span, aiMessage);

            span.setAttribute(SemanticConventions.OUTPUT_VALUE, aiMessage.text());
            span.setAttribute(SemanticConventions.OUTPUT_MIME_TYPE, "text/plain");
        }

        // Set token usage if available
        if (response.tokenUsage() != null) {
            span.setAttribute(SemanticConventions.LLM_TOKEN_COUNT_PROMPT, (long)
                    response.tokenUsage().inputTokenCount());
            span.setAttribute(SemanticConventions.LLM_TOKEN_COUNT_COMPLETION, (long)
                    response.tokenUsage().outputTokenCount());
            span.setAttribute(SemanticConventions.LLM_TOKEN_COUNT_TOTAL, (long)
                    response.tokenUsage().totalTokenCount());
        }
        span.setStatus(StatusCode.OK);
    }
}

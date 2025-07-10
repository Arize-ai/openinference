package io.openinference.instrumentation.langchain4j;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.ChatMessageType;
import dev.langchain4j.model.chat.listener.*;
import io.openinference.common.ModelProvider;
import io.openinference.instrumentation.OITracer;
import io.openinference.semconv.trace.MessageAttributes;
import io.openinference.semconv.trace.SpanAttributes;
import io.opentelemetry.api.common.AttributeKey;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.StatusCode;
import io.opentelemetry.context.Context;
import io.opentelemetry.context.Scope;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Listener for LangChain4j chat models that creates OpenInference spans.
 */
public class LangChain4jModelListener implements ChatModelListener {

    private static final Logger logger = Logger.getLogger(LangChain4jModelListener.class.getName());
    private static final ObjectMapper objectMapper = new ObjectMapper();

    private final OITracer tracer;
    private final Map<ChatModelRequest, SpanContext> activeSpans = new ConcurrentHashMap<>();

    public LangChain4jModelListener(OITracer tracer) {
        this.tracer = tracer;
    }

    @Override
    public void onRequest(ChatModelRequestContext requestContext) {
        ChatModelRequest request = requestContext.request();

        // Create span for the LLM call
        String modelName = request.model() != null ? request.model() : "unknown";
        Span span = tracer.llmSpanBuilder("chat", modelName)
                .setParent(Context.current())
                .startSpan();

        // Set basic attributes
        span.setAttribute(SpanAttributes.LLM_MODEL_NAME, modelName);
        span.setAttribute(SpanAttributes.LLM_SYSTEM, "langchain4j");

        // Detect and set provider based on model name
        String provider = ModelProvider.detectProvider(modelName);
        if (provider != null) {
            span.setAttribute(SpanAttributes.LLM_PROVIDER, provider);
        }

        // Set invocation parameters as a single JSON object
        Map<String, Object> invocationParams = new HashMap<>();
        if (request.temperature() != null) {
            invocationParams.put("temperature", request.temperature());
        }
        if (request.maxTokens() != null) {
            invocationParams.put("max_tokens", request.maxTokens());
        }
        if (request.topP() != null) {
            invocationParams.put("top_p", request.topP());
        }

        try {
            span.setAttribute(
                    SpanAttributes.LLM_INVOCATION_PARAMETERS, objectMapper.writeValueAsString(invocationParams));
        } catch (JsonProcessingException e) {
            logger.log(Level.WARNING, "Failed to serialize invocation parameters", e);
        }

        // Set input attributes
        if (!tracer.getConfig().isHideInputMessages()) {
            try {
                // Set input messages with proper structure
                setInputMessageAttributes(span, request.messages());

                // Also set input.value and input.mime_type for compatibility
                List<Map<String, Object>> messagesList = convertMessages(request.messages());
                String messagesJson = objectMapper.writeValueAsString(messagesList);
                span.setAttribute(SpanAttributes.INPUT_VALUE, messagesJson);
                span.setAttribute(SpanAttributes.INPUT_MIME_TYPE, "application/json");
            } catch (JsonProcessingException e) {
                logger.log(Level.WARNING, "Failed to serialize input messages", e);
            }
        }

        // Store span context for later use
        activeSpans.put(request, new SpanContext(span, Context.current().with(span)));
    }

    @Override
    public void onResponse(ChatModelResponseContext responseContext) {
        SpanContext spanContext = activeSpans.remove(responseContext.request());
        if (spanContext == null) {
            logger.warning("No active span found for response");
            return;
        }

        Span span = spanContext.span;
        ChatModelResponse response = responseContext.response();

        try (Scope scope = spanContext.context.makeCurrent()) {
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

                span.setAttribute(SpanAttributes.OUTPUT_VALUE, aiMessage.text());
                span.setAttribute(SpanAttributes.OUTPUT_MIME_TYPE, "text/plain");
            }

            // Set token usage if available
            if (response.tokenUsage() != null) {
                span.setAttribute(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, (long)
                        response.tokenUsage().inputTokenCount());
                span.setAttribute(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, (long)
                        response.tokenUsage().outputTokenCount());
                span.setAttribute(SpanAttributes.LLM_TOKEN_COUNT_TOTAL, (long)
                        response.tokenUsage().totalTokenCount());
            }

            span.setStatus(StatusCode.OK);
        } finally {
            span.end();
        }
    }

    @Override
    public void onError(ChatModelErrorContext errorContext) {
        SpanContext spanContext = activeSpans.remove(errorContext.request());
        if (spanContext == null) {
            logger.warning("No active span found for error");
            return;
        }

        Span span = spanContext.span;

        try (Scope scope = spanContext.context.makeCurrent()) {
            Throwable error = errorContext.error();
            span.recordException(error);
            span.setStatus(StatusCode.ERROR, error.getMessage());
        } finally {
            span.end();
        }
    }

    private void setInputMessageAttributes(Span span, List<ChatMessage> messages) {
        for (int i = 0; i < messages.size(); i++) {
            ChatMessage message = messages.get(i);
            String prefix = String.format("%s.%d.", SpanAttributes.LLM_INPUT_MESSAGES.getKey(), i);

            // Set role
            String role = mapMessageRole(message.type());
            span.setAttribute(AttributeKey.stringKey(prefix + MessageAttributes.MESSAGE_ROLE.getKey()), role);

            // Set content
            span.setAttribute(
                    AttributeKey.stringKey(prefix + MessageAttributes.MESSAGE_CONTENT.getKey()), message.text());
        }
    }

    private void setOutputMessageAttributes(Span span, AiMessage aiMessage) {
        String prefix = String.format("%s.%d.", SpanAttributes.LLM_OUTPUT_MESSAGES.getKey(), 0);

        // Set role
        span.setAttribute(AttributeKey.stringKey(prefix + MessageAttributes.MESSAGE_ROLE.getKey()), "assistant");

        // Set content
        span.setAttribute(
                AttributeKey.stringKey(prefix + MessageAttributes.MESSAGE_CONTENT.getKey()), aiMessage.text());

        // Set tool calls if present
        if (aiMessage.toolExecutionRequests() != null
                && !aiMessage.toolExecutionRequests().isEmpty()) {
            for (int i = 0; i < aiMessage.toolExecutionRequests().size(); i++) {
                // Add tool call attributes here if needed
                span.setAttribute(
                        AttributeKey.stringKey(prefix + MessageAttributes.MESSAGE_TOOL_CALL_ID.getKey()),
                        aiMessage.toolExecutionRequests().get(i).id());
                span.setAttribute(
                        AttributeKey.stringKey(prefix + MessageAttributes.MESSAGE_TOOL_CALLS.getKey()),
                        aiMessage.toolExecutionRequests().get(i).arguments());
            }
        }
    }

    private String mapMessageRole(ChatMessageType type) {
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

    private List<Map<String, Object>> convertMessages(List<ChatMessage> messages) {
        List<Map<String, Object>> result = new ArrayList<>();

        for (ChatMessage message : messages) {
            Map<String, Object> messageMap = new HashMap<>();

            switch (message.type()) {
                case SYSTEM:
                    messageMap.put("role", "system");
                    messageMap.put("content", message.text());
                    break;
                case USER:
                    messageMap.put("role", "user");
                    messageMap.put("content", message.text());
                    break;
                case AI:
                    messageMap.put("role", "assistant");
                    messageMap.put("content", message.text());
                    if (message instanceof AiMessage) {
                        AiMessage aiMessage = (AiMessage) message;
                        if (aiMessage.toolExecutionRequests() != null
                                && !aiMessage.toolExecutionRequests().isEmpty()) {
                            messageMap.put("tool_calls", aiMessage.toolExecutionRequests());
                        }
                    }
                    break;
                case TOOL_EXECUTION_RESULT:
                    messageMap.put("role", "tool");
                    messageMap.put("content", message.text());
                    break;
            }

            result.add(messageMap);
        }

        return result;
    }

    private static class SpanContext {
        final Span span;
        final Context context;

        SpanContext(Span span, Context context) {
            this.span = span;
            this.context = context;
        }
    }
}

package com.arize.instrumentation.springAI;

import com.arize.instrumentation.OITracer;
import com.arize.semconv.trace.SemanticConventions;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.micrometer.observation.Observation;
import io.micrometer.observation.ObservationHandler;
import io.opentelemetry.api.common.AttributeKey;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.SpanKind;
import io.opentelemetry.api.trace.StatusCode;
import io.opentelemetry.context.Context;
import io.opentelemetry.context.Scope;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.ToolResponseMessage;
import org.springframework.ai.chat.model.Generation;
import org.springframework.ai.chat.observation.ChatModelObservationContext;
import org.springframework.ai.chat.prompt.ChatOptions;

public class SpringAIInstrumentor implements ObservationHandler<Observation.Context> {
    private static final Logger log = LoggerFactory.getLogger(SpringAIInstrumentor.class);
    private static final ObjectMapper objectMapper = new ObjectMapper();

    private final OITracer tracer;
    private final Map<Observation.Context, SpanContext> activeSpans = new ConcurrentHashMap<>();

    public SpringAIInstrumentor(OITracer tracer) {
        this.tracer = tracer;
    }

    @Override
    public void onStart(Observation.Context context) {
        if (!(context instanceof ChatModelObservationContext)) {
            return;
        }

        ChatModelObservationContext chatContext = (ChatModelObservationContext) context;

        // Create span for the LLM call
        String modelName = extractModelName(chatContext);
        Span span = tracer.spanBuilder("generate")
                .setParent(Context.current())
                .setSpanKind(SpanKind.CLIENT)
                .startSpan();

        // Set basic attributes
        span.setAttribute(
                SemanticConventions.OPENINFERENCE_SPAN_KIND, SemanticConventions.OpenInferenceSpanKind.LLM.getValue());
        span.setAttribute(SemanticConventions.LLM_MODEL_NAME, modelName);
        span.setAttribute(SemanticConventions.LLM_SYSTEM, "spring-ai");
        span.setAttribute(SemanticConventions.LLM_PROVIDER, SemanticConventions.LLMProvider.OPENAI.getValue());

        // Set invocation parameters
        setInvocationParameters(span, chatContext);

        // Set input attributes
        if (!tracer.getConfig().isHideInputMessages()) {
            setInputMessageAttributes(span, chatContext);
        }

        // Store span context for later use
        activeSpans.put(context, new SpanContext(span, Context.current().with(span)));

        log.debug("Started LLM span for model: {}", modelName);
    }

    @Override
    public void onStop(Observation.Context context) {
        SpanContext spanContext = activeSpans.remove(context);
        if (spanContext == null) {
            log.warn("No active span found for completed observation");
            return;
        }

        if (!(context instanceof ChatModelObservationContext)) {
            spanContext.span.end();
            return;
        }

        ChatModelObservationContext chatContext = (ChatModelObservationContext) context;
        Span span = spanContext.span;

        try (Scope scope = spanContext.context.makeCurrent()) {
            // Set output attributes
            if (!tracer.getConfig().isHideOutputMessages()) {
                setOutputMessageAttributes(span, chatContext);
            }

            // Set token usage if available
            setTokenUsage(span, chatContext);

            span.setStatus(StatusCode.OK);
            log.debug("Completed LLM span successfully");
        } finally {
            span.end();
        }
    }

    @Override
    public void onEvent(Observation.Event event, Observation.Context context) {
        log.info("event occured");
    }

    @Override
    public void onError(Observation.Context context) {
        SpanContext spanContext = activeSpans.remove(context);
        if (spanContext == null) {
            log.warn("No active span found for error");
            return;
        }

        Span span = spanContext.span;

        try (Scope scope = spanContext.context.makeCurrent()) {
            Throwable error = context.getError();
            if (error != null) {
                span.recordException(error);
                span.setStatus(StatusCode.ERROR, error.getMessage());
            } else {
                span.setStatus(StatusCode.ERROR, "Unknown error occurred");
            }
            log.debug("Recorded error in LLM span");
        } finally {
            span.end();
        }
    }

    @Override
    public boolean supportsContext(Observation.Context context) {
        return context instanceof ChatModelObservationContext;
    }

    private String extractModelName(ChatModelObservationContext context) {
        try {
            ChatOptions options = context.getRequest().getOptions();
            if (options != null && options.getModel() != null) {
                return options.getModel();
            }
        } catch (Exception e) {
            log.debug("Could not extract model name from chat options", e);
        }
        return "unknown";
    }

    private void setInvocationParameters(Span span, ChatModelObservationContext context) {
        try {
            Map<String, Object> invocationParams = new HashMap<>();
            ChatOptions options = context.getRequest().getOptions();

            if (options != null) {
                if (options.getTemperature() != null) {
                    invocationParams.put("temperature", options.getTemperature());
                }
                if (options.getMaxTokens() != null) {
                    invocationParams.put("max_tokens", options.getMaxTokens());
                }
                if (options.getTopP() != null) {
                    invocationParams.put("top_p", options.getTopP());
                }
            }

            if (!invocationParams.isEmpty()) {
                span.setAttribute(
                        SemanticConventions.LLM_INVOCATION_PARAMETERS,
                        objectMapper.writeValueAsString(invocationParams));
            }
        } catch (Exception e) {
            log.warn("Failed to set invocation parameters", e);
        }
    }

    protected void setInputMessageAttributes(Span span, ChatModelObservationContext context) {
        try {
            List<Message> messages = context.getRequest().getInstructions();
            if (messages == null || messages.isEmpty()) {
                return;
            }

            // Set individual message attributes
            for (int i = 0; i < messages.size(); i++) {
                Message message = messages.get(i);
                String prefix = String.format("%s.%d.", SemanticConventions.LLM_INPUT_MESSAGES, i);

                span.setAttribute(
                        AttributeKey.stringKey(prefix + SemanticConventions.MESSAGE_ROLE), mapMessageRole(message));
                span.setAttribute(
                        AttributeKey.stringKey(prefix + SemanticConventions.MESSAGE_CONTENT), message.getText());

                // Handle tool calls for assistant messages
                if (message instanceof AssistantMessage) {
                    AssistantMessage assistantMessage = (AssistantMessage) message;
                    if (assistantMessage.getToolCalls() != null
                            && !assistantMessage.getToolCalls().isEmpty()) {
                        setToolCallAttributes(span, prefix, assistantMessage.getToolCalls());
                    }
                }

                // Handle tool responses
                if (message instanceof ToolResponseMessage) {
                    ToolResponseMessage toolResponseMessage = (ToolResponseMessage) message;
                    span.setAttribute(
                            AttributeKey.stringKey(prefix + SemanticConventions.MESSAGE_CONTENT),
                            toolResponseMessage.getResponses().get(0).responseData());
                    span.setAttribute(
                            AttributeKey.stringKey(prefix + SemanticConventions.MESSAGE_TOOL_CALL_ID),
                            toolResponseMessage.getResponses().get(0).id());
                }
            }

            // Set input.value for compatibility
            List<Map<String, Object>> messagesList = convertMessages(messages);
            String messagesJson = objectMapper.writeValueAsString(messagesList);
            span.setAttribute(SemanticConventions.INPUT_VALUE, messagesJson);
            span.setAttribute(SemanticConventions.INPUT_MIME_TYPE, "application/json");
        } catch (Exception e) {
            log.warn("Failed to set input message attributes", e);
        }
    }

    protected void setOutputMessageAttributes(Span span, ChatModelObservationContext context) {
        try {
            if (context.getResponse() == null || context.getResponse().getResults() == null) {
                return;
            }

            // Get the results and combine the output
            List<Generation> results = context.getResponse().getResults();
            List<Message> outs = new ArrayList<>();
            for (int i=0; i < results.size(); i++) {
                Generation generation = results.get(i);
                var output = generation.getOutput();

                if (output instanceof AssistantMessage) {
                    AssistantMessage assistantMessage = (AssistantMessage) output;
                    String prefix = String.format("%s.%d.", SemanticConventions.LLM_OUTPUT_MESSAGES, i);

                    span.setAttribute(AttributeKey.stringKey(prefix + SemanticConventions.MESSAGE_ROLE), "assistant");
                    span.setAttribute(
                            AttributeKey.stringKey(prefix + SemanticConventions.MESSAGE_CONTENT),
                            assistantMessage.getText());

                    // Handle tool calls in output
                    if (assistantMessage.getToolCalls() != null
                            && !assistantMessage.getToolCalls().isEmpty()) {
                        setToolCallAttributes(span, prefix, assistantMessage.getToolCalls());
                    }

                    outs.add(assistantMessage);
                }

                if (!outs.isEmpty()) {
                    List<Map<String, Object>> messagesList = convertMessages(outs);
                    String messagesJson = objectMapper.writeValueAsString(messagesList);

                    span.setAttribute(SemanticConventions.OUTPUT_VALUE, messagesJson);
                    span.setAttribute(SemanticConventions.OUTPUT_MIME_TYPE, "application/json");
                }
            }
        } catch (Exception e) {
            log.warn("Failed to set output message attributes", e);
        }
    }

    private void setTokenUsage(Span span, ChatModelObservationContext context) {
        try {
            if (context.getResponse() == null || context.getResponse().getMetadata() == null) {
                return;
            }

            var metadata = context.getResponse().getMetadata();
            var usage = metadata.getUsage();

            if (usage != null) {
                if (usage.getPromptTokens() != null) {
                    span.setAttribute(
                            SemanticConventions.LLM_TOKEN_COUNT_PROMPT,
                            usage.getPromptTokens().longValue());
                }
                if (usage.getCompletionTokens() != null) {
                    span.setAttribute(
                            SemanticConventions.LLM_TOKEN_COUNT_COMPLETION,
                            usage.getCompletionTokens().longValue());
                }
                if (usage.getTotalTokens() != null) {
                    span.setAttribute(
                            SemanticConventions.LLM_TOKEN_COUNT_TOTAL,
                            usage.getTotalTokens().longValue());
                }
            }
        } catch (Exception e) {
            log.debug("Could not extract token usage", e);
        }
    }

    private String mapMessageRole(Message message) {
        String messageType = message.getMessageType().getValue().toLowerCase();
        switch (messageType) {
            case "system":
                return "system";
            case "user":
                return "user";
            case "assistant":
                return "assistant";
            case "tool":
                return "tool";
            default:
                return messageType;
        }
    }

    private List<Map<String, Object>> convertMessages(List<Message> messages) {
        List<Map<String, Object>> result = new ArrayList<>();

        for (Message message : messages) {
            Map<String, Object> messageMap = new HashMap<>();
            messageMap.put("role", mapMessageRole(message));
            messageMap.put("content", message.getText());

            // Handle tool calls in assistant messages
            if (message instanceof AssistantMessage) {
                AssistantMessage assistantMessage = (AssistantMessage) message;
                if (assistantMessage.getToolCalls() != null
                        && !assistantMessage.getToolCalls().isEmpty()) {
                    List<Map<String, Object>> toolCalls = new ArrayList<>();
                    for (AssistantMessage.ToolCall toolCall : assistantMessage.getToolCalls()) {
                        Map<String, Object> toolCallMap = new HashMap<>();
                        toolCallMap.put("id", toolCall.id());
                        toolCallMap.put("type", toolCall.type());

                        Map<String, Object> functionMap = new HashMap<>();
                        functionMap.put("name", toolCall.name());
                        functionMap.put("arguments", toolCall.arguments());
                        toolCallMap.put("function", functionMap);

                        toolCalls.add(toolCallMap);
                    }
                    messageMap.put("tool_calls", toolCalls);
                }
            }

            // Handle tool response messages
            if (message instanceof ToolResponseMessage) {
                ToolResponseMessage toolResponseMessage = (ToolResponseMessage) message;
                messageMap.put(
                        "tool_call_id",
                        toolResponseMessage.getResponses().get(0).id());
            }

            result.add(messageMap);
        }

        return result;
    }

    private void setToolCallAttributes(Span span, String prefix, List<AssistantMessage.ToolCall> toolCalls) {
        for (int i = 0; i < toolCalls.size(); i++) {
            AssistantMessage.ToolCall toolCall = toolCalls.get(i);
            String toolCallPrefix = prefix + SemanticConventions.MESSAGE_TOOL_CALLS + "." + i + ".";

            span.setAttribute(AttributeKey.stringKey(toolCallPrefix + SemanticConventions.TOOL_CALL_ID), toolCall.id());
            span.setAttribute(
                    AttributeKey.stringKey(toolCallPrefix + SemanticConventions.TOOL_CALL_FUNCTION_NAME),
                    toolCall.name());
            span.setAttribute(
                    AttributeKey.stringKey(toolCallPrefix + SemanticConventions.TOOL_CALL_FUNCTION_ARGUMENTS_JSON),
                    toolCall.arguments());
        }
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

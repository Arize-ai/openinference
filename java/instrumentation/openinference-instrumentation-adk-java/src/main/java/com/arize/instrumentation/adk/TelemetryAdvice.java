package com.arize.instrumentation.adk;

import static com.arize.semconv.trace.SemanticConventions.*;
import static net.bytebuddy.implementation.bytecode.assign.Assigner.Typing.DYNAMIC;

import com.arize.semconv.trace.SemanticConventions;
import com.google.adk.events.Event;
import com.google.adk.models.LlmRequest;
import com.google.adk.models.LlmResponse;
import com.google.adk.tools.BaseTool;
import com.google.genai.types.*;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.StatusCode;
import io.opentelemetry.sdk.trace.ReadWriteSpan;
import io.reactivex.rxjava3.core.Flowable;
import io.reactivex.rxjava3.functions.Consumer;
import java.lang.reflect.Method;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import net.bytebuddy.asm.Advice;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import tools.jackson.databind.json.JsonMapper;

/**
 * Advice class for intercepting the traceToolCall method in Telemetry.
 */
public class TelemetryAdvice {

    public static final Logger log = LoggerFactory.getLogger(TelemetryAdvice.class);

    public static final JsonMapper JSON_MAPPER = new JsonMapper();

    @Advice.OnMethodExit()
    public static void onExit(
            @Advice.Origin Method method,
            @Advice.AllArguments Object[] args,
            @Advice.Return(readOnly = false, typing = DYNAMIC) Object returnValue) {

        log.info("Enhancing method {} onExit.", method.getName());

        switch (method.getName()) {
            case "traceToolCall":
                handleTraceToolCall(args);
                break;
            case "traceCallLlm":
                handleTraceCallLlm(args);
                break;
            case "traceFlowable":
                // This case modifies the return value, so we re-assign it.
                returnValue = handleTraceFlowable(returnValue, args);
                break;
            default:
                break;
        }
    }

    public static void handleTraceToolCall(Object[] args) {
        if (args.length > 0 && args[0] instanceof Map<?, ?> argsMap) {
            Span span = Span.current();
            if (span != null) {
                span.setStatus(StatusCode.OK);
                span.setAttribute(OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKind.TOOL.getValue());
                if (!argsMap.isEmpty()) {
                    String argsJson = JSON_MAPPER.writeValueAsString(argsMap);
                    span.setAttribute(TOOL_PARAMETERS, argsJson);
                    span.setAttribute(INPUT_VALUE, argsJson);
                    span.setAttribute(INPUT_MIME_TYPE, MimeType.JSON.getValue());
                }
            }
        }
    }

    public static void handleTraceCallLlm(Object[] args) {
        if (args.length > 3
                && args[1] instanceof String eventId
                && args[2] instanceof LlmRequest llmRequest
                && args[3] instanceof LlmResponse llmResponse) {
            Span span = Span.current();
            if (span == null) return;
            span.setStatus(StatusCode.OK);
            span.setAttribute(OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKind.LLM.getValue());
            span.setAttribute(LLM_PROVIDER, LLMProvider.GOOGLE.getValue());
            span.setAttribute(LLM_MODEL_NAME, llmRequest.model().orElse(""));
            span.setAttribute(INPUT_VALUE, llmRequest.toJson());
            span.setAttribute(INPUT_MIME_TYPE, MimeType.JSON.getValue());

            // handle tools and set tool attributes
            int i = 0;
            for (String toolKey : llmRequest.tools().keySet()) {
                BaseTool baseTool = llmRequest.tools().get(toolKey);
                Optional<FunctionDeclaration> declaration = baseTool.declaration();
                String toolJsonSchema;
                if (declaration.isPresent()) {
                    toolJsonSchema = declaration.get().toJson();
                } else {
                    toolJsonSchema = JSON_MAPPER.writeValueAsString(
                            Map.of("name", baseTool.name(), "description", baseTool.description()));
                }
                span.setAttribute(LLM_TOOLS + "." + i + "." + TOOL_JSON_SCHEMA, toolJsonSchema);
                i++;
            }

            // Set input message attributes
            int inputMessageIndex = 0;
            String prefix = String.format("%s.%d.", SemanticConventions.LLM_INPUT_MESSAGES, inputMessageIndex);
            Optional<GenerateContentConfig> optionalGenerateContentConfig = llmRequest.config();

            // handle system instruction
            if (optionalGenerateContentConfig.isPresent()) {
                GenerateContentConfig generateContentConfig = optionalGenerateContentConfig.get();
                span.setAttribute(LLM_INVOCATION_PARAMETERS, generateContentConfig.toJson());
                Optional<Content> optionalSystemInstruction = generateContentConfig.systemInstruction();
                if (optionalSystemInstruction.isPresent()) {
                    span.setAttribute(prefix + MESSAGE_ROLE, "system");
                    inputMessageIndex++;
                    Content systemInstruction = optionalSystemInstruction.get();
                    Optional<List<Part>> parts = systemInstruction.parts();
                    if (parts.isPresent()) {
                        List<Part> partList = parts.get();
                        if (!partList.isEmpty()) {
                            for (int j = 0; j < partList.size(); j++) {
                                String systemText = partList.get(j).text().orElse("");
                                span.setAttribute(
                                        prefix + MESSAGE_CONTENTS + "." + j + "." + MESSAGE_CONTENT_TEXT, systemText);
                                span.setAttribute(
                                        prefix + MESSAGE_CONTENTS + "." + j + "." + MESSAGE_CONTENT_TYPE, "text");
                            }
                        }
                    }
                }
            }

            // handle contents
            List<Content> contents = llmRequest.contents();
            for (int k = 0; k < contents.size(); k++) {
                String contentPrefix = LLM_INPUT_MESSAGES + "." + (inputMessageIndex + k) + ".";
                span.setAttribute(
                        contentPrefix + MESSAGE_ROLE, contents.get(k).role().orElse("user"));
                Optional<List<Part>> parts = contents.get(k).parts();
                if (parts.isEmpty()) continue;
                List<Part> partList = parts.get();
                if (partList.isEmpty()) continue;
                handlePartList(span, contentPrefix, partList);
            }

            // Set ouput message attributes
            span.setAttribute(OUTPUT_VALUE, llmResponse.toJson());
            span.setAttribute(OUTPUT_MIME_TYPE, MimeType.JSON.getValue());
            Optional<Content> responseContent = llmResponse.content();
            if (responseContent.isPresent()) {
                Content content = responseContent.get();
                String contentPrefix = LLM_OUTPUT_MESSAGES + ".0.";
                span.setAttribute(contentPrefix + MESSAGE_ROLE, content.role().orElse("model"));
                Optional<List<Part>> parts = content.parts();
                if (parts.isPresent() && !parts.get().isEmpty()) {
                    List<Part> partList = parts.get();
                    handlePartList(span, contentPrefix, partList);
                }
            }

            Optional<GenerateContentResponseUsageMetadata> generateContentResponseUsageMetadata =
                    llmResponse.usageMetadata();
            if (generateContentResponseUsageMetadata.isPresent()) {
                GenerateContentResponseUsageMetadata usageMetadata = generateContentResponseUsageMetadata.get();
                span.setAttribute(
                        LLM_TOKEN_COUNT_TOTAL, usageMetadata.totalTokenCount().orElse(0));
                span.setAttribute(
                        LLM_TOKEN_COUNT_PROMPT, usageMetadata.promptTokenCount().orElse(0));
                int completionTokenCount = 0;
                if (usageMetadata.candidatesTokenCount().isPresent()) {
                    completionTokenCount += usageMetadata.candidatesTokenCount().get();
                }
                int thoughtTokenCount = usageMetadata.thoughtsTokenCount().orElse(0);
                span.setAttribute(LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING, thoughtTokenCount);
                completionTokenCount += thoughtTokenCount;
                span.setAttribute(LLM_TOKEN_COUNT_COMPLETION, completionTokenCount);
            }
        }
    }

    public static void handlePartList(Span span, String contentPrefix, List<Part> partList) {
        for (int l = 0; l < partList.size(); l++) {
            Part part = partList.get(l);
            // handle text
            Optional<String> text = part.text();
            if (text.isPresent()) {
                String contentText = text.get();
                span.setAttribute(contentPrefix + MESSAGE_CONTENTS + "." + l + "." + MESSAGE_CONTENT_TEXT, contentText);
                span.setAttribute(contentPrefix + MESSAGE_CONTENTS + "." + l + "." + MESSAGE_CONTENT_TYPE, "text");
            }
            // handle function call
            Optional<FunctionCall> functionCall = part.functionCall();
            if (functionCall.isPresent()) {
                FunctionCall functionCall1 = functionCall.get();
                span.setAttribute(
                        contentPrefix + MESSAGE_TOOL_CALLS + "." + l + "." + TOOL_CALL_ID,
                        functionCall1.id().orElse(""));
                span.setAttribute(
                        contentPrefix + MESSAGE_TOOL_CALLS + "." + l + "." + TOOL_CALL_FUNCTION_NAME,
                        functionCall1.name().orElse(""));
                if (functionCall1.args().isPresent()) {
                    span.setAttribute(
                            contentPrefix + MESSAGE_TOOL_CALLS + "." + l + "." + TOOL_CALL_FUNCTION_ARGUMENTS_JSON,
                            JSON_MAPPER.writeValueAsString(functionCall1.args().get()));
                }
            }
            // handle function response
            Optional<FunctionResponse> functionResponse = part.functionResponse();
            if (functionResponse.isPresent()) {
                span.setAttribute(contentPrefix + MESSAGE_ROLE, "tool");
                FunctionResponse functionResponse1 = functionResponse.get();
                if (functionResponse1.name().isPresent()) {
                    span.setAttribute(
                            contentPrefix + MESSAGE_NAME,
                            functionResponse1.name().get());
                }
                if (functionResponse1.response().isPresent()) {
                    span.setAttribute(
                            contentPrefix + MESSAGE_CONTENT,
                            JSON_MAPPER.writeValueAsString(
                                    functionResponse1.response().get()));
                }
            }
        }
    }

    @SuppressWarnings("unchecked")
    public static Object handleTraceFlowable(Object returnValue, Object[] args) {
        if (returnValue instanceof Flowable && args.length > 1 && args[1] instanceof Span span) {
            if (span instanceof ReadWriteSpan readWriteSpan
                    && readWriteSpan.getName().startsWith("agent_run")) {
                span.setAttribute(OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKind.AGENT.getValue());
            }
            Flowable<Event> originalFlowable = (Flowable<Event>) returnValue;
            // Modify and return the new Flowable
            return originalFlowable.doOnNext(new FinalEventConsumer(span));
        }
        // Return the original returnValue if conditions are not met
        return returnValue;
    }

    public record FinalEventConsumer(Span span) implements Consumer<Event> {

        @Override
        public void accept(Event event) {
            if (event.finalResponse()) {
                span.setStatus(StatusCode.OK);
                span.setAttribute(OUTPUT_VALUE, event.toJson());
                span.setAttribute(OUTPUT_MIME_TYPE, MimeType.JSON.getValue());
            }
        }
    }
}

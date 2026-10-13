package com.arize.instrumentation.adk;

import static com.arize.semconv.trace.SemanticConventions.INPUT_MIME_TYPE;
import static com.arize.semconv.trace.SemanticConventions.INPUT_VALUE;
import static com.arize.semconv.trace.SemanticConventions.LLM_INPUT_MESSAGES;
import static com.arize.semconv.trace.SemanticConventions.LLM_INVOCATION_PARAMETERS;
import static com.arize.semconv.trace.SemanticConventions.LLM_MODEL_NAME;
import static com.arize.semconv.trace.SemanticConventions.LLM_OUTPUT_MESSAGES;
import static com.arize.semconv.trace.SemanticConventions.LLM_PROVIDER;
import static com.arize.semconv.trace.SemanticConventions.LLM_TOKEN_COUNT_COMPLETION;
import static com.arize.semconv.trace.SemanticConventions.LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING;
import static com.arize.semconv.trace.SemanticConventions.LLM_TOKEN_COUNT_PROMPT;
import static com.arize.semconv.trace.SemanticConventions.LLM_TOKEN_COUNT_TOTAL;
import static com.arize.semconv.trace.SemanticConventions.LLM_TOOLS;
import static com.arize.semconv.trace.SemanticConventions.MESSAGE_CONTENT;
import static com.arize.semconv.trace.SemanticConventions.MESSAGE_CONTENTS;
import static com.arize.semconv.trace.SemanticConventions.MESSAGE_CONTENT_TEXT;
import static com.arize.semconv.trace.SemanticConventions.MESSAGE_CONTENT_TYPE;
import static com.arize.semconv.trace.SemanticConventions.MESSAGE_NAME;
import static com.arize.semconv.trace.SemanticConventions.MESSAGE_ROLE;
import static com.arize.semconv.trace.SemanticConventions.MESSAGE_TOOL_CALLS;
import static com.arize.semconv.trace.SemanticConventions.MESSAGE_TOOL_CALL_ID;
import static com.arize.semconv.trace.SemanticConventions.OPENINFERENCE_SPAN_KIND;
import static com.arize.semconv.trace.SemanticConventions.OUTPUT_MIME_TYPE;
import static com.arize.semconv.trace.SemanticConventions.OUTPUT_VALUE;
import static com.arize.semconv.trace.SemanticConventions.TOOL_CALL_FUNCTION_ARGUMENTS_JSON;
import static com.arize.semconv.trace.SemanticConventions.TOOL_CALL_FUNCTION_NAME;
import static com.arize.semconv.trace.SemanticConventions.TOOL_CALL_ID;
import static com.arize.semconv.trace.SemanticConventions.TOOL_DESCRIPTION;
import static com.arize.semconv.trace.SemanticConventions.TOOL_JSON_SCHEMA;
import static com.arize.semconv.trace.SemanticConventions.TOOL_NAME;
import static com.arize.semconv.trace.SemanticConventions.TOOL_PARAMETERS;

import com.arize.instrumentation.OITracer;
import com.arize.instrumentation.OpenInferenceAgent;
import com.arize.instrumentation.SuppressTracing;
import com.arize.instrumentation.TraceConfig;
import com.arize.semconv.trace.SemanticConventions.LLMProvider;
import com.arize.semconv.trace.SemanticConventions.MimeType;
import com.arize.semconv.trace.SemanticConventions.OpenInferenceSpanKind;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.adk.events.Event;
import com.google.adk.models.LlmRequest;
import com.google.adk.models.LlmResponse;
import com.google.adk.tools.BaseTool;
import com.google.genai.types.Content;
import com.google.genai.types.FunctionCall;
import com.google.genai.types.FunctionDeclaration;
import com.google.genai.types.FunctionResponse;
import com.google.genai.types.GenerateContentConfig;
import com.google.genai.types.GenerateContentResponseUsageMetadata;
import com.google.genai.types.Part;
import io.opentelemetry.api.trace.Span;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Shared helpers for the ADK advice classes: activation checks (valid, recording, not suppressed),
 * TraceConfig-aware attribute writes, JSON serialization that never throws, and the mapping from
 * ADK/genai request and response types to OpenInference LLM attributes.
 */
public final class AdkSpanSupport {

    private static final Logger log = LoggerFactory.getLogger(AdkSpanSupport.class);

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private AdkSpanSupport() {}

    /** Whether the advice should decorate {@code span} at all. */
    public static boolean isActive(Span span) {
        if (span == null || !span.getSpanContext().isValid() || !span.isRecording()) {
            return false;
        }
        if (SuppressTracing.isSuppressed()) {
            return false;
        }
        return !config().isSuppressTracing();
    }

    /** The application's TraceConfig if it registered an OITracer, otherwise the defaults. */
    public static TraceConfig config() {
        OITracer tracer = OpenInferenceAgent.getTracer();
        return tracer != null ? tracer.getConfig() : TraceConfig.getDefault();
    }

    /** Serializes with Jackson; returns {@code null} instead of throwing so advice never breaks the app. */
    public static String toJson(Object value) {
        if (value == null) {
            return null;
        }
        try {
            return MAPPER.writeValueAsString(value);
        } catch (JsonProcessingException e) {
            log.debug(
                    "Failed to serialize {} for span attribute",
                    value.getClass().getName(),
                    e);
            return null;
        }
    }

    public static void set(Span span, String key, String value) {
        if (value == null || shouldHide(key)) {
            return;
        }
        span.setAttribute(key, value);
    }

    public static void set(Span span, String key, long value) {
        if (shouldHide(key)) {
            return;
        }
        span.setAttribute(key, value);
    }

    /** Writes a JSON value and its mime type together (both or neither). */
    public static void setJson(Span span, String valueKey, String mimeTypeKey, String json) {
        if (json == null) {
            return;
        }
        set(span, valueKey, json);
        set(span, mimeTypeKey, MimeType.JSON.getValue());
    }

    /** Mirrors the subset of {@code TracedSpan.shouldHide} that applies to the attributes written here. */
    public static boolean shouldHide(String key) {
        TraceConfig config = config();
        if (config.isHideInputs()) {
            if (INPUT_VALUE.equals(key) || INPUT_MIME_TYPE.equals(key)) return true;
            if (key.startsWith(LLM_INPUT_MESSAGES)) return true;
        }
        if (config.isHideOutputs()) {
            if (OUTPUT_VALUE.equals(key) || OUTPUT_MIME_TYPE.equals(key)) return true;
            if (key.startsWith(LLM_OUTPUT_MESSAGES)) return true;
        }
        if (config.isHideInputMessages() && key.startsWith(LLM_INPUT_MESSAGES)) return true;
        if (config.isHideOutputMessages() && key.startsWith(LLM_OUTPUT_MESSAGES)) return true;
        if (config.isHideInputText()
                && key.startsWith(LLM_INPUT_MESSAGES)
                && key.contains(MESSAGE_CONTENT)
                && !key.contains(MESSAGE_CONTENTS)) return true;
        if (config.isHideOutputText()
                && key.startsWith(LLM_OUTPUT_MESSAGES)
                && key.contains(MESSAGE_CONTENT)
                && !key.contains(MESSAGE_CONTENTS)) return true;
        if (config.isHideToolParameters() && TOOL_PARAMETERS.equals(key)) return true;
        return false;
    }

    /** Attributes for a tool invocation; ADK reports the arguments before the tool runs. */
    public static void writeToolCall(Span span, Map<?, ?> args) {
        span.setAttribute(OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKind.TOOL.getValue());
        if (args == null || args.isEmpty()) {
            return;
        }
        String argsJson = toJson(args);
        set(span, TOOL_PARAMETERS, argsJson);
        setJson(span, INPUT_VALUE, INPUT_MIME_TYPE, argsJson);
    }

    /** Tool name and description, written when ADK runs the tool inside the {@code tool_call} span. */
    public static void writeToolInfo(Span span, BaseTool tool) {
        if (tool == null) {
            return;
        }
        set(span, TOOL_NAME, tool.name());
        set(span, TOOL_DESCRIPTION, tool.description());
    }

    /** The tool's result map as the {@code tool_call} span output. */
    public static void writeToolResult(Span span, Object result) {
        if (result == null) {
            return;
        }
        setJson(span, OUTPUT_VALUE, OUTPUT_MIME_TYPE, toJson(result));
    }

    /**
     * Attributes for ADK's separate {@code tool_response} span: a CHAIN step whose output is the
     * function response body when the event carries exactly one, otherwise the whole event.
     */
    public static void writeToolResponse(Span span, Event event) {
        span.setAttribute(OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKind.CHAIN.getValue());
        List<FunctionResponse> responses = new ArrayList<>();
        Optional<Content> content = event.content();
        if (content.isPresent()) {
            for (Part part : content.get().parts().orElse(List.of())) {
                part.functionResponse().ifPresent(responses::add);
            }
        }
        if (responses.size() == 1 && responses.get(0).response().isPresent()) {
            setJson(
                    span,
                    OUTPUT_VALUE,
                    OUTPUT_MIME_TYPE,
                    toJson(responses.get(0).response().get()));
        } else {
            setJson(span, OUTPUT_VALUE, OUTPUT_MIME_TYPE, event.toJson());
        }
    }

    /**
     * Attributes for one LLM call. ADK invokes {@code traceCallLlm} once per streamed chunk, so
     * partial chunks are ignored; the aggregated (non-partial) response carries the full reply.
     */
    public static void writeLlmCall(Span span, LlmRequest request, LlmResponse response) {
        if (response.partial().orElse(false)) {
            return;
        }
        span.setAttribute(OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKind.LLM.getValue());
        span.setAttribute(LLM_PROVIDER, LLMProvider.GOOGLE.getValue());
        request.model().ifPresent(model -> span.setAttribute(LLM_MODEL_NAME, model));
        setJson(span, INPUT_VALUE, INPUT_MIME_TYPE, request.toJson());

        int toolIndex = 0;
        for (BaseTool tool : request.tools().values()) {
            Optional<FunctionDeclaration> declaration = tool.declaration();
            String schema = declaration.isPresent()
                    ? declaration.get().toJson()
                    : toJson(Map.of("name", tool.name(), "description", String.valueOf(tool.description())));
            set(span, LLM_TOOLS + "." + toolIndex + "." + TOOL_JSON_SCHEMA, schema);
            toolIndex++;
        }

        int messageIndex = 0;
        Optional<GenerateContentConfig> config = request.config();
        if (config.isPresent()) {
            set(span, LLM_INVOCATION_PARAMETERS, config.get().toJson());
            Optional<Content> systemInstruction = config.get().systemInstruction();
            if (systemInstruction.isPresent()) {
                messageIndex += writeContent(span, LLM_INPUT_MESSAGES, messageIndex, systemInstruction.get(), "system");
            }
        }
        for (Content content : request.contents()) {
            messageIndex += writeContent(
                    span,
                    LLM_INPUT_MESSAGES,
                    messageIndex,
                    content,
                    content.role().orElse("user"));
        }

        Optional<Content> output = response.content();
        if (output.isPresent() && hasContent(output.get())) {
            setJson(span, OUTPUT_VALUE, OUTPUT_MIME_TYPE, response.toJson());
            writeContent(
                    span,
                    LLM_OUTPUT_MESSAGES,
                    0,
                    output.get(),
                    output.get().role().orElse("model"));
        }
        response.usageMetadata().ifPresent(usage -> writeUsage(span, usage));
    }

    /**
     * Writes one genai {@link Content} as OpenInference messages starting at {@code startIndex}
     * and returns how many messages were written. Text and function-call parts form a single
     * message with contiguous {@code message.contents.N} and {@code message.tool_calls.N} indices;
     * each function-response part becomes its own {@code tool} message, so parallel tool results
     * are never overwritten.
     */
    public static int writeContent(Span span, String listKey, int startIndex, Content content, String role) {
        List<Part> regularParts = new ArrayList<>();
        List<FunctionResponse> functionResponses = new ArrayList<>();
        for (Part part : content.parts().orElse(List.of())) {
            Optional<FunctionResponse> functionResponse = part.functionResponse();
            if (functionResponse.isPresent()) {
                functionResponses.add(functionResponse.get());
            } else {
                regularParts.add(part);
            }
        }

        int messageIndex = startIndex;
        if (!regularParts.isEmpty() || functionResponses.isEmpty()) {
            String prefix = listKey + "." + messageIndex + ".";
            set(span, prefix + MESSAGE_ROLE, role);
            int contentIndex = 0;
            int toolCallIndex = 0;
            for (Part part : regularParts) {
                Optional<String> text = part.text();
                if (text.isPresent() && !text.get().isEmpty()) {
                    String contentPrefix = prefix + MESSAGE_CONTENTS + "." + contentIndex + ".";
                    set(span, contentPrefix + MESSAGE_CONTENT_TYPE, "text");
                    set(span, contentPrefix + MESSAGE_CONTENT_TEXT, text.get());
                    contentIndex++;
                }
                Optional<FunctionCall> functionCall = part.functionCall();
                if (functionCall.isPresent()) {
                    FunctionCall call = functionCall.get();
                    String toolCallPrefix = prefix + MESSAGE_TOOL_CALLS + "." + toolCallIndex + ".";
                    call.id().ifPresent(id -> set(span, toolCallPrefix + TOOL_CALL_ID, id));
                    call.name().ifPresent(name -> set(span, toolCallPrefix + TOOL_CALL_FUNCTION_NAME, name));
                    call.args()
                            .ifPresent(args ->
                                    set(span, toolCallPrefix + TOOL_CALL_FUNCTION_ARGUMENTS_JSON, toJson(args)));
                    toolCallIndex++;
                }
            }
            messageIndex++;
        }
        for (FunctionResponse functionResponse : functionResponses) {
            String prefix = listKey + "." + messageIndex + ".";
            set(span, prefix + MESSAGE_ROLE, "tool");
            functionResponse.id().ifPresent(id -> set(span, prefix + MESSAGE_TOOL_CALL_ID, id));
            functionResponse.name().ifPresent(name -> set(span, prefix + MESSAGE_NAME, name));
            functionResponse.response().ifPresent(body -> set(span, prefix + MESSAGE_CONTENT, toJson(body)));
            messageIndex++;
        }
        return messageIndex - startIndex;
    }

    /** Whether {@code content} carries anything worth reporting as a message. */
    public static boolean hasContent(Content content) {
        for (Part part : content.parts().orElse(List.of())) {
            if (part.functionCall().isPresent() || part.functionResponse().isPresent()) {
                return true;
            }
            if (part.text().filter(text -> !text.isEmpty()).isPresent()) {
                return true;
            }
        }
        return false;
    }

    /**
     * Token counts, only when reported. Thinking tokens are added to the completion count unless
     * Gemini has already folded them into {@code candidatesTokenCount} (detected the same way as
     * the Python google-adk instrumentor: prompt + candidates == total).
     */
    public static void writeUsage(Span span, GenerateContentResponseUsageMetadata usage) {
        Optional<Integer> prompt = usage.promptTokenCount();
        Optional<Integer> toolUsePrompt = usage.toolUsePromptTokenCount();
        Optional<Integer> candidates = usage.candidatesTokenCount();
        Optional<Integer> thoughts = usage.thoughtsTokenCount();
        Optional<Integer> total = usage.totalTokenCount();

        int promptTotal = prompt.orElse(0) + toolUsePrompt.orElse(0);
        if (prompt.isPresent() || toolUsePrompt.isPresent()) {
            set(span, LLM_TOKEN_COUNT_PROMPT, promptTotal);
        }
        total.ifPresent(value -> set(span, LLM_TOKEN_COUNT_TOTAL, value));

        int completion = candidates.orElse(0);
        if (thoughts.isPresent()) {
            set(span, LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING, thoughts.get());
            boolean thoughtsAlreadyCounted = (prompt.isPresent() || toolUsePrompt.isPresent())
                    && total.isPresent()
                    && promptTotal + candidates.orElse(0) == total.get();
            if (!thoughtsAlreadyCounted) {
                completion += thoughts.get();
            }
        }
        if (candidates.isPresent() || thoughts.isPresent()) {
            set(span, LLM_TOKEN_COUNT_COMPLETION, completion);
        }
    }
}

package com.arize.instrumentation.langchain4j;

import com.arize.instrumentation.OITracer;
import com.arize.instrumentation.langchain4j.utils.SpanContext;
import com.arize.semconv.trace.SemanticConventions;
import com.fasterxml.jackson.databind.ObjectMapper;
import dev.langchain4j.observability.api.event.AiServiceStartedEvent;
import dev.langchain4j.observability.api.listener.AiServiceStartedListener;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.SpanKind;
import io.opentelemetry.context.Context;

import java.util.HashMap;
import java.util.Map;

public class LangChain4jAiServiceStartedListener implements AiServiceStartedListener {

    private final OITracer tracer;
    private final Map<String, SpanContext> activeSpans;

    public LangChain4jAiServiceStartedListener(OITracer tracer, Map<String, SpanContext> activeSpans) {
        this.tracer = tracer;
        this.activeSpans = activeSpans;
    }

    public void setInputAttributes(Span span, AiServiceStartedEvent event) {
        span.setAttribute(
                SemanticConventions.OPENINFERENCE_SPAN_KIND,
                SemanticConventions.OpenInferenceSpanKind.AGENT.getValue()
        );
        Map<String, String> inputMap = new HashMap<>();
        inputMap.put("userMessage", event.userMessage().toString());
        inputMap.put("invocationContext", event.invocationContext().toString());
        if (event.systemMessage().isEmpty()) {
            inputMap.put("systemMessage", event.systemMessage().get().toString());
        }
        ObjectMapper objectMapper = new ObjectMapper();
        try {
            span.setAttribute(SemanticConventions.INPUT_VALUE, objectMapper.writeValueAsString(inputMap));
        } catch (Exception e) {
            span.setAttribute(SemanticConventions.INPUT_VALUE, event.userMessage().toString());
        }

        span.setAttribute(
                SemanticConventions.INPUT_MIME_TYPE,
                SemanticConventions.MimeType.JSON.getValue()
        );
        Map<String, String> metadata = new HashMap<>();
        metadata.put("invocationId", event.invocationContext().invocationId().toString());
        span.setAttribute(
                SemanticConventions.METADATA, metadata.toString()
        );
    }

    @Override
    public void onEvent(AiServiceStartedEvent event) {
        Span span = tracer.spanBuilder("AiService.chat")
                .setParent(Context.current())
                .setSpanKind(SpanKind.CLIENT)
                .startSpan();

        // Set basic attributes
        setInputAttributes(span, event);
        activeSpans.put(
                event.invocationContext().invocationId().toString(),
                new SpanContext(span, Context.current().with(span))
        );
    }
}

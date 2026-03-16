package com.arize.instrumentation.langchain4j;

import com.arize.instrumentation.OITracer;
import com.arize.instrumentation.langchain4j.utils.SpanContext;
import com.arize.semconv.trace.SemanticConventions;
import dev.langchain4j.observability.api.event.ToolExecutedEvent;
import dev.langchain4j.observability.api.listener.ToolExecutedEventListener;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.SpanKind;
import io.opentelemetry.api.trace.StatusCode;
import io.opentelemetry.context.Context;

import java.util.Map;
import java.util.UUID;

public class LangChain4jToolExecutedEventListener implements ToolExecutedEventListener {

    private final OITracer tracer;
    private final Map<String, SpanContext> activeSpans;

    public LangChain4jToolExecutedEventListener(OITracer tracer, Map<String, SpanContext> activeSpans) {
        this.tracer = tracer;
        this.activeSpans = activeSpans;
    }

    @Override
    public void onEvent(ToolExecutedEvent toolExecutedEvent) {
        String invocationId = toolExecutedEvent.invocationContext().invocationId().toString();
        SpanContext parentSpanContext = activeSpans.get(invocationId);
        Context context = parentSpanContext != null
                ? parentSpanContext.context() // Use AiService span's context
                : Context.current();

        Span span = tracer.spanBuilder("AiService.tool")
                .setParent(context)
                .setSpanKind(SpanKind.CLIENT)
                .startSpan();

        // Set basic attributes
        span.setAttribute(
                SemanticConventions.OPENINFERENCE_SPAN_KIND,
                SemanticConventions.OpenInferenceSpanKind.TOOL.getValue()
        );
        span.setAttribute(SemanticConventions.INPUT_VALUE, toolExecutedEvent.request().toString());
        span.setAttribute(SemanticConventions.OUTPUT_VALUE, toolExecutedEvent.resultText());
        span.setStatus(StatusCode.OK);
        span.end();
    }
}

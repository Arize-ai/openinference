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

public class LangChain4jToolExecutedEventListener implements ToolExecutedEventListener {

    private final OITracer tracer;
    private final Map<String, SpanContext> activeSpans;

    public LangChain4jToolExecutedEventListener(OITracer tracer, Map<String, SpanContext> activeSpans) {
        this.tracer = tracer;
        this.activeSpans = activeSpans;
    }

    @Override
    public void onEvent(ToolExecutedEvent event) {
        String invocationId = event.invocationContext().invocationId().toString();
        SpanContext parentSpanContext = activeSpans.get(invocationId);
        Context context = parentSpanContext != null
                ? parentSpanContext.context() // Use AiService span's context
                : Context.current();

        Span span = tracer.spanBuilder("AiService.tool")
                .setParent(context)
                .setSpanKind(SpanKind.CLIENT)
                .startSpan();

        span.setAttribute(
                SemanticConventions.OPENINFERENCE_SPAN_KIND,
                SemanticConventions.OpenInferenceSpanKind.TOOL.getValue()
        );

        span.setAttribute(SemanticConventions.TOOL_CALL_ID, event.request().id());
        span.setAttribute(SemanticConventions.TOOL_NAME, event.request().name());
        span.setAttribute(SemanticConventions.TOOL_PARAMETERS, event.request().arguments());

        span.setAttribute(SemanticConventions.OUTPUT_VALUE, event.resultText());
        span.setStatus(StatusCode.OK);
        span.end();
    }
}

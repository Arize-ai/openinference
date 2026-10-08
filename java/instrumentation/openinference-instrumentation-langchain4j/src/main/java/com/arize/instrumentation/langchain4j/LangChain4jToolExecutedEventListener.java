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

/**
 * Event listener for handling tool execution events in LangChain4j.
 * <p>
 * This listener creates and ends a tracing span for each tool execution event,
 * capturing relevant attributes such as tool name, parameters, and output.
 * </p>
 */
public class LangChain4jToolExecutedEventListener implements ToolExecutedEventListener {

    /**
     * Tracer used for creating spans.
     */
    private final OITracer tracer;
    /**
     * Map of active spans keyed by invocation ID.
     */
    private final Map<String, SpanContext> activeSpans;

    /**
     * Constructs a new listener for tool execution events.
     *
     * @param tracer      the tracer to use for span creation
     * @param activeSpans the map of active spans
     */
    public LangChain4jToolExecutedEventListener(OITracer tracer, Map<String, SpanContext> activeSpans) {
        this.tracer = tracer;
        this.activeSpans = activeSpans;
    }

    /**
     * Handles the tool execution event by creating a span and recording tool attributes.
     *
     * @param event the tool executed event
     */
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
                SemanticConventions.OPENINFERENCE_SPAN_KIND, SemanticConventions.OpenInferenceSpanKind.TOOL.getValue());
        if (!tracer.getConfig().isHideInputs()) {
            span.setAttribute(SemanticConventions.INPUT_VALUE, event.request().arguments());
            span.setAttribute(SemanticConventions.INPUT_MIME_TYPE, SemanticConventions.MimeType.JSON.getValue());
        }
        if (!tracer.getConfig().isHideInputs() && !tracer.getConfig().isHideToolParameters()) {
            span.setAttribute(
                    SemanticConventions.TOOL_PARAMETERS, event.request().arguments());
        }
        if (!tracer.getConfig().isHideOutputs()) {
            span.setAttribute(SemanticConventions.OUTPUT_VALUE, event.resultText());
            span.setAttribute(SemanticConventions.OUTPUT_MIME_TYPE, SemanticConventions.MimeType.TEXT.getValue());
        }
        span.setAttribute(SemanticConventions.TOOL_CALL_ID, event.request().id());
        span.setAttribute(SemanticConventions.TOOL_NAME, event.request().name());
        span.setStatus(StatusCode.OK);
        span.end();
    }
}

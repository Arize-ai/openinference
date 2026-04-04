package com.arize.instrumentation.langchain4j;

import com.arize.instrumentation.OITracer;
import com.arize.instrumentation.langchain4j.utils.SpanContext;
import com.arize.semconv.trace.SemanticConventions;
import dev.langchain4j.observability.api.event.AiServiceCompletedEvent;
import dev.langchain4j.observability.api.listener.AiServiceCompletedListener;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.StatusCode;
import io.opentelemetry.context.Scope;
import java.util.Map;

public class LangChain4jServiceCompletedListener implements AiServiceCompletedListener {

    private final OITracer tracer;
    private final Map<String, SpanContext> activeSpans;

    public LangChain4jServiceCompletedListener(OITracer tracer, Map<String, SpanContext> activeSpans) {
        this.tracer = tracer;
        this.activeSpans = activeSpans;
    }

    @Override
    public void onEvent(AiServiceCompletedEvent event) {
        SpanContext spanContext =
                activeSpans.remove(event.invocationContext().invocationId().toString());
        if (spanContext != null) {
            Span span = spanContext.span();
            try (Scope scope = spanContext.context().makeCurrent()) {
                if (!tracer.getConfig().isHideOutputs() && event.result().isPresent()) {
                    span.setAttribute(
                            SemanticConventions.OUTPUT_VALUE,
                            event.result().get().toString());
                    span.setAttribute(
                            SemanticConventions.OUTPUT_MIME_TYPE, SemanticConventions.MimeType.TEXT.getValue());
                }
                span.setStatus(StatusCode.OK);
            } finally {
                span.end();
            }
        }
    }
}

package com.arize.instrumentation.langchain4j;

import com.arize.instrumentation.OITracer;
import com.arize.instrumentation.langchain4j.utils.SpanContext;
import com.arize.semconv.trace.SemanticConventions;
import dev.langchain4j.observability.api.event.AiServiceCompletedEvent;
import dev.langchain4j.observability.api.event.AiServiceResponseReceivedEvent;
import dev.langchain4j.observability.api.listener.AiServiceCompletedListener;
import dev.langchain4j.observability.api.listener.AiServiceResponseReceivedListener;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.SpanKind;
import io.opentelemetry.api.trace.StatusCode;
import io.opentelemetry.context.Context;

import java.util.Map;
import java.util.UUID;

public class LangChain4jServiceResponseReceivedListener implements AiServiceResponseReceivedListener {

    private final OITracer tracer;
    private final Map<UUID, SpanContext> activeSpans;

    public LangChain4jServiceResponseReceivedListener(OITracer tracer, Map<UUID, SpanContext> activeSpans) {
        this.tracer = tracer;
        this.activeSpans = activeSpans;
    }

    @Override
    public void onEvent(AiServiceResponseReceivedEvent event) {
        UUID invocationId = event.invocationContext().invocationId();
        SpanContext parentSpanContext = activeSpans.get(invocationId);
        Context context = parentSpanContext != null
                ? parentSpanContext.context() // Use AiService span's context
                : Context.current();

        Span span = tracer.spanBuilder("AiService.chat")
                .setParent(context)
                .setSpanKind(SpanKind.CLIENT)
                .startSpan();

        // Set basic attributes
        span.setAttribute(
                SemanticConventions.OPENINFERENCE_SPAN_KIND,
                SemanticConventions.OpenInferenceSpanKind.LLM.getValue()
        );
        span.setAttribute(SemanticConventions.INPUT_VALUE, "LLM REQUEST");
        span.setAttribute(SemanticConventions.OUTPUT_VALUE, event.response().toString());
        span.setStatus(StatusCode.OK);
        span.end();
    }
}

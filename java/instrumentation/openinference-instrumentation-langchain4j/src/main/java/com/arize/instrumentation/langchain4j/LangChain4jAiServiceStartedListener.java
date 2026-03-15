package com.arize.instrumentation.langchain4j;

import com.arize.instrumentation.OITracer;
import com.arize.instrumentation.langchain4j.utils.SpanContext;
import com.arize.semconv.trace.SemanticConventions;
import dev.langchain4j.observability.api.event.AiServiceStartedEvent;
import dev.langchain4j.observability.api.listener.AiServiceStartedListener;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.SpanKind;
import io.opentelemetry.context.Context;

import java.util.Map;
import java.util.UUID;

public class LangChain4jAiServiceStartedListener implements AiServiceStartedListener {

    private final OITracer tracer;
    private final Map<UUID, SpanContext> activeSpans;

    public LangChain4jAiServiceStartedListener(OITracer tracer, Map<UUID, SpanContext> activeSpans) {
        this.tracer = tracer;
        this.activeSpans = activeSpans;
    }

    @Override
    public void onEvent(AiServiceStartedEvent aiServiceStartedEvent) {

        Span span = tracer.spanBuilder("AiService.start")
                .setParent(Context.current())
                .setSpanKind(SpanKind.CLIENT)
                .startSpan();

        // Set basic attributes
        span.setAttribute(
                SemanticConventions.OPENINFERENCE_SPAN_KIND, SemanticConventions.OpenInferenceSpanKind.AGENT.getValue());
        span.setAttribute(SemanticConventions.LLM_SYSTEM, "langchain4j");
        span.setAttribute(SemanticConventions.INPUT_VALUE, "Execution Started");
        activeSpans.put(aiServiceStartedEvent.invocationContext().invocationId(), new SpanContext(span, Context.current().with(span)));
    }
}

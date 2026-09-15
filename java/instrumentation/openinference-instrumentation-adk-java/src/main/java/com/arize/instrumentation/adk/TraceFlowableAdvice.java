package com.arize.instrumentation.adk;

import static com.arize.semconv.trace.SemanticConventions.OPENINFERENCE_SPAN_KIND;
import static com.arize.semconv.trace.SemanticConventions.OUTPUT_MIME_TYPE;
import static com.arize.semconv.trace.SemanticConventions.OUTPUT_VALUE;
import static net.bytebuddy.implementation.bytecode.assign.Assigner.Typing.DYNAMIC;

import com.arize.semconv.trace.SemanticConventions.OpenInferenceSpanKind;
import com.google.adk.events.Event;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.StatusCode;
import io.opentelemetry.sdk.trace.ReadableSpan;
import io.reactivex.rxjava3.core.Flowable;
import net.bytebuddy.asm.Advice;

/**
 * Advice for {@code Telemetry.traceFlowable(Context, Span, Supplier)}: ADK wraps the
 * {@code invocation} and {@code agent_run [...]} event streams through this helper, so the returned
 * Flowable is decorated to record the final response as the span output and to mark stream
 * failures as errors.
 */
public class TraceFlowableAdvice {

    @Advice.OnMethodExit(suppress = Throwable.class)
    public static void onExit(
            @Advice.Argument(1) Span span, @Advice.Return(readOnly = false, typing = DYNAMIC) Object returnValue) {
        if (!(returnValue instanceof Flowable<?> flowable) || !AdkSpanSupport.isActive(span)) {
            return;
        }
        returnValue = decorate(span, flowable);
    }

    /**
     * Sets the AGENT span kind on {@code agent_run} spans when the span name is readable (SDK
     * spans only; bridged or wrapped spans are left to the caller's own kind) and decorates the
     * stream. Events are only inspected when they are ADK {@link Event}s so any other element type
     * flows through untouched.
     */
    public static Flowable<?> decorate(Span span, Flowable<?> flowable) {
        if (span instanceof ReadableSpan readableSpan && readableSpan.getName().startsWith("agent_run")) {
            span.setAttribute(OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKind.AGENT.getValue());
        }
        return flowable.doOnNext(item -> onEvent(span, item)).doOnError(error -> onError(span, error));
    }

    public static void onEvent(Span span, Object item) {
        if (item instanceof Event event && event.finalResponse()) {
            AdkSpanSupport.setJson(span, OUTPUT_VALUE, OUTPUT_MIME_TYPE, event.toJson());
            span.setStatus(StatusCode.OK);
        }
    }

    public static void onError(Span span, Throwable error) {
        span.recordException(error);
        span.setStatus(
                StatusCode.ERROR, error.getMessage() == null ? error.getClass().getName() : error.getMessage());
    }
}

package com.arize.instrumentation.adk;

import com.google.adk.models.LlmRequest;
import com.google.adk.models.LlmResponse;
import io.opentelemetry.api.trace.Span;
import net.bytebuddy.asm.Advice;

/**
 * Advice for the newer span-passing {@code com.google.adk.telemetry.Tracing.traceCallLlm} methods:
 * the observed five-argument form {@code (Span, InvocationContext, String, LlmRequest, LlmResponse)}
 * (tested against ADK 0.9.0) and six-argument form with a trailing {@code Exception} (tested against
 * ADK 1.0.0 and 1.10.1). It writes OpenInference LLM attributes on the span ADK already created for
 * the call.
 *
 * <p>Unlike {@link TraceCallLlmAdvice}, ADK passes its own span as argument 0, so that span is used
 * directly instead of {@code Span.current()}. No span is created or ended, and no status is set.
 */
public class TraceCallLlmWithSpanAdvice {

    @Advice.OnMethodExit(suppress = Throwable.class)
    public static void onExit(
            @Advice.Argument(0) Span span,
            @Advice.Argument(3) LlmRequest llmRequest,
            @Advice.Argument(4) LlmResponse llmResponse) {
        if (llmRequest == null || llmResponse == null || !AdkSpanSupport.isActive(span)) {
            return;
        }
        AdkSpanSupport.writeLlmCall(span, llmRequest, llmResponse);
    }
}

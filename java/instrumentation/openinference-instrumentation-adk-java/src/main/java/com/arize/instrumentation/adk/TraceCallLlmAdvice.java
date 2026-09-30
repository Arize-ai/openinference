package com.arize.instrumentation.adk;

import com.google.adk.models.LlmRequest;
import com.google.adk.models.LlmResponse;
import io.opentelemetry.api.trace.Span;
import net.bytebuddy.asm.Advice;

/**
 * Advice for {@code Telemetry.traceCallLlm(InvocationContext, String, LlmRequest, LlmResponse)}:
 * writes OpenInference LLM attributes on the current {@code call_llm} span.
 *
 * <p>No status is set here: ADK sets ERROR on the span when the call fails, and an OK written on an
 * earlier streamed chunk would make that ERROR impossible to record.
 */
public class TraceCallLlmAdvice {

    @Advice.OnMethodExit(suppress = Throwable.class)
    public static void onExit(@Advice.Argument(2) LlmRequest llmRequest, @Advice.Argument(3) LlmResponse llmResponse) {
        Span span = Span.current();
        if (llmRequest == null || llmResponse == null || !AdkSpanSupport.isActive(span)) {
            return;
        }
        AdkSpanSupport.writeLlmCall(span, llmRequest, llmResponse);
    }
}

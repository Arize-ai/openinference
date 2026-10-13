package com.arize.instrumentation.adk;

import com.google.adk.events.Event;
import io.opentelemetry.api.trace.Span;
import net.bytebuddy.asm.Advice;

/**
 * Advice for {@code Telemetry.traceToolResponse(InvocationContext, String, Event)}: classifies the
 * current {@code tool_response [...]} span and records the function response as its output.
 *
 * <p>ADK builds the function-response event on its own span, a sibling of the {@code tool_call}
 * span rather than a child, so the two cannot be merged; the tool result is also written onto the
 * {@code tool_call} span by {@link ToolRunAdvice}. This span is marked as a CHAIN step so it does
 * not render as UNKNOWN and does not double-count as a tool call.
 */
public class TraceToolResponseAdvice {

    @Advice.OnMethodExit(suppress = Throwable.class)
    public static void onExit(@Advice.Argument(2) Event functionResponseEvent) {
        Span span = Span.current();
        if (functionResponseEvent == null || !AdkSpanSupport.isActive(span)) {
            return;
        }
        AdkSpanSupport.writeToolResponse(span, functionResponseEvent);
    }
}

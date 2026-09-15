package com.arize.instrumentation.adk;

import io.opentelemetry.api.trace.Span;
import java.util.Map;
import net.bytebuddy.asm.Advice;

/**
 * Advice for {@code Telemetry.traceToolCall(Map)}: marks the current {@code tool_call} span as a
 * TOOL span and records the arguments.
 *
 * <p>ADK calls {@code traceToolCall} before the tool executes, so no status is set here; ADK
 * records failures on the span itself and a premature OK would make them final.
 */
public class TraceToolCallAdvice {

    @Advice.OnMethodExit(suppress = Throwable.class)
    public static void onExit(@Advice.Argument(0) Map<String, Object> args) {
        Span span = Span.current();
        if (!AdkSpanSupport.isActive(span)) {
            return;
        }
        AdkSpanSupport.writeToolCall(span, args);
    }
}

package com.arize.instrumentation.adk;

import static net.bytebuddy.implementation.bytecode.assign.Assigner.Typing.DYNAMIC;

import com.google.adk.tools.BaseTool;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.sdk.trace.ReadableSpan;
import io.reactivex.rxjava3.core.Single;
import net.bytebuddy.asm.Advice;

/**
 * Advice for {@code BaseTool.runAsync(Map, ToolContext)} on every tool class: records the tool's
 * name, description and result on the current {@code tool_call [...]} span.
 *
 * <p>ADK opens the {@code tool_call} span, calls {@code Telemetry.traceToolCall(args)} and then
 * {@code tool.runAsync(args, ctx)} inside that span's scope, and only ends the span once the
 * returned {@code Single} terminates. The tool result itself is reported by ADK on a separate
 * {@code tool_response [...]} span, so this advice decorates the returned {@code Single} to copy
 * the result onto the {@code tool_call} span as {@code output.value}, which keeps the call and its
 * result on one TOOL span.
 */
public class ToolRunAdvice {

    @Advice.OnMethodExit(suppress = Throwable.class)
    public static void onExit(
            @Advice.This BaseTool tool, @Advice.Return(readOnly = false, typing = DYNAMIC) Object returnValue) {
        Span span = Span.current();
        if (!(returnValue instanceof Single<?> single) || !AdkSpanSupport.isActive(span) || !isToolCallSpan(span)) {
            return;
        }
        returnValue = decorate(span, tool, single);
    }

    /**
     * Whether {@code span} is ADK's {@code tool_call} span. The name is only readable on SDK spans;
     * other span implementations are trusted because ADK always calls {@code runAsync} inside the
     * {@code tool_call} scope.
     */
    public static boolean isToolCallSpan(Span span) {
        return !(span instanceof ReadableSpan readableSpan)
                || readableSpan.getName().startsWith("tool_call");
    }

    public static Single<?> decorate(Span span, BaseTool tool, Single<?> single) {
        AdkSpanSupport.writeToolInfo(span, tool);
        return single.doOnSuccess(result -> AdkSpanSupport.writeToolResult(span, result));
    }
}

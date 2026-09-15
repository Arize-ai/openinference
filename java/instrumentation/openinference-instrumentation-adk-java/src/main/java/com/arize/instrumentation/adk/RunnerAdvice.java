package com.arize.instrumentation.adk;

import static com.arize.semconv.trace.SemanticConventions.AGENT_NAME;
import static com.arize.semconv.trace.SemanticConventions.INPUT_MIME_TYPE;
import static com.arize.semconv.trace.SemanticConventions.INPUT_VALUE;
import static com.arize.semconv.trace.SemanticConventions.METADATA;
import static com.arize.semconv.trace.SemanticConventions.OPENINFERENCE_SPAN_KIND;
import static com.arize.semconv.trace.SemanticConventions.SESSION_ID;
import static com.arize.semconv.trace.SemanticConventions.USER_ID;
import static net.bytebuddy.implementation.bytecode.assign.Assigner.Typing.DYNAMIC;

import com.arize.semconv.trace.SemanticConventions.OpenInferenceSpanKind;
import com.google.adk.agents.BaseAgent;
import com.google.adk.sessions.Session;
import com.google.genai.types.Content;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.context.Context;
import io.reactivex.rxjava3.core.Flowable;
import java.util.Map;
import net.bytebuddy.asm.Advice;

/**
 * Advice for {@code Runner.runAsync(Session, Content, RunConfig, Map)}: stamps session, user, agent
 * and input attributes on ADK's {@code invocation} span.
 *
 * <p>ADK's {@code Telemetry.traceFlowable} makes the invocation span current when the method is
 * called and only closes that scope when the returned Flowable terminates, so at method exit
 * {@link Span#current()} is the invocation span. The advice validates the span before writing so a
 * change in that behaviour degrades to a no-op instead of decorating an unrelated span. The returned
 * Flowable is also decorated to restore the caller's context when it terminates (see
 * {@link #restoreCallerContext}).
 */
public class RunnerAdvice {

    /** The caller's context and thread, captured before ADK makes the invocation span current. */
    @Advice.OnMethodEnter
    public static Object[] onEnter() {
        return new Object[] {Context.current(), Thread.currentThread()};
    }

    @Advice.OnMethodExit(suppress = Throwable.class)
    public static void onExit(
            @Advice.Enter Object[] caller,
            @Advice.FieldValue("agent") BaseAgent agent,
            @Advice.Argument(0) Session session,
            @Advice.Argument(1) Content newMessage,
            @Advice.Argument(3) Map<String, Object> stateDelta,
            @Advice.Return(readOnly = false, typing = DYNAMIC) Object returnValue) {
        if (returnValue instanceof Flowable<?> flowable) {
            returnValue = restoreCallerContext(flowable, (Context) caller[0], (Thread) caller[1]);
        }
        Span span = Span.current();
        if (!AdkSpanSupport.isActive(span)) {
            return;
        }

        span.setAttribute(OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKind.CHAIN.getValue());
        if (agent != null && agent.name() != null) {
            span.setAttribute(AGENT_NAME, agent.name());
        }
        if (session != null) {
            if (session.userId() != null) {
                span.setAttribute(USER_ID, session.userId());
            }
            String sessionId = session.id();
            if (sessionId != null) {
                int separator = sessionId.indexOf("::");
                if (separator >= 0) {
                    sessionId = sessionId.substring(0, separator);
                }
                span.setAttribute(SESSION_ID, sessionId);
            }
        }
        if (newMessage != null) {
            AdkSpanSupport.setJson(span, INPUT_VALUE, INPUT_MIME_TYPE, newMessage.toJson());
        }
        if (stateDelta != null && !stateDelta.isEmpty()) {
            AdkSpanSupport.set(span, METADATA, AdkSpanSupport.toJson(stateDelta));
        }
    }

    /**
     * Puts the caller's context back once the invocation stream terminates.
     *
     * <p>ADK's {@code traceFlowable} closes the invocation scope in a {@code doFinally}, but RxJava
     * runs an outer {@code doFinally} before the inner ones, so the invocation scope is closed while
     * the {@code agent_run} scope is still current. OpenTelemetry ignores that out-of-order close
     * and the finished invocation span stays current on the calling thread: the next
     * {@code runAsync} on the same thread nests under it and a surrounding
     * {@code SuppressTracing} scope can never be closed. Restoring the caller's context here makes
     * ADK's own closes no-ops and leaves the thread as it was found. Only the thread that called
     * {@code runAsync} is touched; a stream that terminates elsewhere is left alone.
     */
    public static Flowable<?> restoreCallerContext(Flowable<?> flowable, Context caller, Thread thread) {
        return flowable.doFinally(() -> {
            if (Thread.currentThread() == thread && Context.current() != caller) {
                // The returned Scope is intentionally not closed: attaching the caller's context is
                // the restoration itself.
                caller.makeCurrent();
            }
        });
    }
}

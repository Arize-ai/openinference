package com.arize.instrumentation.adk;

import static net.bytebuddy.matcher.ElementMatchers.hasSuperType;
import static net.bytebuddy.matcher.ElementMatchers.isInterface;
import static net.bytebuddy.matcher.ElementMatchers.isStatic;
import static net.bytebuddy.matcher.ElementMatchers.named;
import static net.bytebuddy.matcher.ElementMatchers.not;
import static net.bytebuddy.matcher.ElementMatchers.takesArgument;
import static net.bytebuddy.matcher.ElementMatchers.takesArguments;

import java.lang.instrument.Instrumentation;
import java.util.Collections;
import java.util.List;
import net.bytebuddy.agent.builder.AgentBuilder;
import net.bytebuddy.asm.Advice;
import net.bytebuddy.utility.JavaModule;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Java agent that decorates the spans Google ADK already creates with OpenInference attributes.
 *
 * <p>The complete invocation, agent, LLM, and tool span coverage targets {@code
 * com.google.adk.Telemetry} (ADK 0.4.0). The additional compatibility work only enriches {@code
 * call_llm} spans for the observed {@code com.google.adk.telemetry.Tracing.traceCallLlm}
 * signatures (tested against ADK 0.6.0, 0.9.0, 1.0.0, and 1.10.1). Type and argument matchers use
 * class names rather than class literals so that no ADK class is loaded by the agent itself.
 */
public class TelemetryAgent {

    private static final Logger log = LoggerFactory.getLogger(TelemetryAgent.class);

    public static void premain(String agentArgs, Instrumentation inst) {
        install(inst);
    }

    public static void agentmain(String agentArgs, Instrumentation inst) {
        install(inst);
    }

    static void install(Instrumentation inst) {
        new AgentBuilder.Default()
                // Advice only rewrites method bodies, so classes that ADK already loaded before the
                // agent was attached (agentmain) can be retransformed in place.
                .disableClassFormatChanges()
                .with(AgentBuilder.RedefinitionStrategy.RETRANSFORMATION)
                .with(new AgentBuilder.RedefinitionStrategy.Listener.Adapter() {
                    @Override
                    public Iterable<? extends List<Class<?>>> onError(
                            int index, List<Class<?>> batch, Throwable throwable, List<Class<?>> types) {
                        log.warn("OpenInference ADK instrumentation failed to retransform {}", batch, throwable);
                        return Collections.emptyList();
                    }
                })
                .with(new AgentBuilder.Listener.Adapter() {
                    @Override
                    public void onError(
                            String typeName,
                            ClassLoader classLoader,
                            JavaModule module,
                            boolean loaded,
                            Throwable throwable) {
                        log.warn("OpenInference ADK instrumentation failed to transform {}", typeName, throwable);
                    }
                })
                .type(named("com.google.adk.Telemetry"))
                .transform((builder, typeDescription, classLoader, javaModule, protectionDomain) -> builder.visit(
                                Advice.to(TraceFlowableAdvice.class).on(named("traceFlowable")))
                        .visit(Advice.to(TraceToolCallAdvice.class).on(named("traceToolCall")))
                        .visit(Advice.to(TraceToolResponseAdvice.class).on(named("traceToolResponse")))
                        .visit(Advice.to(TraceCallLlmAdvice.class).on(named("traceCallLlm"))))
                .type(named("com.google.adk.telemetry.Tracing"))
                .transform((builder, typeDescription, classLoader, javaModule, protectionDomain) -> builder.visit(
                                Advice.to(TraceCallLlmAdvice.class)
                                        .on(isStatic()
                                                .and(named("traceCallLlm"))
                                                .and(takesArguments(4))
                                                .and(takesArgument(0, named("com.google.adk.agents.InvocationContext")))
                                                .and(takesArgument(1, named("java.lang.String")))
                                                .and(takesArgument(2, named("com.google.adk.models.LlmRequest")))
                                                .and(takesArgument(3, named("com.google.adk.models.LlmResponse")))))
                        .visit(Advice.to(TraceCallLlmWithSpanAdvice.class)
                                .on(isStatic()
                                        .and(named("traceCallLlm"))
                                        .and(takesArguments(5))
                                        .and(takesArgument(0, named("io.opentelemetry.api.trace.Span")))
                                        .and(takesArgument(1, named("com.google.adk.agents.InvocationContext")))
                                        .and(takesArgument(2, named("java.lang.String")))
                                        .and(takesArgument(3, named("com.google.adk.models.LlmRequest")))
                                        .and(takesArgument(4, named("com.google.adk.models.LlmResponse")))
                                        .or(isStatic()
                                                .and(named("traceCallLlm"))
                                                .and(takesArguments(6))
                                                .and(takesArgument(0, named("io.opentelemetry.api.trace.Span")))
                                                .and(takesArgument(1, named("com.google.adk.agents.InvocationContext")))
                                                .and(takesArgument(2, named("java.lang.String")))
                                                .and(takesArgument(3, named("com.google.adk.models.LlmRequest")))
                                                .and(takesArgument(4, named("com.google.adk.models.LlmResponse")))
                                                .and(takesArgument(5, named("java.lang.Exception")))))))
                // BaseTool itself is excluded: BaseTool.runAsync(Map, ToolContext) is concrete
                // and its default implementation throws UnsupportedOperationException. Resolving
                // the typed ToolRunAdvice while BaseTool itself is being defined causes the
                // cold-load/reentrant-definition problem. Executable subclasses that override
                // the method remain matched.
                .type(not(named("com.google.adk.tools.BaseTool"))
                        .and(hasSuperType(named("com.google.adk.tools.BaseTool")))
                        .and(not(isInterface())))
                .transform((builder, typeDescription, classLoader, javaModule, protectionDomain) ->
                        builder.visit(Advice.to(ToolRunAdvice.class)
                                .on(named("runAsync")
                                        .and(takesArguments(2))
                                        .and(takesArgument(1, named("com.google.adk.tools.ToolContext"))))))
                .type(named("com.google.adk.runner.Runner"))
                .transform((builder, typeDescription, classLoader, javaModule, protectionDomain) ->
                        builder.visit(Advice.to(RunnerAdvice.class)
                                .on(named("runAsync")
                                        .and(takesArguments(4))
                                        .and(takesArgument(0, named("com.google.adk.sessions.Session"))))))
                .installOn(inst);
        log.info("OpenInference ADK instrumentation installed");
    }
}

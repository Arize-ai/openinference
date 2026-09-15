package com.arize.instrumentation.adk;

import static net.bytebuddy.matcher.ElementMatchers.hasSuperType;
import static net.bytebuddy.matcher.ElementMatchers.isInterface;
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
 * <p>ADK's {@code com.google.adk.Telemetry} helpers are advised so LLM and tool spans get
 * OpenInference span kinds, messages and token counts, every {@code BaseTool.runAsync} is advised
 * so the tool result lands on the {@code tool_call} span, and {@code Runner.runAsync} is advised
 * so the invocation span gets session, user and input attributes. Type and argument matchers use
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
                .type(hasSuperType(named("com.google.adk.tools.BaseTool")).and(not(isInterface())))
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

package com.arize.instrumentation.adk;

import static net.bytebuddy.matcher.ElementMatchers.named;

import com.google.adk.agents.RunConfig;
import com.google.adk.sessions.Session;
import com.google.genai.types.Content;
import java.lang.instrument.Instrumentation;
import java.util.Map;
import net.bytebuddy.agent.builder.AgentBuilder;
import net.bytebuddy.asm.Advice;
import net.bytebuddy.matcher.ElementMatchers;

/**
 * Java agent for enhancing Telemetry class at runtime.
 */
public class TelemetryAgent {

    public static void premain(String agentArgs, Instrumentation inst) {
        new AgentBuilder.Default()
                .type(named("com.google.adk.Telemetry"))
                .transform((builder, typeDescription, classLoader, javaModule, protectionDomain) -> builder.method(
                                named("traceToolCall")
                                        .or(named("traceToolResponse"))
                                        .or(named("traceCallLlm"))
                                        .or(named("traceFlowable")))
                        .intercept(Advice.to(TelemetryAdvice.class)))
                // Rule 2: Instrument com.google.adk.runner.Runner
                .type(named("com.google.adk.runner.Runner"))
                .transform((builder, typeDescription, classLoader, javaModule, protectionDomain) -> builder.method(
                                named("runAsync")
                                        .and(ElementMatchers.takesArguments(
                                                Session.class, Content.class, RunConfig.class, Map.class)))
                        .intercept(Advice.to(RunnerAdvice.class)))
                .installOn(inst);
    }
}

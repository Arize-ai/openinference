package com.arize.instrumentation.annotations;

import static net.bytebuddy.matcher.ElementMatchers.*;

import java.lang.instrument.Instrumentation;
import java.util.jar.JarFile;
import net.bytebuddy.agent.ByteBuddyAgent;
import net.bytebuddy.agent.builder.AgentBuilder;
import net.bytebuddy.asm.Advice;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class OpenInferenceAgentInstaller {

    private static final Logger log = LoggerFactory.getLogger(OpenInferenceAgentInstaller.class);

    /**
     * Install the annotation tracing agent at runtime (no -javaagent flag needed).
     * Call this before any annotated classes are loaded for best results.
     */
    public static void install() {
        log.info("Installing OpenInference annotation tracing agent (runtime attach)");
        Instrumentation inst = ByteBuddyAgent.install();
        installAgent(inst);
        log.info("OpenInference annotation tracing agent installed");
    }

    /**
     * Entry point when loaded as a -javaagent.
     */
    public static void premain(String args, Instrumentation inst) {
        log.info("Installing OpenInference annotation tracing agent (premain)");

        try {
            String agentJarPath = OpenInferenceAgentInstaller.class
                    .getProtectionDomain()
                    .getCodeSource()
                    .getLocation()
                    .toURI()
                    .getPath();
            inst.appendToBootstrapClassLoaderSearch(new JarFile(agentJarPath));
        } catch (Exception e) {
            log.warn("Failed to append agent JAR to bootstrap classloader", e);
        }

        installAgent(inst);
        log.info("OpenInference annotation tracing agent installed");
    }

    private static void installAgent(Instrumentation inst) {
        new AgentBuilder.Default()
                .with(AgentBuilder.RedefinitionStrategy.RETRANSFORMATION)
                .type(declaresMethod(isAnnotatedWith(TraceChain.class)
                        .or(isAnnotatedWith(TraceLLM.class))
                        .or(isAnnotatedWith(TraceTool.class))
                        .or(isAnnotatedWith(TraceAgent.class))))
                .transform((builder, type, classLoader, module, domain) -> builder.visit(Advice.to(TraceAdvice.class)
                        .on(isAnnotatedWith(TraceChain.class)
                                .or(isAnnotatedWith(TraceLLM.class))
                                .or(isAnnotatedWith(TraceTool.class))
                                .or(isAnnotatedWith(TraceAgent.class)))))
                .with(AgentBuilder.Listener.StreamWriting.toSystemError().withErrorsOnly())
                .installOn(inst);
    }
}

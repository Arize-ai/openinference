package com.arize.instrumentation.annotation;

import static net.bytebuddy.matcher.ElementMatchers.*;

import java.io.File;
import java.lang.instrument.Instrumentation;
import java.net.URI;
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
        appendToBootstrapClassLoader(inst);
        installAgent(inst);
        log.info("OpenInference annotation tracing agent installed");
    }

    /**
     * Entry point when attached to a running JVM.
     */
    public static void agentmain(String args, Instrumentation inst) {
        log.info("Installing OpenInference annotation tracing agent (agentmain)");
        appendToBootstrapClassLoader(inst);
        installAgent(inst);
        log.info("OpenInference annotation tracing agent installed");
    }

    private static void appendToBootstrapClassLoader(Instrumentation inst) {
        try {
            URI agentJarUri = OpenInferenceAgentInstaller.class
                    .getProtectionDomain()
                    .getCodeSource()
                    .getLocation()
                    .toURI();
            inst.appendToBootstrapClassLoaderSearch(new JarFile(new File(agentJarUri)));
        } catch (Exception e) {
            log.warn("Failed to append agent JAR to bootstrap classloader", e);
        }
    }

    private static void installAgent(Instrumentation inst) {
        new AgentBuilder.Default()
                .with(AgentBuilder.RedefinitionStrategy.RETRANSFORMATION)
                .type(declaresMethod(isAnnotatedWith(Chain.class)
                        .or(isAnnotatedWith(LLM.class))
                        .or(isAnnotatedWith(Tool.class))
                        .or(isAnnotatedWith(Agent.class))
                        .or(isAnnotatedWith(Span.class))))
                .transform((builder, type, classLoader, module, domain) -> builder.visit(Advice.to(TraceAdvice.class)
                        .on(isAnnotatedWith(Chain.class)
                                .or(isAnnotatedWith(LLM.class))
                                .or(isAnnotatedWith(Tool.class))
                                .or(isAnnotatedWith(Agent.class))
                                .or(isAnnotatedWith(Span.class)))))
                .with(AgentBuilder.Listener.StreamWriting.toSystemError().withErrorsOnly())
                .installOn(inst);
    }
}

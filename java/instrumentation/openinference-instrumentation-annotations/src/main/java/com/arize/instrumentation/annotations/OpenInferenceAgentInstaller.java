package com.arize.instrumentation.annotations;

import static net.bytebuddy.matcher.ElementMatchers.*;

import java.lang.instrument.Instrumentation;
import java.util.jar.JarFile;
import net.bytebuddy.agent.builder.AgentBuilder;
import net.bytebuddy.asm.Advice;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class OpenInferenceAgentInstaller {

    private static final Logger log = LoggerFactory.getLogger(OpenInferenceAgentInstaller.class);

    public static void premain(String args, Instrumentation inst) {
        log.info("Installing OpenInference annotation tracing agent");

        // Append agent JAR to bootstrap classloader so that advice classes
        // (TracedSpan, TraceAdvice, etc.) are visible from any classloader.
        // ByteBuddy Advice is inlined into target methods, so all referenced
        // classes must be on the target class's classloader or bootstrap.
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

        log.info("OpenInference annotation tracing agent installed");
    }
}

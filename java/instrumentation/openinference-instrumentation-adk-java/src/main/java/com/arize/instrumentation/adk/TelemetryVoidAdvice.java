package com.arize.instrumentation.adk;

import java.lang.reflect.Method;
import net.bytebuddy.asm.Advice;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Advice class for intercepting void methods in Telemetry.
 */
public class TelemetryVoidAdvice {

    public static final Logger log = LoggerFactory.getLogger(TelemetryVoidAdvice.class);

    @Advice.OnMethodExit
    public static void onExit(@Advice.Origin Method method, @Advice.AllArguments Object[] args) {
        log.info("Enhancing void method {} onExit.", method.getName());

        switch (method.getName()) {
            case "traceToolCall":
                TelemetryAdvice.handleTraceToolCall(args);
                break;
            case "traceCallLlm":
                TelemetryAdvice.handleTraceCallLlm(args);
                break;
            default:
                break;
        }
    }
}
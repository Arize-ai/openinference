package com.arize.instrumentation.annotations;

import com.arize.instrumentation.OITracer;
import java.util.concurrent.atomic.AtomicBoolean;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class OpenInferenceAgent {

    private static final Logger log = LoggerFactory.getLogger(OpenInferenceAgent.class);
    private static volatile OITracer tracer;
    private static final AtomicBoolean registered = new AtomicBoolean(false);

    public static void register(OITracer tracer) {
        if (registered.compareAndSet(false, true)) {
            OpenInferenceAgent.tracer = tracer;
            log.info("OpenInference tracing registered");
        } else {
            log.warn("OpenInferenceAgent already registered. Ignoring duplicate registration.");
        }
    }

    public static void unregister() {
        registered.set(false);
        OpenInferenceAgent.tracer = null;
        log.info("OpenInference tracing unregistered");
    }

    static OITracer getTracer() {
        return tracer;
    }
}

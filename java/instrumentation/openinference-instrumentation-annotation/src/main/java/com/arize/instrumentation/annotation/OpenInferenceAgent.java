package com.arize.instrumentation.annotation;

import com.arize.instrumentation.OITracer;
import java.util.concurrent.atomic.AtomicReference;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class OpenInferenceAgent {

    private static final Logger log = LoggerFactory.getLogger(OpenInferenceAgent.class);
    private static final AtomicReference<OITracer> tracerRef = new AtomicReference<>(null);

    public static void register(OITracer tracer) {
        if (tracerRef.compareAndSet(null, tracer)) {
            log.info("OpenInference tracing registered");
        } else {
            log.warn("OpenInferenceAgent already registered. Ignoring duplicate registration.");
        }
    }

    public static void unregister() {
        tracerRef.set(null);
        log.info("OpenInference tracing unregistered");
    }

    public static OITracer getTracer() {
        return tracerRef.get();
    }
}

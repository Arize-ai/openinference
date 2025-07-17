package com.arize.instrumentation.langchain4j;

import com.arize.instrumentation.OITracer;
import com.arize.instrumentation.TraceConfig;
import io.opentelemetry.api.GlobalOpenTelemetry;
import io.opentelemetry.api.trace.Tracer;
import io.opentelemetry.api.trace.TracerProvider;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.logging.Logger;

/**
 * Main instrumentor for LangChain4j that sets up OpenInference tracing.
 */
public class LangChain4jInstrumentor {

    private static final Logger logger = Logger.getLogger(LangChain4jInstrumentor.class.getName());
    private static final String INSTRUMENTATION_NAME = "com.arize.langchain4j";
    private static final AtomicBoolean isInstrumented = new AtomicBoolean(false);

    private final OITracer tracer;
    private final TraceConfig config;

    private LangChain4jInstrumentor(OITracer tracer, TraceConfig config) {
        this.tracer = tracer;
        this.config = config;
    }

    /**
     * Instruments LangChain4j with default configuration.
     */
    public static LangChain4jInstrumentor instrument() {
        return instrument(GlobalOpenTelemetry.getTracerProvider(), TraceConfig.getDefault());
    }

    /**
     * Instruments LangChain4j with custom tracer provider.
     */
    public static LangChain4jInstrumentor instrument(TracerProvider tracerProvider) {
        return instrument(tracerProvider, TraceConfig.getDefault());
    }

    /**
     * Instruments LangChain4j with custom configuration.
     */
    public static LangChain4jInstrumentor instrument(TraceConfig config) {
        return instrument(GlobalOpenTelemetry.getTracerProvider(), config);
    }

    /**
     * Instruments LangChain4j with custom tracer provider and configuration.
     */
    public static LangChain4jInstrumentor instrument(TracerProvider tracerProvider, TraceConfig config) {
        if (isInstrumented.compareAndSet(false, true)) {
            logger.info("Instrumenting LangChain4j with OpenInference");

            Tracer otelTracer = tracerProvider.get(INSTRUMENTATION_NAME);
            OITracer tracer = new OITracer(otelTracer, config);

            LangChain4jInstrumentor instrumentor = new LangChain4jInstrumentor(tracer, config);

            // Register interceptors and callbacks
            instrumentor.registerInterceptors();

            return instrumentor;
        } else {
            logger.warning("LangChain4j is already instrumented");
            throw new IllegalStateException("LangChain4j is already instrumented");
        }
    }

    /**
     * Uninstruments LangChain4j.
     */
    public void uninstrument() {
        if (isInstrumented.compareAndSet(true, false)) {
            logger.info("Uninstrumenting LangChain4j");
            // Unregister interceptors and callbacks
            unregisterInterceptors();
        }
    }

    private void registerInterceptors() {
        // Register model listeners
        LangChain4jModelListener modelListener = new LangChain4jModelListener(tracer);

        // TODO: Register with LangChain4j's listener/interceptor mechanism
        // This will depend on LangChain4j's API for adding global interceptors

        logger.info("Registered LangChain4j interceptors");
    }

    private void unregisterInterceptors() {
        // TODO: Unregister from LangChain4j
        logger.info("Unregistered LangChain4j interceptors");
    }

    public OITracer getTracer() {
        return tracer;
    }

    public TraceConfig getConfig() {
        return config;
    }

    /**
     * Creates a model listener for manual instrumentation.
     */
    public LangChain4jModelListener createModelListener() {
        return new LangChain4jModelListener(tracer);
    }
}

package com.arize.instrumentation.embabel;

import com.arize.instrumentation.TraceConfig;
import io.opentelemetry.sdk.common.CompletableResultCode;
import io.opentelemetry.sdk.trace.data.SpanData;
import io.opentelemetry.sdk.trace.export.SpanExporter;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Objects;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A {@link SpanExporter} that adds OpenInference attributes to the spans Embabel's
 * {@code embabel-agent-observability} module emits, then hands every span to {@code delegate}.
 *
 * <p>Agent runs become {@code AGENT} spans, Spring AI chat model calls {@code LLM} spans (model,
 * provider, token counts, invocation parameters, input and output messages), tool calls
 * {@code TOOL} spans, embeddings {@code EMBEDDING} spans, RAG lookups {@code RETRIEVER} spans and
 * the remaining Embabel steps (actions, goals, planning, tool loops) {@code CHAIN} spans. Other
 * spans pass through unchanged.
 *
 * <p>Content that a {@link TraceConfig} hide flag covers is removed from the exported span,
 * including the copies Embabel writes under its own and the GenAI keys.
 *
 * <p>Spans are exported unmapped while {@link TraceConfig#isSuppressTracing()} is set, and so are
 * spans that {@link OpenInferenceEmbabelSpanProcessor} saw start inside
 * {@link com.arize.instrumentation.SuppressTracing#begin()}. An exporter cannot see the context a
 * span started in, so the processor must be registered for {@code SuppressTracing} to apply.
 */
public final class OpenInferenceEmbabelSpanExporter implements SpanExporter {

    private static final Logger log = LoggerFactory.getLogger(OpenInferenceEmbabelSpanExporter.class);

    private final SpanExporter delegate;
    private final EmbabelSpanMapper mapper;

    public OpenInferenceEmbabelSpanExporter(SpanExporter delegate) {
        this(delegate, TraceConfig.getDefault());
    }

    public OpenInferenceEmbabelSpanExporter(SpanExporter delegate, TraceConfig config) {
        this.delegate = Objects.requireNonNull(delegate, "delegate must not be null");
        this.mapper = new EmbabelSpanMapper(Objects.requireNonNull(config, "config must not be null"));
    }

    @Override
    public CompletableResultCode export(Collection<SpanData> spans) {
        List<SpanData> mapped = new ArrayList<>(spans.size());
        for (SpanData span : spans) {
            mapped.add(mapSafely(span));
        }
        return delegate.export(mapped);
    }

    private SpanData mapSafely(SpanData span) {
        try {
            return mapper.map(span);
        } catch (RuntimeException e) {
            log.debug("Failed to add OpenInference attributes to span {}", span.getName(), e);
            return span;
        }
    }

    @Override
    public CompletableResultCode flush() {
        return delegate.flush();
    }

    @Override
    public CompletableResultCode shutdown() {
        return delegate.shutdown();
    }

    @Override
    public String toString() {
        return "OpenInferenceEmbabelSpanExporter{delegate=" + delegate + "}";
    }
}

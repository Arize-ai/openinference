package com.arize.instrumentation.adk;

import static org.assertj.core.api.Assertions.assertThat;

import io.opentelemetry.api.common.AttributeKey;
import io.opentelemetry.sdk.testing.exporter.InMemorySpanExporter;
import io.opentelemetry.sdk.trace.data.SpanData;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

/** Small helpers shared by the tests for reading exported spans. */
final class SpanAssertions {

    private SpanAssertions() {}

    static String str(SpanData span, String key) {
        return span.getAttributes().get(AttributeKey.stringKey(key));
    }

    static Long lng(SpanData span, String key) {
        return span.getAttributes().get(AttributeKey.longKey(key));
    }

    static List<String> keys(SpanData span) {
        return span.getAttributes().asMap().keySet().stream()
                .map(AttributeKey::getKey)
                .sorted()
                .collect(Collectors.toList());
    }

    static List<String> keysStartingWith(SpanData span, String prefix) {
        return keys(span).stream().filter(k -> k.startsWith(prefix)).collect(Collectors.toList());
    }

    static SpanData single(InMemorySpanExporter exporter) {
        List<SpanData> spans = exporter.getFinishedSpanItems();
        assertThat(spans).hasSize(1);
        return spans.get(0);
    }

    static SpanData named(List<SpanData> spans, String name) {
        Optional<SpanData> match =
                spans.stream().filter(s -> s.getName().equals(name)).findFirst();
        assertThat(match).as("span named %s in %s", name, names(spans)).isPresent();
        return match.get();
    }

    static List<SpanData> allNamed(List<SpanData> spans, String name) {
        return spans.stream().filter(s -> s.getName().equals(name)).collect(Collectors.toList());
    }

    static List<String> names(List<SpanData> spans) {
        return spans.stream().map(SpanData::getName).collect(Collectors.toList());
    }

    /** Waits for {@code count} finished spans; ADK may end spans on RxJava threads. */
    static List<SpanData> awaitSpans(InMemorySpanExporter exporter, int count) throws InterruptedException {
        long deadline = System.currentTimeMillis() + 10_000;
        while (exporter.getFinishedSpanItems().size() < count && System.currentTimeMillis() < deadline) {
            Thread.sleep(20);
        }
        List<SpanData> spans = exporter.getFinishedSpanItems();
        assertThat(spans).as("finished spans: %s", names(spans)).hasSizeGreaterThanOrEqualTo(count);
        return spans;
    }
}

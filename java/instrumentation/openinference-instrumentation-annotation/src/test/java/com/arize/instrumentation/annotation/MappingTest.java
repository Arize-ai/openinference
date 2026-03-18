package com.arize.instrumentation.annotation;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.Map;
import org.junit.jupiter.api.Test;

class MappingTest {

    @Test
    void extractFieldSimplePath() {
        Map<String, Object> obj = Map.of("name", "test");
        Object result = SpanHelper.extractField(obj, "name");
        assertThat(result).isEqualTo("test");
    }

    @Test
    void extractFieldNestedPath() {
        Map<String, Object> obj = Map.of("usage", Map.of("totalTokens", 150));
        Object result = SpanHelper.extractField(obj, "usage.totalTokens");
        assertThat(result).isEqualTo(150);
    }

    @Test
    void extractFieldMissingPathReturnsNull() {
        Map<String, Object> obj = Map.of("name", "test");
        Object result = SpanHelper.extractField(obj, "nonexistent.path");
        assertThat(result).isNull();
    }

    @Test
    void extractFieldNullObjectReturnsNull() {
        Object result = SpanHelper.extractField(null, "name");
        assertThat(result).isNull();
    }

    @Test
    void extractFieldEmptyPathReturnsNull() {
        Object result = SpanHelper.extractField(Map.of("k", "v"), "");
        assertThat(result).isNull();
    }
}

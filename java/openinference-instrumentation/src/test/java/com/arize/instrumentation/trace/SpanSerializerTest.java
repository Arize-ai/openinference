package com.arize.instrumentation.trace;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.Map;
import org.junit.jupiter.api.Test;

class SpanSerializerTest {

    @Test
    void serializeStringReturnsStringDirectly() {
        SpanSerializer.SerializedValue result = SpanSerializer.serialize("hello");
        assertThat(result.value()).isEqualTo("hello");
        assertThat(result.isJson()).isFalse();
    }

    @Test
    void serializeObjectReturnsJson() {
        SpanSerializer.SerializedValue result = SpanSerializer.serialize(Map.of("key", "value"));
        assertThat(result.value()).contains("key");
        assertThat(result.value()).contains("value");
        assertThat(result.isJson()).isTrue();
    }

    @Test
    void serializeNullReturnsNull() {
        SpanSerializer.SerializedValue result = SpanSerializer.serialize(null);
        assertThat(result).isNull();
    }

    @Test
    void serializeNonSerializableReturnsNonNull() {
        Object circular = new Object() {
            @Override
            public String toString() {
                return "not-serializable";
            }
        };
        // Should not throw, should return fallback or null
        SpanSerializer.SerializedValue result = SpanSerializer.serialize(circular);
        assertThat(result).isNotNull();
    }

    @Test
    void serializeIntegerReturnsJson() {
        SpanSerializer.SerializedValue result = SpanSerializer.serialize(42);
        assertThat(result.value()).isEqualTo("42");
        assertThat(result.isJson()).isTrue();
    }

    @Test
    void serializeListReturnsJson() {
        SpanSerializer.SerializedValue result = SpanSerializer.serialize(java.util.List.of("a", "b", "c"));
        assertThat(result.isJson()).isTrue();
        assertThat(result.value()).contains("a");
        assertThat(result.value()).contains("b");
    }

    @Test
    void serializeBooleanReturnsJson() {
        SpanSerializer.SerializedValue result = SpanSerializer.serialize(true);
        assertThat(result.value()).isEqualTo("true");
        assertThat(result.isJson()).isTrue();
    }

    @Test
    void nonSerializableFallsBackToString() {
        Object obj = new Object() {
            @Override
            public String toString() {
                return "fallback-string";
            }
        };
        SpanSerializer.SerializedValue result = SpanSerializer.serialize(obj);
        assertThat(result).isNotNull();
        assertThat(result.value()).isEqualTo("fallback-string");
        assertThat(result.isJson()).isFalse();
    }

    @Test
    void extractFieldSimplePath() {
        Map<String, Object> obj = Map.of("name", "test");
        Object result = SpanSerializer.extractField(obj, "name");
        assertThat(result).isEqualTo("test");
    }

    @Test
    void extractFieldNestedPath() {
        Map<String, Object> obj = Map.of("usage", Map.of("totalTokens", 150));
        Object result = SpanSerializer.extractField(obj, "usage.totalTokens");
        assertThat(result).isEqualTo(150);
    }

    @Test
    void extractFieldMissingPathReturnsNull() {
        Map<String, Object> obj = Map.of("name", "test");
        Object result = SpanSerializer.extractField(obj, "nonexistent.path");
        assertThat(result).isNull();
    }

    @Test
    void extractFieldNullObjectReturnsNull() {
        Object result = SpanSerializer.extractField(null, "name");
        assertThat(result).isNull();
    }

    @Test
    void extractFieldEmptyPathReturnsNull() {
        Object result = SpanSerializer.extractField(Map.of("k", "v"), "");
        assertThat(result).isNull();
    }
}

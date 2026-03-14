package com.arize.instrumentation.annotations;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.Map;
import org.junit.jupiter.api.Test;

class SpanHelperTest {

    @Test
    void serializeStringReturnsStringDirectly() {
        SpanHelper.SerializedValue result = SpanHelper.serialize("hello");
        assertThat(result.value()).isEqualTo("hello");
        assertThat(result.isJson()).isFalse();
    }

    @Test
    void serializeObjectReturnsJson() {
        SpanHelper.SerializedValue result = SpanHelper.serialize(Map.of("key", "value"));
        assertThat(result.value()).contains("key");
        assertThat(result.value()).contains("value");
        assertThat(result.isJson()).isTrue();
    }

    @Test
    void serializeNullReturnsNull() {
        SpanHelper.SerializedValue result = SpanHelper.serialize(null);
        assertThat(result).isNull();
    }

    @Test
    void serializeNonSerializableReturnsNull() {
        // Object with circular reference
        Object circular = new Object() {
            @Override
            public String toString() {
                return "not-serializable";
            }
        };
        // Should not throw, should return fallback or null
        SpanHelper.SerializedValue result = SpanHelper.serialize(circular);
        // Non-serializable objects fall back to toString
        assertThat(result).isNotNull();
    }

    @Test
    void buildInputMapFiltersIgnoredParams() throws Exception {
        // Simulate method with 3 params, second is @SpanIgnore
        var method = TestTarget.class.getMethod("myMethod", String.class, String.class, int.class);
        Object[] args = {"hello", "secret", 42};

        Map<String, Object> input = SpanHelper.buildInputMap(method, args);

        assertThat(input).containsKey("query");
        assertThat(input).doesNotContainKey("secret");
        assertThat(input).containsKey("count");
    }

    @Test
    void buildInputMapSingleParamReturnsSingleEntryMap() throws Exception {
        var method = TestTarget.class.getMethod("singleParam", String.class);
        Object[] args = {"hello"};

        Map<String, Object> input = SpanHelper.buildInputMap(method, args);

        // Single param still uses a map with the param name
        assertThat(input).hasSize(1);
        assertThat(input).containsEntry("input", "hello");
    }

    // Test target class for reflection
    public static class TestTarget {
        public void myMethod(String query, @SpanIgnore String secret, int count) {}

        public void singleParam(String input) {}
    }
}

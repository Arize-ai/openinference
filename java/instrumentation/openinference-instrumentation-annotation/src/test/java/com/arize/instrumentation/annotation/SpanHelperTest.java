package com.arize.instrumentation.annotation;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.Map;
import org.junit.jupiter.api.Test;

class SpanHelperTest {

    @Test
    void buildInputMapFiltersIgnoredParams() throws Exception {
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

        assertThat(input).hasSize(1);
        assertThat(input).containsEntry("input", "hello");
    }

    @Test
    void buildInputMapNoParams() throws Exception {
        var method = TestTarget.class.getMethod("noParams");
        Object[] args = {};

        Map<String, Object> input = SpanHelper.buildInputMap(method, args);

        assertThat(input).isEmpty();
    }

    @Test
    void buildInputMapAllIgnored() throws Exception {
        var method = TestTarget.class.getMethod("allIgnored", String.class, String.class);
        Object[] args = {"a", "b"};

        Map<String, Object> input = SpanHelper.buildInputMap(method, args);

        assertThat(input).isEmpty();
    }

    // Test target class for reflection
    public static class TestTarget {
        public void myMethod(String query, @SpanIgnore String secret, int count) {}

        public void singleParam(String input) {}

        public void noParams() {}

        public void allIgnored(@SpanIgnore String a, @SpanIgnore String b) {}
    }
}

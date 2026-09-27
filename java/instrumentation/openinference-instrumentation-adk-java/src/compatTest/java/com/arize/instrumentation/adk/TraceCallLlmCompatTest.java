package com.arize.instrumentation.adk;

import static com.arize.semconv.trace.SemanticConventions.INPUT_MIME_TYPE;
import static com.arize.semconv.trace.SemanticConventions.INPUT_VALUE;
import static com.arize.semconv.trace.SemanticConventions.LLM_INPUT_MESSAGES;
import static com.arize.semconv.trace.SemanticConventions.LLM_MODEL_NAME;
import static com.arize.semconv.trace.SemanticConventions.LLM_OUTPUT_MESSAGES;
import static com.arize.semconv.trace.SemanticConventions.LLM_PROVIDER;
import static com.arize.semconv.trace.SemanticConventions.MESSAGE_CONTENTS;
import static com.arize.semconv.trace.SemanticConventions.MESSAGE_CONTENT_TEXT;
import static com.arize.semconv.trace.SemanticConventions.MESSAGE_ROLE;
import static com.arize.semconv.trace.SemanticConventions.OPENINFERENCE_SPAN_KIND;
import static com.arize.semconv.trace.SemanticConventions.OUTPUT_MIME_TYPE;
import static com.arize.semconv.trace.SemanticConventions.OUTPUT_VALUE;
import static org.assertj.core.api.Assertions.assertThat;

import io.opentelemetry.api.common.AttributeKey;
import io.opentelemetry.api.common.Attributes;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.StatusCode;
import io.opentelemetry.api.trace.Tracer;
import io.opentelemetry.context.Scope;
import io.opentelemetry.sdk.testing.exporter.InMemorySpanExporter;
import io.opentelemetry.sdk.trace.SdkTracerProvider;
import io.opentelemetry.sdk.trace.data.SpanData;
import io.opentelemetry.sdk.trace.export.SimpleSpanProcessor;
import java.lang.instrument.Instrumentation;
import java.lang.reflect.Method;
import java.util.List;
import java.util.Map;
import net.bytebuddy.agent.ByteBuddyAgent;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.mockito.stubbing.Answer;

/**
 * Exercises the installed agent against one real ADK version per Gradle task. This class compiles
 * without any {@code com.google.adk.*} type on its classpath and reaches ADK only by reflection, so
 * the same compiled test runs against ADK 0.6.0, 0.9.0, 1.0.0 and 1.10.1 in separate JVMs.
 *
 * <p>The expected {@code traceCallLlm} arity is supplied by the task through {@code
 * adk.expected.arity}. A missing class or method fails the test; it is never skipped.
 */
class TraceCallLlmCompatTest {

    private static final String TRACING = "com.google.adk.telemetry.Tracing";
    private static final String INVOCATION_CONTEXT = "com.google.adk.agents.InvocationContext";
    private static final String SESSION = "com.google.adk.sessions.Session";
    private static final String LLM_REQUEST = "com.google.adk.models.LlmRequest";
    private static final String LLM_RESPONSE = "com.google.adk.models.LlmResponse";
    private static final String LIVE_CONNECT_CONFIG = "com.google.genai.types.LiveConnectConfig";
    private static final String CONTENT = "com.google.genai.types.Content";
    private static final String PART = "com.google.genai.types.Part";
    private static final String SPAN = "io.opentelemetry.api.trace.Span";

    private static final String INVOCATION_ID = "invocation-compat";
    private static final String SESSION_ID = "session-compat";
    private static final String COMPAT_INPUT_TEXT = "compat-input";
    private static final String COMPAT_OUTPUT_TEXT = "compat-output";

    @Test
    void traceCallLlmEnrichesCallLlmSpans() throws Exception {
        int expectedArity = Integer.parseInt(System.getProperty("adk.expected.arity", "0"));
        if (expectedArity != 4 && expectedArity != 5 && expectedArity != 6) {
            throw new IllegalStateException("adk.expected.arity must be 4, 5 or 6 but was " + expectedArity);
        }

        // Initialize test infrastructure before installing the agent, while leaving ADK's Tracing
        // and BaseTool cold-loading path unpreloaded.
        Mockito.framework();
        Instrumentation instrumentation = ByteBuddyAgent.install();
        TelemetryAgent.install(instrumentation);

        Class<?> tracingClass = Class.forName(TRACING); // fails the test if ADK moved or renamed it
        Class<?> spanClass = Class.forName(SPAN);
        Class<?> invocationContextClass = Class.forName(INVOCATION_CONTEXT);
        Class<?> llmRequestClass = Class.forName(LLM_REQUEST);
        Class<?> llmResponseClass = Class.forName(LLM_RESPONSE);

        Class<?>[] parameters =
                expectedParameters(expectedArity, spanClass, invocationContextClass, llmRequestClass, llmResponseClass);
        Method traceCallLlm;
        try {
            traceCallLlm = tracingClass.getDeclaredMethod("traceCallLlm", parameters);
        } catch (NoSuchMethodException e) {
            throw new AssertionError(
                    "ADK does not declare the expected " + expectedArity + "-argument traceCallLlm", e);
        }
        assertThat(java.lang.reflect.Modifier.isStatic(traceCallLlm.getModifiers()))
                .isTrue();

        Object requestContent = textContent(COMPAT_INPUT_TEXT, "user");
        Object responseContent = textContent(COMPAT_OUTPUT_TEXT, "model");
        Object llmRequest = buildLlmRequest(llmRequestClass, requestContent);
        Object llmResponse = buildLlmResponse(llmResponseClass, responseContent);
        Object invocationContext = mockInvocationContext(invocationContextClass);

        InMemorySpanExporter exporter = InMemorySpanExporter.create();
        SdkTracerProvider provider = SdkTracerProvider.builder()
                .addSpanProcessor(SimpleSpanProcessor.create(exporter))
                .build();
        Tracer tracer = provider.get("adk-tracing-compat");
        Span span = tracer.spanBuilder("call_llm").startSpan();

        Scope scope = null;
        Exception failure = expectedArity == 6 ? new RuntimeException("compat-error") : null;
        try {
            Object[] args;
            if (expectedArity == 4) {
                // The four-argument form uses Span.current().
                scope = span.makeCurrent();
                args = new Object[] {invocationContext, "event-compat", llmRequest, llmResponse};
            } else if (expectedArity == 5) {
                args = new Object[] {span, invocationContext, "event-compat", llmRequest, llmResponse};
            } else {
                args = new Object[] {span, invocationContext, "event-compat", llmRequest, llmResponse, failure};
            }
            traceCallLlm.invoke(null, args);
        } finally {
            if (scope != null) {
                scope.close();
            }
            span.end();
        }
        provider.forceFlush();

        List<SpanData> finished = exporter.getFinishedSpanItems();
        SpanData exported = finished.stream()
                .filter(item -> "call_llm".equals(item.getName()))
                .findFirst()
                .orElseThrow(() -> new AssertionError("call_llm span was not exported; finished spans: " + finished));
        provider.shutdown();

        Attributes attributes = exported.getAttributes();
        // ADK adds version-specific native attributes, so only the OpenInference results are checked.
        assertThat(attributes.get(AttributeKey.stringKey(OPENINFERENCE_SPAN_KIND)))
                .isEqualTo("LLM");
        assertThat(attributes.get(AttributeKey.stringKey(LLM_PROVIDER))).isEqualTo("google");
        assertThat(attributes.get(AttributeKey.stringKey(LLM_MODEL_NAME))).isEqualTo("compat-model");
        assertThat(attributes.get(AttributeKey.stringKey(INPUT_MIME_TYPE))).isEqualTo("application/json");
        assertThat(attributes.get(AttributeKey.stringKey(INPUT_VALUE))).contains(COMPAT_INPUT_TEXT);
        assertThat(attributes.get(AttributeKey.stringKey(LLM_INPUT_MESSAGES + ".0." + MESSAGE_ROLE)))
                .isEqualTo("user");
        String inputTextKey = LLM_INPUT_MESSAGES + ".0." + MESSAGE_CONTENTS + ".0." + MESSAGE_CONTENT_TEXT;
        assertThat(attributes.get(AttributeKey.stringKey(inputTextKey))).isEqualTo(COMPAT_INPUT_TEXT);
        assertThat(attributes.get(AttributeKey.stringKey(OUTPUT_MIME_TYPE))).isEqualTo("application/json");
        assertThat(attributes.get(AttributeKey.stringKey(OUTPUT_VALUE))).contains(COMPAT_OUTPUT_TEXT);
        assertThat(attributes.get(AttributeKey.stringKey(LLM_OUTPUT_MESSAGES + ".0." + MESSAGE_ROLE)))
                .isEqualTo("model");
        String outputTextKey = LLM_OUTPUT_MESSAGES + ".0." + MESSAGE_CONTENTS + ".0." + MESSAGE_CONTENT_TEXT;
        assertThat(attributes.get(AttributeKey.stringKey(outputTextKey))).isEqualTo(COMPAT_OUTPUT_TEXT);

        if (expectedArity == 6) {
            assertThat(exported.getStatus().getStatusCode()).isEqualTo(StatusCode.ERROR);
            assertThat(exported.getEvents())
                    .anySatisfy(event -> assertThat(event.getName()).isEqualTo("exception"));
        }
    }

    private static Class<?>[] expectedParameters(
            int arity, Class<?> span, Class<?> invocationContext, Class<?> llmRequest, Class<?> llmResponse) {
        switch (arity) {
            case 4:
                return new Class<?>[] {invocationContext, String.class, llmRequest, llmResponse};
            case 5:
                return new Class<?>[] {span, invocationContext, String.class, llmRequest, llmResponse};
            default:
                return new Class<?>[] {span, invocationContext, String.class, llmRequest, llmResponse, Exception.class};
        }
    }

    /** A real {@code com.google.genai.types.Content} holding a single text part with the given role. */
    private static Object textContent(String text, String role) throws Exception {
        Class<?> partClass = Class.forName(PART);
        Object part = invoke(partClass.getMethod("fromText", String.class), null, text);
        Class<?> contentClass = Class.forName(CONTENT);
        Object builder = invoke(contentClass.getMethod("builder"), null);
        setBuilderProperty(builder, "parts", List.class, List.of(part));
        setBuilderProperty(builder, "role", String.class, role);
        return invoke(builder.getClass().getMethod("build"), builder);
    }

    private static Object buildLlmRequest(Class<?> llmRequestClass, Object content) throws Exception {
        Object builder = invoke(llmRequestClass.getMethod("builder"), null);
        setBuilderProperty(builder, "model", String.class, "compat-model");
        setBuilderProperty(builder, "contents", List.class, List.of(content));
        setBuilderProperty(builder, "liveConnectConfig", Class.forName(LIVE_CONNECT_CONFIG), liveConnectConfig());
        // BaseTool loads when writeLlmCall processes the request tools, not via ADK serialization.
        // tools(Map) is public in ADK 1.0+ but package-private in 0.6/0.9, so it is set reflectively.
        setBuilderProperty(builder, "tools", Map.class, Map.of());
        return invoke(builder.getClass().getMethod("build"), builder);
    }

    private static Object buildLlmResponse(Class<?> llmResponseClass, Object content) throws Exception {
        Object builder = invoke(llmResponseClass.getMethod("builder"), null);
        setBuilderProperty(builder, "content", Class.forName(CONTENT), content);
        return invoke(builder.getClass().getMethod("build"), builder);
    }

    private static Object liveConnectConfig() throws Exception {
        Class<?> liveConnectConfigClass = Class.forName(LIVE_CONNECT_CONFIG);
        Object builder = invoke(liveConnectConfigClass.getMethod("builder"), null);
        return invoke(builder.getClass().getMethod("build"), builder);
    }

    /** Locates a one-argument builder setter by name, including package-private ones, and calls it. */
    private static void setBuilderProperty(Object builder, String property, Class<?> valueType, Object value)
            throws Exception {
        for (Class<?> type = builder.getClass(); type != null; type = type.getSuperclass()) {
            for (Method method : type.getDeclaredMethods()) {
                if (!method.getName().equals(property) || method.getParameterCount() != 1) {
                    continue;
                }
                if (!method.getParameterTypes()[0].isAssignableFrom(valueType)) {
                    continue;
                }
                invoke(method, builder, value);
                return;
            }
        }
        throw new IllegalStateException(
                "no builder setter " + property + " accepting " + valueType + " on " + builder.getClass());
    }

    /**
     * Mockito mock of {@code InvocationContext} with a version-independent answer: {@code
     * invocationId()} and {@code Session.id()} return non-empty strings and everything else falls
     * back to Mockito's defaults. No ADK type is imported or compiled against.
     */
    private static Object mockInvocationContext(Class<?> invocationContextClass) throws Exception {
        Class<?> sessionClass = Class.forName(SESSION);
        Object session = Mockito.mock(sessionClass, (Answer<Object>) invocation -> {
            if ("id".equals(invocation.getMethod().getName())) {
                return SESSION_ID;
            }
            return Mockito.RETURNS_DEFAULTS.answer(invocation);
        });
        return Mockito.mock(invocationContextClass, (Answer<Object>) invocation -> {
            switch (invocation.getMethod().getName()) {
                case "invocationId":
                    return INVOCATION_ID;
                case "session":
                    return session;
                default:
                    return Mockito.RETURNS_DEFAULTS.answer(invocation);
            }
        });
    }

    /** Invokes a method that may be declared on a package-private class, so accessibility is forced. */
    private static Object invoke(Method method, Object target, Object... args) throws Exception {
        method.setAccessible(true);
        return method.invoke(target, args);
    }
}

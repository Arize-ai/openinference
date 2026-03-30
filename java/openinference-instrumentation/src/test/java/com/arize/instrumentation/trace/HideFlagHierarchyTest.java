package com.arize.instrumentation.trace;

import static org.assertj.core.api.Assertions.assertThat;

import com.arize.instrumentation.OITracer;
import com.arize.instrumentation.TraceConfig;
import com.arize.semconv.trace.SemanticConventions;
import io.opentelemetry.api.common.AttributeKey;
import io.opentelemetry.sdk.testing.exporter.InMemorySpanExporter;
import io.opentelemetry.sdk.trace.SdkTracerProvider;
import io.opentelemetry.sdk.trace.data.SpanData;
import io.opentelemetry.sdk.trace.export.SimpleSpanProcessor;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class HideFlagHierarchyTest {

    private InMemorySpanExporter exporter;

    @BeforeEach
    void setUp() {
        exporter = InMemorySpanExporter.create();
    }

    private OITracer createTracer(TraceConfig config) {
        SdkTracerProvider provider = SdkTracerProvider.builder()
                .addSpanProcessor(SimpleSpanProcessor.create(exporter))
                .build();
        return new OITracer(provider.get("test"), config);
    }

    // ---- Test 1: hideInputs suppresses input.value, input.mime_type, llm.input_messages.*, llm.prompts ----

    @Test
    void hideInputsSuppressesAllInputAttributes() {
        OITracer tracer = createTracer(TraceConfig.builder().hideInputs(true).build());

        try (LLMSpan span = LLMSpan.start(tracer, "test")) {
            span.setAttribute(SemanticConventions.INPUT_VALUE, "secret input");
            span.setAttribute(SemanticConventions.INPUT_MIME_TYPE, "text/plain");
            span.setAttribute(SemanticConventions.LLM_INPUT_MESSAGES, "[{\"role\":\"user\"}]");
            span.setAttribute("llm.input_messages.0.message.role", "user");
            span.setAttribute("llm.input_messages.0.message.content", "secret");
            span.setAttribute(SemanticConventions.LLM_PROMPTS, "prompt text");
        }

        SpanData data = exporter.getFinishedSpanItems().get(0);
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.INPUT_VALUE))).isNull();
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.INPUT_MIME_TYPE))).isNull();
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.LLM_INPUT_MESSAGES))).isNull();
        assertThat(data.getAttributes().get(AttributeKey.stringKey("llm.input_messages.0.message.role"))).isNull();
        assertThat(data.getAttributes().get(AttributeKey.stringKey("llm.input_messages.0.message.content"))).isNull();
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.LLM_PROMPTS))).isNull();
    }

    // ---- Test 2: hideOutputs suppresses output.value, output.mime_type, llm.output_messages.* ----

    @Test
    void hideOutputsSuppressesAllOutputAttributes() {
        OITracer tracer = createTracer(TraceConfig.builder().hideOutputs(true).build());

        try (LLMSpan span = LLMSpan.start(tracer, "test")) {
            span.setAttribute(SemanticConventions.OUTPUT_VALUE, "secret output");
            span.setAttribute(SemanticConventions.OUTPUT_MIME_TYPE, "text/plain");
            span.setAttribute(SemanticConventions.LLM_OUTPUT_MESSAGES, "[{\"role\":\"assistant\"}]");
            span.setAttribute("llm.output_messages.0.message.role", "assistant");
            span.setAttribute("llm.output_messages.0.message.content", "secret");
        }

        SpanData data = exporter.getFinishedSpanItems().get(0);
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.OUTPUT_VALUE))).isNull();
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.OUTPUT_MIME_TYPE))).isNull();
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.LLM_OUTPUT_MESSAGES))).isNull();
        assertThat(data.getAttributes().get(AttributeKey.stringKey("llm.output_messages.0.message.role"))).isNull();
        assertThat(data.getAttributes().get(AttributeKey.stringKey("llm.output_messages.0.message.content"))).isNull();
    }

    // ---- Test 3: hideInputMessages suppresses only llm.input_messages.* but NOT input.value ----

    @Test
    void hideInputMessagesSuppressesOnlyMessages() {
        OITracer tracer = createTracer(TraceConfig.builder().hideInputMessages(true).build());

        try (LLMSpan span = LLMSpan.start(tracer, "test")) {
            span.setAttribute(SemanticConventions.INPUT_VALUE, "visible input");
            span.setAttribute(SemanticConventions.LLM_INPUT_MESSAGES, "[{\"role\":\"user\"}]");
            span.setAttribute("llm.input_messages.0.message.role", "user");
        }

        SpanData data = exporter.getFinishedSpanItems().get(0);
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.INPUT_VALUE)))
                .isEqualTo("visible input");
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.LLM_INPUT_MESSAGES))).isNull();
        assertThat(data.getAttributes().get(AttributeKey.stringKey("llm.input_messages.0.message.role"))).isNull();
    }

    // ---- Test 4: hideOutputMessages suppresses only llm.output_messages.* but NOT output.value ----

    @Test
    void hideOutputMessagesSuppressesOnlyMessages() {
        OITracer tracer = createTracer(TraceConfig.builder().hideOutputMessages(true).build());

        try (LLMSpan span = LLMSpan.start(tracer, "test")) {
            span.setAttribute(SemanticConventions.OUTPUT_VALUE, "visible output");
            span.setAttribute(SemanticConventions.LLM_OUTPUT_MESSAGES, "[{\"role\":\"assistant\"}]");
            span.setAttribute("llm.output_messages.0.message.role", "assistant");
        }

        SpanData data = exporter.getFinishedSpanItems().get(0);
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.OUTPUT_VALUE)))
                .isEqualTo("visible output");
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.LLM_OUTPUT_MESSAGES))).isNull();
        assertThat(data.getAttributes().get(AttributeKey.stringKey("llm.output_messages.0.message.role"))).isNull();
    }

    // ---- Test 5: hideInputs also suppresses LLMSpan.setInputMessages() ----

    @Test
    void hideInputsSuppressesSetInputMessages() {
        OITracer tracer = createTracer(TraceConfig.builder().hideInputs(true).build());

        try (LLMSpan span = LLMSpan.start(tracer, "test")) {
            span.setInputMessages(List.of(Map.of("role", "user", "content", "secret")));
        }

        SpanData data = exporter.getFinishedSpanItems().get(0);
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.LLM_INPUT_MESSAGES))).isNull();
    }

    // ---- Test 6: hideOutputs also suppresses LLMSpan.setOutputMessages() ----

    @Test
    void hideOutputsSuppressesSetOutputMessages() {
        OITracer tracer = createTracer(TraceConfig.builder().hideOutputs(true).build());

        try (LLMSpan span = LLMSpan.start(tracer, "test")) {
            span.setOutputMessages(List.of(Map.of("role", "assistant", "content", "secret")));
        }

        SpanData data = exporter.getFinishedSpanItems().get(0);
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.LLM_OUTPUT_MESSAGES))).isNull();
    }

    // ---- Test 7: hideInputText suppresses message.content but NOT message.contents ----

    @Test
    void hideInputTextSuppressesContentButNotContents() {
        OITracer tracer = createTracer(TraceConfig.builder().hideInputText(true).build());

        try (LLMSpan span = LLMSpan.start(tracer, "test")) {
            span.setAttribute("llm.input_messages.0.message.content", "secret text");
            span.setAttribute("llm.input_messages.0.message.contents.0.message_content.text", "visible text");
            span.setAttribute("llm.input_messages.0.message.role", "user");
        }

        SpanData data = exporter.getFinishedSpanItems().get(0);
        assertThat(data.getAttributes().get(AttributeKey.stringKey("llm.input_messages.0.message.content")))
                .isNull();
        assertThat(data.getAttributes().get(
                        AttributeKey.stringKey("llm.input_messages.0.message.contents.0.message_content.text")))
                .isEqualTo("visible text");
        assertThat(data.getAttributes().get(AttributeKey.stringKey("llm.input_messages.0.message.role")))
                .isEqualTo("user");
    }

    // ---- Test 8: hidePromptTemplate, hidePromptTemplateVariables, hidePromptTemplateVersion ----

    @Test
    void hidePromptTemplateSuppressesTemplate() {
        OITracer tracer = createTracer(TraceConfig.builder().hidePromptTemplate(true).build());

        try (LLMSpan span = LLMSpan.start(tracer, "test")) {
            span.setAttribute(SemanticConventions.PROMPT_TEMPLATE_TEMPLATE, "Hello {name}");
            span.setAttribute(SemanticConventions.PROMPT_TEMPLATE_VARIABLES, "{\"name\":\"world\"}");
            span.setAttribute(SemanticConventions.PROMPT_TEMPLATE_VERSION, "v1");
        }

        SpanData data = exporter.getFinishedSpanItems().get(0);
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.PROMPT_TEMPLATE_TEMPLATE)))
                .isNull();
        // variables and version should still be present
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.PROMPT_TEMPLATE_VARIABLES)))
                .isEqualTo("{\"name\":\"world\"}");
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.PROMPT_TEMPLATE_VERSION)))
                .isEqualTo("v1");
    }

    @Test
    void hidePromptTemplateVariablesSuppressesVariables() {
        OITracer tracer = createTracer(TraceConfig.builder().hidePromptTemplateVariables(true).build());

        try (LLMSpan span = LLMSpan.start(tracer, "test")) {
            span.setAttribute(SemanticConventions.PROMPT_TEMPLATE_TEMPLATE, "Hello {name}");
            span.setAttribute(SemanticConventions.PROMPT_TEMPLATE_VARIABLES, "{\"name\":\"world\"}");
        }

        SpanData data = exporter.getFinishedSpanItems().get(0);
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.PROMPT_TEMPLATE_TEMPLATE)))
                .isEqualTo("Hello {name}");
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.PROMPT_TEMPLATE_VARIABLES)))
                .isNull();
    }

    @Test
    void hidePromptTemplateVersionSuppressesVersion() {
        OITracer tracer = createTracer(TraceConfig.builder().hidePromptTemplateVersion(true).build());

        try (LLMSpan span = LLMSpan.start(tracer, "test")) {
            span.setAttribute(SemanticConventions.PROMPT_TEMPLATE_VERSION, "v1");
            span.setAttribute(SemanticConventions.PROMPT_TEMPLATE_TEMPLATE, "Hello {name}");
        }

        SpanData data = exporter.getFinishedSpanItems().get(0);
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.PROMPT_TEMPLATE_VERSION)))
                .isNull();
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.PROMPT_TEMPLATE_TEMPLATE)))
                .isEqualTo("Hello {name}");
    }

    // ---- Test 9: hideToolParameters suppresses tool.parameters ----

    @Test
    void hideToolParametersSuppressesToolParameters() {
        OITracer tracer = createTracer(TraceConfig.builder().hideToolParameters(true).build());

        try (LLMSpan span = LLMSpan.start(tracer, "test")) {
            span.setAttribute(SemanticConventions.TOOL_PARAMETERS, "{\"param\":\"value\"}");
            span.setAttribute(SemanticConventions.TOOL_NAME, "my_tool");
        }

        SpanData data = exporter.getFinishedSpanItems().get(0);
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.TOOL_PARAMETERS))).isNull();
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.TOOL_NAME)))
                .isEqualTo("my_tool");
    }

    // ---- Test 10: Default config (all flags false) does not suppress anything ----

    @Test
    void defaultConfigDoesNotSuppressAnything() {
        OITracer tracer = createTracer(TraceConfig.getDefault());

        try (LLMSpan span = LLMSpan.start(tracer, "test")) {
            span.setAttribute(SemanticConventions.INPUT_VALUE, "input");
            span.setAttribute(SemanticConventions.INPUT_MIME_TYPE, "text/plain");
            span.setAttribute(SemanticConventions.OUTPUT_VALUE, "output");
            span.setAttribute(SemanticConventions.OUTPUT_MIME_TYPE, "text/plain");
            span.setAttribute(SemanticConventions.LLM_INPUT_MESSAGES, "[{\"role\":\"user\"}]");
            span.setAttribute(SemanticConventions.LLM_OUTPUT_MESSAGES, "[{\"role\":\"assistant\"}]");
            span.setAttribute("llm.input_messages.0.message.content", "text");
            span.setAttribute("llm.output_messages.0.message.content", "text");
            span.setAttribute(SemanticConventions.LLM_PROMPTS, "prompt");
            span.setAttribute(SemanticConventions.PROMPT_TEMPLATE_TEMPLATE, "template");
            span.setAttribute(SemanticConventions.PROMPT_TEMPLATE_VARIABLES, "vars");
            span.setAttribute(SemanticConventions.PROMPT_TEMPLATE_VERSION, "v1");
            span.setAttribute(SemanticConventions.TOOL_PARAMETERS, "params");
        }

        SpanData data = exporter.getFinishedSpanItems().get(0);
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.INPUT_VALUE)))
                .isEqualTo("input");
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.INPUT_MIME_TYPE)))
                .isEqualTo("text/plain");
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.OUTPUT_VALUE)))
                .isEqualTo("output");
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.OUTPUT_MIME_TYPE)))
                .isEqualTo("text/plain");
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.LLM_INPUT_MESSAGES)))
                .isNotNull();
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.LLM_OUTPUT_MESSAGES)))
                .isNotNull();
        assertThat(data.getAttributes().get(AttributeKey.stringKey("llm.input_messages.0.message.content")))
                .isEqualTo("text");
        assertThat(data.getAttributes().get(AttributeKey.stringKey("llm.output_messages.0.message.content")))
                .isEqualTo("text");
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.LLM_PROMPTS)))
                .isEqualTo("prompt");
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.PROMPT_TEMPLATE_TEMPLATE)))
                .isEqualTo("template");
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.PROMPT_TEMPLATE_VARIABLES)))
                .isEqualTo("vars");
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.PROMPT_TEMPLATE_VERSION)))
                .isEqualTo("v1");
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.TOOL_PARAMETERS)))
                .isEqualTo("params");
    }

    // ---- Additional: hideOutputText, hideInputImages, hideOutputImages ----

    @Test
    void hideOutputTextSuppressesContentButNotContents() {
        OITracer tracer = createTracer(TraceConfig.builder().hideOutputText(true).build());

        try (LLMSpan span = LLMSpan.start(tracer, "test")) {
            span.setAttribute("llm.output_messages.0.message.content", "secret text");
            span.setAttribute("llm.output_messages.0.message.contents.0.message_content.text", "visible text");
        }

        SpanData data = exporter.getFinishedSpanItems().get(0);
        assertThat(data.getAttributes().get(AttributeKey.stringKey("llm.output_messages.0.message.content")))
                .isNull();
        assertThat(data.getAttributes().get(
                        AttributeKey.stringKey("llm.output_messages.0.message.contents.0.message_content.text")))
                .isEqualTo("visible text");
    }

    @Test
    void hideInputImagesSuppressesImageWithinInputMessages() {
        OITracer tracer = createTracer(TraceConfig.builder().hideInputImages(true).build());

        try (LLMSpan span = LLMSpan.start(tracer, "test")) {
            span.setAttribute(
                    "llm.input_messages.0.message.contents.0.message_content.image.url", "http://example.com/img.png");
            span.setAttribute("llm.input_messages.0.message.content", "visible text");
        }

        SpanData data = exporter.getFinishedSpanItems().get(0);
        assertThat(data.getAttributes()
                        .get(AttributeKey.stringKey(
                                "llm.input_messages.0.message.contents.0.message_content.image.url")))
                .isNull();
        assertThat(data.getAttributes().get(AttributeKey.stringKey("llm.input_messages.0.message.content")))
                .isEqualTo("visible text");
    }

    @Test
    void hideOutputImagesSuppressesImageWithinOutputMessages() {
        OITracer tracer = createTracer(TraceConfig.builder().hideOutputImages(true).build());

        try (LLMSpan span = LLMSpan.start(tracer, "test")) {
            span.setAttribute(
                    "llm.output_messages.0.message.contents.0.message_content.image.url",
                    "http://example.com/img.png");
            span.setAttribute("llm.output_messages.0.message.content", "visible text");
        }

        SpanData data = exporter.getFinishedSpanItems().get(0);
        assertThat(data.getAttributes()
                        .get(AttributeKey.stringKey(
                                "llm.output_messages.0.message.contents.0.message_content.image.url")))
                .isNull();
        assertThat(data.getAttributes().get(AttributeKey.stringKey("llm.output_messages.0.message.content")))
                .isEqualTo("visible text");
    }

    @Test
    void hideInputAudioSuppressesAudioWithinInputMessages() {
        OITracer tracer = createTracer(TraceConfig.builder().hideInputAudio(true).build());

        try (LLMSpan span = LLMSpan.start(tracer, "test")) {
            span.setAttribute("llm.input_messages.0.message.contents.0.audio.url", "http://example.com/audio.wav");
            span.setAttribute("llm.input_messages.0.message.content", "visible text");
        }

        SpanData data = exporter.getFinishedSpanItems().get(0);
        assertThat(data.getAttributes()
                        .get(AttributeKey.stringKey("llm.input_messages.0.message.contents.0.audio.url")))
                .isNull();
        assertThat(data.getAttributes().get(AttributeKey.stringKey("llm.input_messages.0.message.content")))
                .isEqualTo("visible text");
    }

    @Test
    void hideOutputAudioSuppressesAudioWithinOutputMessages() {
        OITracer tracer = createTracer(TraceConfig.builder().hideOutputAudio(true).build());

        try (LLMSpan span = LLMSpan.start(tracer, "test")) {
            span.setAttribute("llm.output_messages.0.message.contents.0.audio.url", "http://example.com/audio.wav");
            span.setAttribute("llm.output_messages.0.message.content", "visible text");
        }

        SpanData data = exporter.getFinishedSpanItems().get(0);
        assertThat(data.getAttributes()
                        .get(AttributeKey.stringKey("llm.output_messages.0.message.contents.0.audio.url")))
                .isNull();
        assertThat(data.getAttributes().get(AttributeKey.stringKey("llm.output_messages.0.message.content")))
                .isEqualTo("visible text");
    }

    @Test
    void hideInputEmbeddingsSuppressesEmbeddingVectors() {
        OITracer tracer = createTracer(TraceConfig.builder().hideInputEmbeddings(true).build());

        try (LLMSpan span = LLMSpan.start(tracer, "test")) {
            span.setAttribute("embedding.embeddings.0.embedding.vector", "[0.1, 0.2, 0.3]");
            span.setAttribute("embedding.embeddings.0.embedding.text", "visible text");
        }

        SpanData data = exporter.getFinishedSpanItems().get(0);
        assertThat(data.getAttributes()
                        .get(AttributeKey.stringKey("embedding.embeddings.0.embedding.vector")))
                .isNull();
        assertThat(data.getAttributes()
                        .get(AttributeKey.stringKey("embedding.embeddings.0.embedding.text")))
                .isEqualTo("visible text");
    }

    // ---- Orthogonality: hideInputs does NOT suppress output attributes ----

    @Test
    void hideInputsDoesNotSuppressOutputAttributes() {
        OITracer tracer = createTracer(TraceConfig.builder().hideInputs(true).build());

        try (LLMSpan span = LLMSpan.start(tracer, "test")) {
            span.setAttribute(SemanticConventions.OUTPUT_VALUE, "visible output");
            span.setAttribute(SemanticConventions.OUTPUT_MIME_TYPE, "text/plain");
            span.setAttribute(SemanticConventions.LLM_OUTPUT_MESSAGES, "[{\"role\":\"assistant\"}]");
            span.setAttribute("llm.output_messages.0.message.content", "visible");
        }

        SpanData data = exporter.getFinishedSpanItems().get(0);
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.OUTPUT_VALUE)))
                .isEqualTo("visible output");
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.OUTPUT_MIME_TYPE)))
                .isEqualTo("text/plain");
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.LLM_OUTPUT_MESSAGES)))
                .isNotNull();
        assertThat(data.getAttributes().get(AttributeKey.stringKey("llm.output_messages.0.message.content")))
                .isEqualTo("visible");
    }

    // ---- Orthogonality: hideOutputs does NOT suppress input attributes ----

    @Test
    void hideOutputsDoesNotSuppressInputAttributes() {
        OITracer tracer = createTracer(TraceConfig.builder().hideOutputs(true).build());

        try (LLMSpan span = LLMSpan.start(tracer, "test")) {
            span.setAttribute(SemanticConventions.INPUT_VALUE, "visible input");
            span.setAttribute(SemanticConventions.INPUT_MIME_TYPE, "text/plain");
            span.setAttribute(SemanticConventions.LLM_INPUT_MESSAGES, "[{\"role\":\"user\"}]");
            span.setAttribute("llm.input_messages.0.message.content", "visible");
        }

        SpanData data = exporter.getFinishedSpanItems().get(0);
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.INPUT_VALUE)))
                .isEqualTo("visible input");
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.INPUT_MIME_TYPE)))
                .isEqualTo("text/plain");
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.LLM_INPUT_MESSAGES)))
                .isNotNull();
        assertThat(data.getAttributes().get(AttributeKey.stringKey("llm.input_messages.0.message.content")))
                .isEqualTo("visible");
    }

    // ---- Combined flags: hideInputs + hideOutputs together ----

    @Test
    void hideInputsAndOutputsCombinedSuppressesBoth() {
        OITracer tracer = createTracer(
                TraceConfig.builder().hideInputs(true).hideOutputs(true).build());

        try (LLMSpan span = LLMSpan.start(tracer, "test")) {
            span.setAttribute(SemanticConventions.INPUT_VALUE, "secret input");
            span.setAttribute(SemanticConventions.OUTPUT_VALUE, "secret output");
            span.setAttribute(SemanticConventions.LLM_INPUT_MESSAGES, "[{\"role\":\"user\"}]");
            span.setAttribute(SemanticConventions.LLM_OUTPUT_MESSAGES, "[{\"role\":\"assistant\"}]");
            span.setAttribute(SemanticConventions.LLM_MODEL_NAME, "gpt-4o");
        }

        SpanData data = exporter.getFinishedSpanItems().get(0);
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.INPUT_VALUE))).isNull();
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.OUTPUT_VALUE))).isNull();
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.LLM_INPUT_MESSAGES))).isNull();
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.LLM_OUTPUT_MESSAGES))).isNull();
        // Non-input/output attributes should still be present
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.LLM_MODEL_NAME)))
                .isEqualTo("gpt-4o");
    }

    // ---- Null/empty key safety ----

    @Test
    void setAttributeIgnoresNullAndEmptyKeysAndNullValues() {
        OITracer tracer = createTracer(TraceConfig.getDefault());

        try (LLMSpan span = LLMSpan.start(tracer, "test")) {
            // These should not throw or set any attribute
            span.setAttribute(null, "value");
            span.setAttribute("", "value");
            span.setAttribute("valid.key", null);
            span.setAttribute(SemanticConventions.INPUT_VALUE, "present");
        }

        SpanData data = exporter.getFinishedSpanItems().get(0);
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.INPUT_VALUE)))
                .isEqualTo("present");
    }

    // ---- hideInputText does NOT suppress non-text message attributes ----

    @Test
    void hideInputTextDoesNotSuppressRoleOrImages() {
        OITracer tracer = createTracer(TraceConfig.builder().hideInputText(true).build());

        try (LLMSpan span = LLMSpan.start(tracer, "test")) {
            span.setAttribute("llm.input_messages.0.message.role", "user");
            span.setAttribute(
                    "llm.input_messages.0.message.contents.0.message_content.image.url", "http://example.com/img.png");
            span.setAttribute("llm.input_messages.0.message.content", "secret text");
        }

        SpanData data = exporter.getFinishedSpanItems().get(0);
        // Role and image should be visible
        assertThat(data.getAttributes().get(AttributeKey.stringKey("llm.input_messages.0.message.role")))
                .isEqualTo("user");
        assertThat(data.getAttributes()
                        .get(AttributeKey.stringKey(
                                "llm.input_messages.0.message.contents.0.message_content.image.url")))
                .isEqualTo("http://example.com/img.png");
        // Text content should be hidden
        assertThat(data.getAttributes().get(AttributeKey.stringKey("llm.input_messages.0.message.content")))
                .isNull();
    }

    // ---- hideInputMessages also suppresses via LLMSpan.setInputMessages() ----

    @Test
    void hideInputMessagesSuppressesSetInputMessages() {
        OITracer tracer = createTracer(TraceConfig.builder().hideInputMessages(true).build());

        try (LLMSpan span = LLMSpan.start(tracer, "test")) {
            span.setInputMessages(List.of(Map.of("role", "user", "content", "secret")));
        }

        SpanData data = exporter.getFinishedSpanItems().get(0);
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.LLM_INPUT_MESSAGES))).isNull();
    }

    // ---- hideOutputMessages also suppresses via LLMSpan.setOutputMessages() ----

    @Test
    void hideOutputMessagesSuppressesSetOutputMessages() {
        OITracer tracer = createTracer(TraceConfig.builder().hideOutputMessages(true).build());

        try (LLMSpan span = LLMSpan.start(tracer, "test")) {
            span.setOutputMessages(List.of(Map.of("role", "assistant", "content", "secret")));
        }

        SpanData data = exporter.getFinishedSpanItems().get(0);
        assertThat(data.getAttributes().get(AttributeKey.stringKey(SemanticConventions.LLM_OUTPUT_MESSAGES))).isNull();
    }
}

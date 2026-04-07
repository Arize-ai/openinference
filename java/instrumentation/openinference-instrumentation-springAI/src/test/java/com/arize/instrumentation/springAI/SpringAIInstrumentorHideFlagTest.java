package com.arize.instrumentation.springAI;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.*;

import com.arize.instrumentation.OITracer;
import com.arize.instrumentation.TraceConfig;
import com.arize.semconv.trace.SemanticConventions;
import io.opentelemetry.api.common.AttributeKey;
import io.opentelemetry.sdk.testing.exporter.InMemorySpanExporter;
import io.opentelemetry.sdk.trace.SdkTracerProvider;
import io.opentelemetry.sdk.trace.data.SpanData;
import io.opentelemetry.sdk.trace.export.SimpleSpanProcessor;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.Generation;
import org.springframework.ai.chat.observation.ChatModelObservationContext;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.ai.chat.prompt.Prompt;

/**
 * Tests for the hide flag hierarchy in SpringAIInstrumentor.
 *
 * The reference implementation defines:
 * - hideInputs (broad): suppresses input.value, input.mime_type, AND llm.input_messages.*
 * - hideInputMessages (narrow): suppresses only llm.input_messages.*
 * - hideOutputs (broad): suppresses output.value, output.mime_type, AND llm.output_messages.*
 * - hideOutputMessages (narrow): suppresses only llm.output_messages.*
 */
class SpringAIInstrumentorHideFlagTest {

    private InMemorySpanExporter spanExporter;
    private SdkTracerProvider tracerProvider;

    @BeforeEach
    void setUp() {
        spanExporter = InMemorySpanExporter.create();
        tracerProvider = SdkTracerProvider.builder()
                .addSpanProcessor(SimpleSpanProcessor.create(spanExporter))
                .build();
    }

    @AfterEach
    void tearDown() {
        spanExporter.reset();
        tracerProvider.close();
    }

    // ── Helper methods ──────────────────────────────────────────────────

    private SpringAIInstrumentor createInstrumentor(TraceConfig config) {
        OITracer tracer = new OITracer(tracerProvider.get("test"), config);
        return new SpringAIInstrumentor(tracer);
    }

    private ChatModelObservationContext createMockContext(String userInput, String assistantOutput) {
        // Create real messages
        List<Message> inputMessages = new ArrayList<>();
        inputMessages.add(new UserMessage(userInput));

        // Mock Prompt
        Prompt prompt = mock(Prompt.class);
        when(prompt.getInstructions()).thenReturn(inputMessages);
        ChatOptions options = mock(ChatOptions.class);
        when(options.getModel()).thenReturn("gpt-4");
        when(prompt.getOptions()).thenReturn(options);

        // Mock ChatModelObservationContext
        ChatModelObservationContext context = mock(ChatModelObservationContext.class);
        when(context.getRequest()).thenReturn(prompt);

        // Set up response if assistantOutput is provided
        if (assistantOutput != null) {
            AssistantMessage assistantMessage = new AssistantMessage(assistantOutput);
            Generation generation = new Generation(assistantMessage);
            ChatResponse response = mock(ChatResponse.class);
            when(response.getResults()).thenReturn(List.of(generation));
            when(context.getResponse()).thenReturn(response);
        }

        return context;
    }

    private void simulateFullCall(SpringAIInstrumentor instrumentor, ChatModelObservationContext context) {
        instrumentor.onStart(context);
        instrumentor.onStop(context);
    }

    private SpanData getSpan() {
        List<SpanData> spans = spanExporter.getFinishedSpanItems();
        assertThat(spans).hasSize(1);
        return spans.get(0);
    }

    // ── Default config: everything visible ──────────────────────────────

    @Test
    void defaultConfig_allAttributesPresent() {
        SpringAIInstrumentor instrumentor = createInstrumentor(TraceConfig.getDefault());
        ChatModelObservationContext context = createMockContext("Hello world", "Hi there");

        simulateFullCall(instrumentor, context);
        SpanData span = getSpan();

        // Input messages should be present
        assertThat(span.getAttributes().get(AttributeKey.stringKey("llm.input_messages.0.message.role")))
                .isEqualTo("user");
        assertThat(span.getAttributes().get(AttributeKey.stringKey("llm.input_messages.0.message.content")))
                .isEqualTo("Hello world");

        // Input value should be present
        assertThat(span.getAttributes().get(AttributeKey.stringKey(SemanticConventions.INPUT_VALUE)))
                .isNotNull();
        assertThat(span.getAttributes().get(AttributeKey.stringKey(SemanticConventions.INPUT_MIME_TYPE)))
                .isEqualTo("application/json");

        // Output messages should be present
        assertThat(span.getAttributes().get(AttributeKey.stringKey("llm.output_messages.0.message.role")))
                .isEqualTo("assistant");
        assertThat(span.getAttributes().get(AttributeKey.stringKey("llm.output_messages.0.message.content")))
                .isEqualTo("Hi there");

        // Output value should be present
        assertThat(span.getAttributes().get(AttributeKey.stringKey(SemanticConventions.OUTPUT_VALUE)))
                .isNotNull();
        assertThat(span.getAttributes().get(AttributeKey.stringKey(SemanticConventions.OUTPUT_MIME_TYPE)))
                .isEqualTo("application/json");

        // Model attributes present
        assertThat(span.getAttributes().get(AttributeKey.stringKey(SemanticConventions.LLM_MODEL_NAME)))
                .isEqualTo("gpt-4");
    }

    // ── hideInputMessages: suppresses llm.input_messages.* but NOT input.value ──

    @Test
    void hideInputMessages_suppressesInputMessages_butKeepsInputValue() {
        TraceConfig config = TraceConfig.builder().hideInputMessages(true).build();
        SpringAIInstrumentor instrumentor = createInstrumentor(config);
        ChatModelObservationContext context = createMockContext("Secret input", "Response");

        simulateFullCall(instrumentor, context);
        SpanData span = getSpan();

        // Input messages should be suppressed
        assertThat(span.getAttributes().get(AttributeKey.stringKey("llm.input_messages.0.message.role")))
                .isNull();
        assertThat(span.getAttributes().get(AttributeKey.stringKey("llm.input_messages.0.message.content")))
                .isNull();

        // Input value should still be present (only gated by hideInputs)
        assertThat(span.getAttributes().get(AttributeKey.stringKey(SemanticConventions.INPUT_VALUE)))
                .isNotNull();
        assertThat(span.getAttributes().get(AttributeKey.stringKey(SemanticConventions.INPUT_MIME_TYPE)))
                .isEqualTo("application/json");

        // Output side should be unaffected
        assertThat(span.getAttributes().get(AttributeKey.stringKey("llm.output_messages.0.message.role")))
                .isEqualTo("assistant");
        assertThat(span.getAttributes().get(AttributeKey.stringKey(SemanticConventions.OUTPUT_VALUE)))
                .isNotNull();
    }

    // ── hideOutputMessages: suppresses llm.output_messages.* but NOT output.value ──

    @Test
    void hideOutputMessages_suppressesOutputMessages_butKeepsOutputValue() {
        TraceConfig config = TraceConfig.builder().hideOutputMessages(true).build();
        SpringAIInstrumentor instrumentor = createInstrumentor(config);
        ChatModelObservationContext context = createMockContext("Input", "Secret output");

        simulateFullCall(instrumentor, context);
        SpanData span = getSpan();

        // Output messages should be suppressed
        assertThat(span.getAttributes().get(AttributeKey.stringKey("llm.output_messages.0.message.role")))
                .isNull();
        assertThat(span.getAttributes().get(AttributeKey.stringKey("llm.output_messages.0.message.content")))
                .isNull();

        // Output value should still be present (only gated by hideOutputs)
        assertThat(span.getAttributes().get(AttributeKey.stringKey(SemanticConventions.OUTPUT_VALUE)))
                .isNotNull();
        assertThat(span.getAttributes().get(AttributeKey.stringKey(SemanticConventions.OUTPUT_MIME_TYPE)))
                .isEqualTo("application/json");

        // Input side should be unaffected
        assertThat(span.getAttributes().get(AttributeKey.stringKey("llm.input_messages.0.message.role")))
                .isEqualTo("user");
        assertThat(span.getAttributes().get(AttributeKey.stringKey(SemanticConventions.INPUT_VALUE)))
                .isNotNull();
    }

    // ── hideInputs: suppresses BOTH input.value AND llm.input_messages.* ──

    @Test
    void hideInputs_suppressesBothInputValueAndInputMessages() {
        TraceConfig config = TraceConfig.builder().hideInputs(true).build();
        SpringAIInstrumentor instrumentor = createInstrumentor(config);
        ChatModelObservationContext context = createMockContext("Secret", "Response");

        simulateFullCall(instrumentor, context);
        SpanData span = getSpan();

        // Input messages should be suppressed
        assertThat(span.getAttributes().get(AttributeKey.stringKey("llm.input_messages.0.message.role")))
                .isNull();
        assertThat(span.getAttributes().get(AttributeKey.stringKey("llm.input_messages.0.message.content")))
                .isNull();

        // Input value should be suppressed
        assertThat(span.getAttributes().get(AttributeKey.stringKey(SemanticConventions.INPUT_VALUE)))
                .isNull();
        assertThat(span.getAttributes().get(AttributeKey.stringKey(SemanticConventions.INPUT_MIME_TYPE)))
                .isNull();

        // Output side should be unaffected
        assertThat(span.getAttributes().get(AttributeKey.stringKey("llm.output_messages.0.message.role")))
                .isEqualTo("assistant");
        assertThat(span.getAttributes().get(AttributeKey.stringKey(SemanticConventions.OUTPUT_VALUE)))
                .isNotNull();

        // Model attributes still present
        assertThat(span.getAttributes().get(AttributeKey.stringKey(SemanticConventions.LLM_MODEL_NAME)))
                .isEqualTo("gpt-4");
    }

    // ── hideOutputs: suppresses BOTH output.value AND llm.output_messages.* ──

    @Test
    void hideOutputs_suppressesBothOutputValueAndOutputMessages() {
        TraceConfig config = TraceConfig.builder().hideOutputs(true).build();
        SpringAIInstrumentor instrumentor = createInstrumentor(config);
        ChatModelObservationContext context = createMockContext("Hello", "Secret response");

        simulateFullCall(instrumentor, context);
        SpanData span = getSpan();

        // Output messages should be suppressed
        assertThat(span.getAttributes().get(AttributeKey.stringKey("llm.output_messages.0.message.role")))
                .isNull();
        assertThat(span.getAttributes().get(AttributeKey.stringKey("llm.output_messages.0.message.content")))
                .isNull();

        // Output value should be suppressed
        assertThat(span.getAttributes().get(AttributeKey.stringKey(SemanticConventions.OUTPUT_VALUE)))
                .isNull();
        assertThat(span.getAttributes().get(AttributeKey.stringKey(SemanticConventions.OUTPUT_MIME_TYPE)))
                .isNull();

        // Input side should be unaffected
        assertThat(span.getAttributes().get(AttributeKey.stringKey("llm.input_messages.0.message.role")))
                .isEqualTo("user");
        assertThat(span.getAttributes().get(AttributeKey.stringKey(SemanticConventions.INPUT_VALUE)))
                .isNotNull();

        // Model attributes still present
        assertThat(span.getAttributes().get(AttributeKey.stringKey(SemanticConventions.LLM_MODEL_NAME)))
                .isEqualTo("gpt-4");
    }

    // ── Both hideInputs and hideInputMessages: everything suppressed ──

    @Test
    void hideInputsAndHideInputMessages_bothSuppressEverything() {
        TraceConfig config =
                TraceConfig.builder().hideInputs(true).hideInputMessages(true).build();
        SpringAIInstrumentor instrumentor = createInstrumentor(config);
        ChatModelObservationContext context = createMockContext("Secret", "Response");

        simulateFullCall(instrumentor, context);
        SpanData span = getSpan();

        // All input attributes suppressed
        assertThat(span.getAttributes().get(AttributeKey.stringKey("llm.input_messages.0.message.role")))
                .isNull();
        assertThat(span.getAttributes().get(AttributeKey.stringKey(SemanticConventions.INPUT_VALUE)))
                .isNull();
        assertThat(span.getAttributes().get(AttributeKey.stringKey(SemanticConventions.INPUT_MIME_TYPE)))
                .isNull();
    }

    // ── Both hideOutputs and hideOutputMessages: everything suppressed ──

    @Test
    void hideOutputsAndHideOutputMessages_bothSuppressEverything() {
        TraceConfig config =
                TraceConfig.builder().hideOutputs(true).hideOutputMessages(true).build();
        SpringAIInstrumentor instrumentor = createInstrumentor(config);
        ChatModelObservationContext context = createMockContext("Hello", "Secret");

        simulateFullCall(instrumentor, context);
        SpanData span = getSpan();

        // All output attributes suppressed
        assertThat(span.getAttributes().get(AttributeKey.stringKey("llm.output_messages.0.message.role")))
                .isNull();
        assertThat(span.getAttributes().get(AttributeKey.stringKey(SemanticConventions.OUTPUT_VALUE)))
                .isNull();
        assertThat(span.getAttributes().get(AttributeKey.stringKey(SemanticConventions.OUTPUT_MIME_TYPE)))
                .isNull();
    }

    // ── Hide all input and output ──

    @Test
    void hideEverything_noInputOrOutputAttributes() {
        TraceConfig config = TraceConfig.builder()
                .hideInputs(true)
                .hideOutputs(true)
                .hideInputMessages(true)
                .hideOutputMessages(true)
                .build();
        SpringAIInstrumentor instrumentor = createInstrumentor(config);
        ChatModelObservationContext context = createMockContext("Secret", "Secret");

        simulateFullCall(instrumentor, context);
        SpanData span = getSpan();

        // No input attributes
        assertThat(span.getAttributes().get(AttributeKey.stringKey("llm.input_messages.0.message.role")))
                .isNull();
        assertThat(span.getAttributes().get(AttributeKey.stringKey(SemanticConventions.INPUT_VALUE)))
                .isNull();

        // No output attributes
        assertThat(span.getAttributes().get(AttributeKey.stringKey("llm.output_messages.0.message.role")))
                .isNull();
        assertThat(span.getAttributes().get(AttributeKey.stringKey(SemanticConventions.OUTPUT_VALUE)))
                .isNull();

        // But model/span-kind attributes still present
        assertThat(span.getAttributes().get(AttributeKey.stringKey(SemanticConventions.LLM_MODEL_NAME)))
                .isEqualTo("gpt-4");
        assertThat(span.getAttributes().get(AttributeKey.stringKey(SemanticConventions.OPENINFERENCE_SPAN_KIND)))
                .isEqualTo("LLM");
    }

    // ── Cross-flag independence ──

    @Test
    void hideInputMessages_doesNotAffectOutputMessages() {
        TraceConfig config = TraceConfig.builder().hideInputMessages(true).build();
        SpringAIInstrumentor instrumentor = createInstrumentor(config);
        ChatModelObservationContext context = createMockContext("Secret input", "Visible output");

        simulateFullCall(instrumentor, context);
        SpanData span = getSpan();

        // Input messages suppressed
        assertThat(span.getAttributes().get(AttributeKey.stringKey("llm.input_messages.0.message.role")))
                .isNull();

        // Output messages and value present
        assertThat(span.getAttributes().get(AttributeKey.stringKey("llm.output_messages.0.message.role")))
                .isEqualTo("assistant");
        assertThat(span.getAttributes().get(AttributeKey.stringKey("llm.output_messages.0.message.content")))
                .isEqualTo("Visible output");
        assertThat(span.getAttributes().get(AttributeKey.stringKey(SemanticConventions.OUTPUT_VALUE)))
                .isNotNull();
        assertThat(span.getAttributes().get(AttributeKey.stringKey(SemanticConventions.OUTPUT_MIME_TYPE)))
                .isEqualTo("application/json");
    }

    @Test
    void hideOutputMessages_doesNotAffectInputMessages() {
        TraceConfig config = TraceConfig.builder().hideOutputMessages(true).build();
        SpringAIInstrumentor instrumentor = createInstrumentor(config);
        ChatModelObservationContext context = createMockContext("Visible input", "Secret output");

        simulateFullCall(instrumentor, context);
        SpanData span = getSpan();

        // Output messages suppressed
        assertThat(span.getAttributes().get(AttributeKey.stringKey("llm.output_messages.0.message.role")))
                .isNull();

        // Input messages and value present
        assertThat(span.getAttributes().get(AttributeKey.stringKey("llm.input_messages.0.message.role")))
                .isEqualTo("user");
        assertThat(span.getAttributes().get(AttributeKey.stringKey("llm.input_messages.0.message.content")))
                .isEqualTo("Visible input");
        assertThat(span.getAttributes().get(AttributeKey.stringKey(SemanticConventions.INPUT_VALUE)))
                .isNotNull();
        assertThat(span.getAttributes().get(AttributeKey.stringKey(SemanticConventions.INPUT_MIME_TYPE)))
                .isEqualTo("application/json");
    }

    // ── Key hierarchy test: hideInputs subsumes hideInputMessages ──

    @Test
    void hideInputs_aloneIsSufficient_hideInputMessagesNotNeeded() {
        // hideInputs=true should suppress both input.value AND llm.input_messages.*
        // even without hideInputMessages=true
        TraceConfig config = TraceConfig.builder().hideInputs(true).build();
        SpringAIInstrumentor instrumentor = createInstrumentor(config);
        ChatModelObservationContext context = createMockContext("Secret", "Response");

        simulateFullCall(instrumentor, context);
        SpanData span = getSpan();

        // All input suppressed
        assertThat(span.getAttributes().get(AttributeKey.stringKey("llm.input_messages.0.message.role")))
                .isNull();
        assertThat(span.getAttributes().get(AttributeKey.stringKey("llm.input_messages.0.message.content")))
                .isNull();
        assertThat(span.getAttributes().get(AttributeKey.stringKey(SemanticConventions.INPUT_VALUE)))
                .isNull();
        assertThat(span.getAttributes().get(AttributeKey.stringKey(SemanticConventions.INPUT_MIME_TYPE)))
                .isNull();
    }

    @Test
    void hideOutputs_aloneIsSufficient_hideOutputMessagesNotNeeded() {
        // hideOutputs=true should suppress both output.value AND llm.output_messages.*
        // even without hideOutputMessages=true
        TraceConfig config = TraceConfig.builder().hideOutputs(true).build();
        SpringAIInstrumentor instrumentor = createInstrumentor(config);
        ChatModelObservationContext context = createMockContext("Hello", "Secret");

        simulateFullCall(instrumentor, context);
        SpanData span = getSpan();

        // All output suppressed
        assertThat(span.getAttributes().get(AttributeKey.stringKey("llm.output_messages.0.message.role")))
                .isNull();
        assertThat(span.getAttributes().get(AttributeKey.stringKey("llm.output_messages.0.message.content")))
                .isNull();
        assertThat(span.getAttributes().get(AttributeKey.stringKey(SemanticConventions.OUTPUT_VALUE)))
                .isNull();
        assertThat(span.getAttributes().get(AttributeKey.stringKey(SemanticConventions.OUTPUT_MIME_TYPE)))
                .isNull();
    }
}

package com.arize.instrumentation.springAI;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.*;

import com.arize.instrumentation.OITracer;
import com.arize.instrumentation.TraceConfig;
import com.arize.semconv.trace.SemanticConventions;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.opentelemetry.api.common.AttributeKey;
import io.opentelemetry.sdk.testing.exporter.InMemorySpanExporter;
import io.opentelemetry.sdk.trace.SdkTracerProvider;
import io.opentelemetry.sdk.trace.data.SpanData;
import io.opentelemetry.sdk.trace.export.SimpleSpanProcessor;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.util.Base64;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import javax.imageio.ImageIO;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.ToolResponseMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.Generation;
import org.springframework.ai.chat.observation.ChatModelObservationContext;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.content.Media;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.util.MimeType;
import org.springframework.util.MimeTypeUtils;

class SpringAIInstrumentorMediaTest {

    private static final String JPEG_DATA_URL = "data:image/jpeg;base64,AQIDBA==";
    private static final String PROMPT = "Describe this image.";

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

    private static Media jpeg() {
        return new Media(MimeTypeUtils.IMAGE_JPEG, new ByteArrayResource(new byte[] {1, 2, 3, 4}));
    }

    /** A genuine encoded image, so the captured payload is a real decodable image. */
    private static byte[] realImage(String format) throws Exception {
        System.setProperty("java.awt.headless", "true");
        BufferedImage image = new BufferedImage(8, 8, BufferedImage.TYPE_INT_RGB);
        Graphics2D graphics = image.createGraphics();
        graphics.setColor(Color.RED);
        graphics.fillRect(0, 0, 8, 8);
        graphics.dispose();
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        ImageIO.write(image, format, out);
        return out.toByteArray();
    }

    private SpringAIInstrumentor instrumentor(TraceConfig config) {
        return new SpringAIInstrumentor(new OITracer(tracerProvider.get("test"), config));
    }

    private ChatModelObservationContext context(List<Message> inputs, AssistantMessage output) {
        Prompt prompt = mock(Prompt.class);
        when(prompt.getInstructions()).thenReturn(inputs);
        ChatOptions options = mock(ChatOptions.class);
        when(options.getModel()).thenReturn("gpt-4o");
        when(prompt.getOptions()).thenReturn(options);

        ChatModelObservationContext ctx = mock(ChatModelObservationContext.class);
        when(ctx.getRequest()).thenReturn(prompt);
        if (output != null) {
            ChatResponse response = mock(ChatResponse.class);
            when(response.getResults()).thenReturn(List.of(new Generation(output)));
            when(ctx.getResponse()).thenReturn(response);
        }
        return ctx;
    }

    private SpanData run(SpringAIInstrumentor instrumentor, ChatModelObservationContext ctx) {
        instrumentor.onStart(ctx);
        instrumentor.onStop(ctx);
        List<SpanData> spans = spanExporter.getFinishedSpanItems();
        assertThat(spans).hasSize(1);
        return spans.get(0);
    }

    private static String attr(SpanData span, String key) {
        return span.getAttributes().get(AttributeKey.stringKey(key));
    }

    private static Message imagePrompt(Media... media) {
        return UserMessage.builder().text(PROMPT).media(List.of(media)).build();
    }

    /** Span attributes minus the ones every span carries, so a test can assert the exact remainder. */
    private static Map<String, Object> remainingAttributes(SpanData span) {
        Map<String, Object> attributes = new HashMap<>();
        span.getAttributes().forEach((key, value) -> attributes.put(key.getKey(), value));
        attributes
                .keySet()
                .removeAll(java.util.Set.of(
                        "openinference.span.kind",
                        "llm.model_name",
                        "llm.system",
                        "llm.provider",
                        "llm.invocation_parameters",
                        "input.value",
                        "input.mime_type",
                        "output.value",
                        "output.mime_type"));
        return attributes;
    }

    private static void assertWellFormedMultiPart(String json) throws Exception {
        JsonNode messages = new ObjectMapper().readTree(json);
        assertThat(messages.isArray()).isTrue();
        for (JsonNode message : messages) {
            JsonNode content = message.get("content");
            if (content == null || !content.isArray()) {
                continue; // text-only message: content is a plain string
            }
            for (JsonNode part : content) {
                assertThat(part.hasNonNull("type"))
                        .as("part needs a type: " + part)
                        .isTrue();
                if ("image_url".equals(part.get("type").asText())) {
                    assertThat(part.hasNonNull("image_url"))
                            .as("malformed: " + part)
                            .isTrue();
                    assertThat(part.get("image_url").hasNonNull("url"))
                            .as("malformed: " + part)
                            .isTrue();
                }
            }
        }
    }

    @Test
    void multiModalMessage_emitsExactlyTheExpectedAttributes() {
        SpanData span = run(
                instrumentor(TraceConfig.getDefault()),
                context(List.of(imagePrompt(jpeg())), new AssistantMessage("A red square.")));

        Map<String, Object> remaining = remainingAttributes(span);
        assertThat(remaining.remove("llm.input_messages.0.message.role")).isEqualTo("user");
        assertThat(remaining.remove("llm.input_messages.0.message.contents.0.message_content.type"))
                .isEqualTo("text");
        assertThat(remaining.remove("llm.input_messages.0.message.contents.0.message_content.text"))
                .isEqualTo(PROMPT);
        assertThat(remaining.remove("llm.input_messages.0.message.contents.1.message_content.type"))
                .isEqualTo("image");
        assertThat(remaining.remove("llm.input_messages.0.message.contents.1.message_content.image.image.url"))
                .isEqualTo(JPEG_DATA_URL);
        assertThat(remaining.remove("llm.output_messages.0.message.role")).isEqualTo("assistant");
        assertThat(remaining.remove("llm.output_messages.0.message.content")).isEqualTo("A red square.");

        assertThat(remaining).as("unexpected attributes leaked onto the span").isEmpty();

        assertThat(attr(span, SemanticConventions.INPUT_VALUE))
                .contains("\"type\":\"text\"")
                .contains("\"type\":\"image_url\"")
                .contains(JPEG_DATA_URL);
    }

    @Test
    void messagesWithoutImages_keepFlatContent() {
        Media pdf = new Media(MimeType.valueOf("application/pdf"), new ByteArrayResource(new byte[] {9}));
        SpanData span = run(
                instrumentor(TraceConfig.getDefault()),
                context(
                        List.of(
                                new UserMessage("Hello world"),
                                UserMessage.builder()
                                        .text("Read this")
                                        .media(List.of(pdf))
                                        .build()),
                        null));

        assertThat(attr(span, "llm.input_messages.0.message.content")).isEqualTo("Hello world");
        assertThat(attr(span, "llm.input_messages.1.message.content"))
                .as("non-image media does not switch the message to multi-part")
                .isEqualTo("Read this");
        assertThat(attr(span, "llm.input_messages.1.message.contents.0.message_content.type"))
                .isNull();
        assertThat(attr(span, SemanticConventions.INPUT_VALUE)).doesNotContain("image_url");
    }

    @Test
    void imageWithoutText_emitsTheImageAsTheFirstPart() {
        Message emptyText =
                UserMessage.builder().text("").media(List.of(jpeg())).build();
        AssistantMessage nullText = new AssistantMessage(null, Map.of(), List.of(), List.of(jpeg()));
        SpanData span = run(instrumentor(TraceConfig.getDefault()), context(List.of(emptyText), nullText));

        assertThat(attr(span, "llm.input_messages.0.message.contents.0.message_content.type"))
                .as("empty text: the image takes index 0")
                .isEqualTo("image");
        assertThat(attr(span, "llm.input_messages.0.message.contents.0.message_content.image.image.url"))
                .isEqualTo(JPEG_DATA_URL);
        assertThat(attr(span, "llm.input_messages.0.message.contents.1.message_content.type"))
                .isNull();
        assertThat(attr(span, SemanticConventions.INPUT_VALUE))
                .contains("\"type\":\"image_url\"")
                .doesNotContain("\"type\":\"text\"");

        assertThat(attr(span, "llm.output_messages.0.message.contents.0.message_content.type"))
                .as("null text: the image still takes index 0")
                .isEqualTo("image");
        assertThat(attr(span, SemanticConventions.OUTPUT_VALUE))
                .contains("\"type\":\"image_url\"")
                .doesNotContain("\"type\":\"text\"");
    }

    @Test
    void assistantMessageWithImage_emitsMultiPartContents() {
        AssistantMessage output = new AssistantMessage("Here it is", Map.of(), List.of(), List.of(jpeg()));
        SpanData span =
                run(instrumentor(TraceConfig.getDefault()), context(List.of(new UserMessage("draw a cat")), output));

        assertThat(attr(span, "llm.output_messages.0.message.contents.0.message_content.text"))
                .isEqualTo("Here it is");
        assertThat(attr(span, "llm.output_messages.0.message.contents.1.message_content.image.image.url"))
                .isEqualTo(JPEG_DATA_URL);
        assertThat(attr(span, "llm.output_messages.0.message.content")).isNull();
        assertThat(attr(span, SemanticConventions.OUTPUT_VALUE)).contains("\"type\":\"image_url\"");
    }

    @Test
    void multiTurnConversation_preservesMessageAndPartOrderAndSkipsUnreadableMedia() throws Exception {
        byte[] realJpeg = realImage("jpg");
        SpanData span = run(
                instrumentor(TraceConfig.getDefault()),
                context(
                        List.of(
                                new SystemMessage("You are a vision assistant."),
                                imagePrompt(jpeg()),
                                new AssistantMessage("A red square."),
                                UserMessage.builder()
                                        .text("Compare these")
                                        .media(List.of(
                                                new Media(
                                                        MimeTypeUtils.IMAGE_JPEG,
                                                        new ByteArrayResource(new byte[0])), // unreadable
                                                new Media(MimeTypeUtils.IMAGE_JPEG, new ByteArrayResource(realJpeg)),
                                                new Media(
                                                        MimeTypeUtils.IMAGE_PNG,
                                                        new ByteArrayResource(realImage("png")))))
                                        .build()),
                        null));

        assertThat(attr(span, "llm.input_messages.0.message.content")).isEqualTo("You are a vision assistant.");
        assertThat(attr(span, "llm.input_messages.1.message.contents.0.message_content.text"))
                .isEqualTo(PROMPT);
        assertThat(attr(span, "llm.input_messages.2.message.content")).isEqualTo("A red square.");

        // The unreadable image is skipped without leaving a gap in the content indices.
        assertThat(attr(span, "llm.input_messages.3.message.contents.0.message_content.type"))
                .isEqualTo("text");
        String jpegUrl = attr(span, "llm.input_messages.3.message.contents.1.message_content.image.image.url");
        assertThat(jpegUrl).startsWith("data:image/jpeg;base64,");
        assertThat(attr(span, "llm.input_messages.3.message.contents.2.message_content.image.image.url"))
                .as("each image keeps its own mime type")
                .startsWith("data:image/png;base64,");
        assertThat(attr(span, "llm.input_messages.3.message.contents.3.message_content.type"))
                .isNull();

        byte[] decoded = Base64.getDecoder().decode(jpegUrl.substring(jpegUrl.indexOf(",") + 1));
        assertThat(decoded).isEqualTo(realJpeg);
        assertThat(ImageIO.read(new ByteArrayInputStream(decoded)))
                .as("captured bytes must still decode as an image")
                .isNotNull();
    }

    @Test
    void hiddenImages_areRedactedWithoutEverReadingThePayload() {
        Media hidden = spy(jpeg());
        SpanData span = run(
                instrumentor(TraceConfig.builder().hideInputImages(true).build()),
                context(List.of(imagePrompt(hidden)), null));

        Map<String, Object> remaining = remainingAttributes(span);
        assertThat(remaining.remove("llm.input_messages.0.message.role")).isEqualTo("user");
        assertThat(remaining.remove("llm.input_messages.0.message.contents.0.message_content.type"))
                .isEqualTo("text");
        assertThat(remaining.remove("llm.input_messages.0.message.contents.0.message_content.text"))
                .isEqualTo(PROMPT);
        assertThat(remaining.remove("llm.input_messages.0.message.contents.1.message_content.type"))
                .as("the part survives so a consumer can tell 'no image' from 'image withheld'")
                .isEqualTo("image");
        assertThat(remaining.remove("llm.input_messages.0.message.contents.1.message_content.image.image.url"))
                .isEqualTo("__REDACTED__");
        assertThat(remaining).isEmpty();

        assertThat(attr(span, SemanticConventions.INPUT_VALUE))
                .contains("\"image_url\":{\"url\":\"__REDACTED__\"}")
                .doesNotContain(JPEG_DATA_URL);
        verify(hidden, never()).getData();
        verify(hidden, never()).getDataAsByteArray();

        // Anti-vacuity guard: a visible image does read the payload.
        spanExporter.reset();
        Media visible = spy(jpeg());
        run(instrumentor(TraceConfig.getDefault()), context(List.of(imagePrompt(visible)), null));
        verify(visible, atLeastOnce()).getData();
    }

    @Test
    void hideText_redactsMultiPartFlatAndToolResponseContent() {
        ToolResponseMessage toolResponse = new ToolResponseMessage(
                List.of(new ToolResponseMessage.ToolResponse("call_1", "getWeather", "{\"tempF\":72}")));

        SpanData visible = run(
                instrumentor(TraceConfig.getDefault()),
                context(List.of(toolResponse), new AssistantMessage("Visible output")));
        assertThat(attr(visible, "llm.input_messages.0.message.content")).isEqualTo("{\"tempF\":72}");
        assertThat(attr(visible, "llm.input_messages.0.message.tool_call_id")).isEqualTo("call_1");

        spanExporter.reset();
        ToolResponseMessage noData =
                new ToolResponseMessage(List.of(new ToolResponseMessage.ToolResponse("call_2", "getWeather", null)));
        SpanData missing = run(instrumentor(TraceConfig.getDefault()), context(List.of(noData), null));
        assertThat(attr(missing, "llm.input_messages.0.message.content"))
                .as("with no response data the message's own (empty) text stands, rather than a null overwrite")
                .isEmpty();
        assertThat(attr(missing, "llm.input_messages.0.message.tool_call_id")).isEqualTo("call_2");

        spanExporter.reset();
        SpanData hiddenInput = run(
                instrumentor(TraceConfig.builder().hideInputText(true).build()),
                context(List.of(imagePrompt(jpeg()), new UserMessage("Hello world"), toolResponse), null));
        assertThat(attr(hiddenInput, "llm.input_messages.0.message.contents.0.message_content.text"))
                .as("text inside a multi-part message")
                .isEqualTo("__REDACTED__");
        assertThat(attr(hiddenInput, "llm.input_messages.0.message.contents.1.message_content.image.image.url"))
                .as("hiding text leaves images alone")
                .isEqualTo(JPEG_DATA_URL);
        assertThat(attr(hiddenInput, "llm.input_messages.1.message.content"))
                .as("flat text content")
                .isEqualTo("__REDACTED__");
        assertThat(attr(hiddenInput, "llm.input_messages.2.message.content"))
                .as("tool responses carry raw api payloads and must redact too")
                .isEqualTo("__REDACTED__");
        assertThat(attr(hiddenInput, "llm.input_messages.2.message.tool_call_id"))
                .isEqualTo("call_1");
        assertThat(attr(hiddenInput, SemanticConventions.INPUT_VALUE))
                .doesNotContain(PROMPT)
                .doesNotContain("Hello world");

        spanExporter.reset();
        SpanData hiddenOutput = run(
                instrumentor(TraceConfig.builder().hideOutputText(true).build()),
                context(List.of(new UserMessage("Visible input")), new AssistantMessage("Secret output")));
        assertThat(attr(hiddenOutput, "llm.output_messages.0.message.content")).isEqualTo("__REDACTED__");
        assertThat(attr(hiddenOutput, "llm.input_messages.0.message.content"))
                .as("hiding output text leaves input alone")
                .isEqualTo("Visible input");
        assertThat(attr(hiddenOutput, SemanticConventions.OUTPUT_VALUE)).doesNotContain("Secret output");
    }

    @Test
    void hideInputMessages_stillSuppressesMultiPartContents() {
        TraceConfig config = TraceConfig.builder().hideInputMessages(true).build();
        SpanData span = run(instrumentor(config), context(List.of(imagePrompt(jpeg())), null));

        assertThat(attr(span, "llm.input_messages.0.message.contents.0.message_content.type"))
                .isNull();
        assertThat(attr(span, "llm.input_messages.0.message.contents.1.message_content.image.image.url"))
                .isNull();
    }

    @Test
    void serializedContentIsWellFormedUnderEveryImagePolicy() throws Exception {
        List<TraceConfig> configs = List.of(
                TraceConfig.getDefault(),
                TraceConfig.builder().hideInputImages(true).build(),
                TraceConfig.builder().hideOutputImages(true).build(),
                TraceConfig.builder().hideInputText(true).build());

        for (TraceConfig config : configs) {
            spanExporter.reset();
            Message message =
                    imagePrompt(jpeg(), new Media(MimeTypeUtils.IMAGE_JPEG, new ByteArrayResource(new byte[0])));
            SpanData span = run(instrumentor(config), context(List.of(message), new AssistantMessage("A red square.")));

            assertWellFormedMultiPart(attr(span, SemanticConventions.INPUT_VALUE));
            assertWellFormedMultiPart(attr(span, SemanticConventions.OUTPUT_VALUE));
        }
    }
}

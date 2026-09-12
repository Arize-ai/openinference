package com.arize.instrumentation.springAI;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.when;
import static org.mockito.Mockito.withSettings;

import com.arize.instrumentation.TraceConfig;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.net.URI;
import java.util.Arrays;
import java.util.Base64;
import java.util.List;
import javax.imageio.ImageIO;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.content.Media;
import org.springframework.ai.content.MediaContent;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.util.MimeType;
import org.springframework.util.MimeTypeUtils;

class MediaContentSupportTest {

    private static final String DATA_URL = "data:image/jpeg;base64,AQIDBA==";
    private static final String HTTP_URL = "https://example.com/cat.jpg";
    private static final String REDACTED = MediaContentSupport.REDACTED_VALUE;

    private static Media jpeg(byte[] bytes) {
        return new Media(MimeTypeUtils.IMAGE_JPEG, new ByteArrayResource(bytes));
    }

    private static Media byteImage() {
        return jpeg(new byte[] {1, 2, 3, 4});
    }

    /** A genuine encoded image, so the base64 payload is real rather than arbitrary bytes. */
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

    private static byte[] decodeDataUrl(String dataUrl) {
        return Base64.getDecoder().decode(dataUrl.substring(dataUrl.indexOf(",") + 1));
    }

    private static String oversized() {
        return "data:image/jpeg;base64," + "A".repeat(100_000);
    }

    @Test
    void imageMediaOf_keepsOnlyImageMediaInOrder() {
        assertThat(MediaContentSupport.imageMediaOf(null)).as("null message").isEmpty();
        assertThat(MediaContentSupport.imageMediaOf(new SystemMessage("bot")))
                .as("message type without media")
                .isEmpty();
        assertThat(MediaContentSupport.imageMediaOf(
                        UserMessage.builder().text("hi").media(List.of()).build()))
                .as("empty media list")
                .isEmpty();

        Message nullMedia = mock(Message.class, withSettings().extraInterfaces(MediaContent.class));
        when(((MediaContent) nullMedia).getMedia()).thenReturn(null);
        assertThat(MediaContentSupport.imageMediaOf(nullMedia))
                .as("null media list")
                .isEmpty();

        Media jpeg = byteImage();
        Media png = new Media(MimeTypeUtils.IMAGE_PNG, new ByteArrayResource(new byte[] {5}));
        Media pdf = new Media(MimeType.valueOf("application/pdf"), new ByteArrayResource(new byte[] {9}));
        assertThat(MediaContentSupport.imageMediaOf(UserMessage.builder()
                        .text("look")
                        .media(List.of(jpeg, pdf, png))
                        .build()))
                .as("non-image media is dropped, image order preserved")
                .containsExactly(jpeg, png);

        // Spring AI rejects null elements, but imageMediaOf accepts any MediaContent implementation.
        Message custom = mock(Message.class, withSettings().extraInterfaces(MediaContent.class));
        when(((MediaContent) custom).getMedia()).thenReturn(Arrays.asList(null, jpeg));
        assertThat(MediaContentSupport.imageMediaOf(custom))
                .as("null media entries are skipped")
                .containsExactly(jpeg);

        Message noMimeType = mock(Message.class, withSettings().extraInterfaces(MediaContent.class));
        Media untyped = spy(byteImage());
        when(untyped.getMimeType()).thenReturn(null);
        when(((MediaContent) noMimeType).getMedia()).thenReturn(List.of(untyped));
        assertThat(MediaContentSupport.imageMediaOf(noMimeType))
                .as("media without a mime type cannot be classified as an image")
                .isEmpty();
    }

    @Test
    void toUrl_rendersUrisVerbatimAndBytesAsBase64() {
        assertThat(MediaContentSupport.toUrl(new Media(MimeTypeUtils.IMAGE_JPEG, URI.create(HTTP_URL))))
                .isEqualTo(HTTP_URL);
        assertThat(MediaContentSupport.toUrl(new Media(MimeTypeUtils.IMAGE_JPEG, URI.create(""))))
                .as("an empty uri carries no image")
                .isNull();
        assertThat(MediaContentSupport.toUrl(byteImage())).isEqualTo(DATA_URL);

        // Payloads that are neither a uri nor a raw byte[] fall back to getDataAsByteArray().
        Media resourceBacked = spy(byteImage());
        when(resourceBacked.getData()).thenReturn(new Object());
        when(resourceBacked.getDataAsByteArray()).thenReturn(new byte[] {9, 9});
        assertThat(MediaContentSupport.toUrl(resourceBacked)).isEqualTo("data:image/jpeg;base64,CQk=");

        Media untyped = spy(byteImage());
        when(untyped.getMimeType()).thenReturn(null);
        assertThat(MediaContentSupport.toUrl(untyped))
                .as("a missing mime type falls back to octet-stream")
                .isEqualTo("data:application/octet-stream;base64,AQIDBA==");
    }

    @Test
    void toUrl_realImages_roundTripUnderTheirOwnMimeType() throws Exception {
        byte[] jpeg = realImage("jpg");
        String jpegUrl = MediaContentSupport.toUrl(jpeg(jpeg));
        assertThat(jpegUrl).startsWith("data:image/jpeg;base64,");
        byte[] decodedJpeg = decodeDataUrl(jpegUrl);
        assertThat(decodedJpeg).isEqualTo(jpeg);
        assertThat(decodedJpeg[0] & 0xFF).as("JPEG SOI").isEqualTo(0xFF);
        assertThat(decodedJpeg[1] & 0xFF).as("JPEG SOI").isEqualTo(0xD8);
        assertThat(decodedJpeg[decodedJpeg.length - 2] & 0xFF).as("JPEG EOI").isEqualTo(0xFF);
        assertThat(decodedJpeg[decodedJpeg.length - 1] & 0xFF).as("JPEG EOI").isEqualTo(0xD9);

        byte[] png = realImage("png");
        String pngUrl = MediaContentSupport.toUrl(new Media(MimeTypeUtils.IMAGE_PNG, new ByteArrayResource(png)));
        assertThat(pngUrl).startsWith("data:image/png;base64,");
        assertThat(decodeDataUrl(pngUrl)).isEqualTo(png);
        assertThat(new String(decodeDataUrl(pngUrl), 1, 3)).as("PNG signature").isEqualTo("PNG");
    }

    @Test
    void toUrl_returnsNullWhenThePayloadCannotBeRead() {
        assertThat(MediaContentSupport.toUrl(null)).as("null media").isNull();
        assertThat(MediaContentSupport.toUrl(jpeg(new byte[0])))
                .as("empty bytes")
                .isNull();

        Media noData = spy(byteImage());
        when(noData.getData()).thenReturn(null);
        assertThat(MediaContentSupport.toUrl(noData)).as("null payload").isNull();

        Media unreadable = spy(byteImage());
        when(unreadable.getData()).thenReturn(new Object());
        when(unreadable.getDataAsByteArray()).thenReturn(null);
        assertThat(MediaContentSupport.toUrl(unreadable)).as("null byte array").isNull();

        Media broken = spy(byteImage());
        when(broken.getData()).thenThrow(new IllegalStateException("boom"));
        assertThat(MediaContentSupport.toUrl(broken))
                .as("a throwing payload must not break the span")
                .isNull();
    }

    @Test
    void isImageHidden_followsDirectionAndToleratesNullConfig() {
        TraceConfig hideInput = TraceConfig.builder().hideInputImages(true).build();
        assertThat(MediaContentSupport.isImageHidden(hideInput, true)).isTrue();
        assertThat(MediaContentSupport.isImageHidden(hideInput, false)).isFalse();

        TraceConfig hideOutput = TraceConfig.builder().hideOutputImages(true).build();
        assertThat(MediaContentSupport.isImageHidden(hideOutput, false)).isTrue();
        assertThat(MediaContentSupport.isImageHidden(hideOutput, true)).isFalse();

        assertThat(MediaContentSupport.isImageHidden(null, true)).isFalse();
    }

    @Test
    void applyImagePolicy_redactsHiddenImagesInTheMatchingDirectionOnly() {
        TraceConfig hideInput = TraceConfig.builder().hideInputImages(true).build();
        assertThat(MediaContentSupport.applyImagePolicy(DATA_URL, hideInput, true))
                .isEqualTo(REDACTED);
        assertThat(MediaContentSupport.applyImagePolicy(DATA_URL, hideInput, false))
                .isEqualTo(DATA_URL);

        TraceConfig hideOutput = TraceConfig.builder().hideOutputImages(true).build();
        assertThat(MediaContentSupport.applyImagePolicy(DATA_URL, hideOutput, false))
                .isEqualTo(REDACTED);
        assertThat(MediaContentSupport.applyImagePolicy(DATA_URL, hideOutput, true))
                .isEqualTo(DATA_URL);

        // Null in, null out: an unreadable payload stays distinguishable from a withheld one.
        assertThat(MediaContentSupport.applyImagePolicy(null, hideInput, true)).isNull();
        assertThat(MediaContentSupport.applyImagePolicy(DATA_URL, null, true)).isEqualTo(DATA_URL);
    }

    @Test
    void applyImagePolicy_neverCapsImageSize() {
        // No length limit is applied, whatever base64ImageMaxLength says.
        for (TraceConfig config : List.of(
                TraceConfig.getDefault(),
                TraceConfig.builder().base64ImageMaxLength("10").build(),
                TraceConfig.builder().base64ImageMaxLength("not-a-number").build())) {
            assertThat(MediaContentSupport.applyImagePolicy(oversized(), config, true))
                    .as("no cap under '" + config.getBase64ImageMaxLength() + "'")
                    .isEqualTo(oversized());
        }
    }

    @Test
    void isBase64Url_recognisesOnlyBase64EncodedImageDataUrls() {
        assertThat(MediaContentSupport.isBase64Url(DATA_URL)).isTrue();
        assertThat(MediaContentSupport.isBase64Url(HTTP_URL)).as("remote url").isFalse();
        assertThat(MediaContentSupport.isBase64Url("data:image/svg+xml,<svg/>"))
                .as("data url that is not base64 encoded")
                .isFalse();
        assertThat(MediaContentSupport.isBase64Url("data:audio/mp3;base64,AAAA"))
                .as("base64 data url that is not an image")
                .isFalse();
        assertThat(MediaContentSupport.isBase64Url(null)).isFalse();
    }

    @Test
    void applyTextPolicy_redactsInTheMatchingDirectionOnly() {
        TraceConfig hideInput = TraceConfig.builder().hideInputText(true).build();
        assertThat(MediaContentSupport.applyTextPolicy("hello", hideInput, true))
                .isEqualTo(REDACTED);
        assertThat(MediaContentSupport.applyTextPolicy("hello", hideInput, false))
                .isEqualTo("hello");

        TraceConfig hideOutput = TraceConfig.builder().hideOutputText(true).build();
        assertThat(MediaContentSupport.applyTextPolicy("hello", hideOutput, false))
                .isEqualTo(REDACTED);
        assertThat(MediaContentSupport.applyTextPolicy("hello", hideOutput, true))
                .isEqualTo("hello");

        assertThat(MediaContentSupport.applyTextPolicy(null, hideInput, true)).isNull();
        assertThat(MediaContentSupport.applyTextPolicy("hello", null, true)).isEqualTo("hello");
    }
}

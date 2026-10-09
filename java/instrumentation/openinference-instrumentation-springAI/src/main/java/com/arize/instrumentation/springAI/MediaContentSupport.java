package com.arize.instrumentation.springAI;

import com.arize.instrumentation.TraceConfig;
import java.util.ArrayList;
import java.util.Base64;
import java.util.Collections;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.content.Media;
import org.springframework.ai.content.MediaContent;
import org.springframework.util.MimeType;

final class MediaContentSupport {

    private static final Logger log = LoggerFactory.getLogger(MediaContentSupport.class);

    static final String REDACTED_VALUE = "__REDACTED__";

    private static final String BASE64_URL_PREFIX = "data:";
    private static final String BASE64_URL_MARKER = ";base64,";
    private static final String BASE64_IMAGE_URL_PREFIX = "data:image/";
    private static final String BASE64_MARKER = "base64";
    private static final String DEFAULT_MIME_TYPE = "application/octet-stream";
    private static final String IMAGE_TYPE = "image";

    private MediaContentSupport() {}

    /**
     * Returns the {@code image/*} media attached to the message, or an empty list when the
     * message carries none or does not support media at all.
     */
    static List<Media> imageMediaOf(Message message) {
        if (!(message instanceof MediaContent)) {
            return Collections.emptyList();
        }
        List<Media> media = ((MediaContent) message).getMedia();
        if (media == null || media.isEmpty()) {
            return Collections.emptyList();
        }
        List<Media> images = new ArrayList<>();
        for (Media item : media) {
            if (isImage(item)) {
                images.add(item);
            }
        }
        return images;
    }

    private static boolean isImage(Media media) {
        if (media == null) {
            return false;
        }
        MimeType mimeType = media.getMimeType();
        return mimeType != null && IMAGE_TYPE.equalsIgnoreCase(mimeType.getType());
    }

    /**
     * Renders the media as an image URL. URI-backed media yields the URI verbatim; byte-backed
     * media yields a {@code data:<mime>;base64,<data>} URL. Returns null when the payload cannot
     * be read.
     */
    static String toUrl(Media media) {
        try {
            if (media == null) {
                return null;
            }
            Object data = media.getData();
            if (data == null) {
                return null;
            }
            // Media built from a URI stores the URI text; getDataAsByteArray() throws for these.
            if (data instanceof String uri) {
                return uri.isEmpty() ? null : uri;
            }
            byte[] bytes = (data instanceof byte[]) ? (byte[]) data : media.getDataAsByteArray();
            if (bytes == null || bytes.length == 0) {
                return null;
            }
            return BASE64_URL_PREFIX
                    + mimeTypeOf(media)
                    + BASE64_URL_MARKER
                    + Base64.getEncoder().encodeToString(bytes);
        } catch (Exception e) {
            log.debug("Could not render media as an image url", e);
            return null;
        }
    }

    private static String mimeTypeOf(Media media) {
        MimeType mimeType = media.getMimeType();
        return mimeType == null ? DEFAULT_MIME_TYPE : mimeType.toString();
    }

    static boolean isImageHidden(TraceConfig config, boolean input) {
        if (config == null) {
            return false;
        }
        return input ? config.isHideInputImages() : config.isHideOutputImages();
    }

    /**
     * Applies the image privacy policy to a rendered url. Hidden images collapse to
     * {@link #REDACTED_VALUE}, per the {@code Hiding Images} section of the multimodal
     * specification. Null in means null out, so an unreadable payload stays distinguishable from a
     * redacted one.
     */
    static String applyImagePolicy(String url, TraceConfig config, boolean input) {
        if (url == null || config == null) {
            return url;
        }
        if (input ? config.isHideInputImages() : config.isHideOutputImages()) {
            return REDACTED_VALUE;
        }
        return url;
    }

    static String applyTextPolicy(String text, TraceConfig config, boolean input) {
        if (text == null || config == null) {
            return text;
        }
        if (input ? config.isHideInputText() : config.isHideOutputText()) {
            return REDACTED_VALUE;
        }
        return text;
    }

    static boolean isBase64Url(String url) {
        return url != null && url.startsWith(BASE64_IMAGE_URL_PREFIX) && url.contains(BASE64_MARKER);
    }
}

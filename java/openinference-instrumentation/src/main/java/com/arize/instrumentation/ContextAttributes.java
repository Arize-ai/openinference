package com.arize.instrumentation;

import com.arize.instrumentation.trace.TracedSpan;
import com.arize.semconv.trace.SemanticConventions;
import io.opentelemetry.context.Context;
import io.opentelemetry.context.ContextKey;
import io.opentelemetry.context.Scope;
import java.util.List;

/**
 * Utility for propagating context attributes (session ID, user ID, metadata, tags)
 * to OpenInference spans via OpenTelemetry Context.
 *
 * <p>Usage:
 * <pre>{@code
 * try (Scope ignored = ContextAttributes.builder()
 *         .sessionId("session-123")
 *         .userId("user-456")
 *         .build()) {
 *     // All OpenInference spans created in this block will have these attributes
 * }
 * }</pre>
 */
public final class ContextAttributes {

    static final ContextKey<String> SESSION_ID_KEY = ContextKey.named(SemanticConventions.SESSION_ID);
    static final ContextKey<String> USER_ID_KEY = ContextKey.named(SemanticConventions.USER_ID);
    static final ContextKey<String> METADATA_KEY = ContextKey.named(SemanticConventions.METADATA);
    static final ContextKey<List<String>> TAGS_KEY = ContextKey.named(SemanticConventions.TAG_TAGS);

    private ContextAttributes() {}

    /**
     * Creates a new builder for setting context attributes.
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Applies any context attributes found in the current OTel Context to the given span.
     */
    public static void applyToSpan(TracedSpan span) {
        Context ctx = Context.current();

        String sessionId = ctx.get(SESSION_ID_KEY);
        if (sessionId != null && !sessionId.isEmpty()) {
            span.setSessionId(sessionId);
        }

        String userId = ctx.get(USER_ID_KEY);
        if (userId != null && !userId.isEmpty()) {
            span.setUserId(userId);
        }

        String metadata = ctx.get(METADATA_KEY);
        if (metadata != null && !metadata.isEmpty()) {
            span.setAttribute(SemanticConventions.METADATA, metadata);
        }

        List<String> tags = ctx.get(TAGS_KEY);
        if (tags != null && !tags.isEmpty()) {
            span.setTags(tags);
        }
    }

    public static final class Builder {
        private String sessionId;
        private String userId;
        private String metadata;
        private List<String> tags;

        private Builder() {}

        public Builder sessionId(String sessionId) {
            this.sessionId = sessionId;
            return this;
        }

        public Builder userId(String userId) {
            this.userId = userId;
            return this;
        }

        public Builder metadata(String metadataJson) {
            this.metadata = metadataJson;
            return this;
        }

        public Builder tags(List<String> tags) {
            this.tags = tags;
            return this;
        }

        /**
         * Attaches the context attributes to the current OTel Context and returns a Scope.
         * Close the scope to detach the attributes.
         */
        public Scope build() {
            Context ctx = Context.current();
            if (sessionId != null && !sessionId.isEmpty()) {
                ctx = ctx.with(SESSION_ID_KEY, sessionId);
            }
            if (userId != null && !userId.isEmpty()) {
                ctx = ctx.with(USER_ID_KEY, userId);
            }
            if (metadata != null && !metadata.isEmpty()) {
                ctx = ctx.with(METADATA_KEY, metadata);
            }
            if (tags != null && !tags.isEmpty()) {
                ctx = ctx.with(TAGS_KEY, tags);
            }
            return ctx.makeCurrent();
        }
    }
}

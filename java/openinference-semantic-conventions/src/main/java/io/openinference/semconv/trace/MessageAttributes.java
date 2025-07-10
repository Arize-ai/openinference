package io.openinference.semconv.trace;

import io.opentelemetry.api.common.AttributeKey;

/**
 * Attributes for a message sent to or from an LLM.
 */
public final class MessageAttributes {

    public static final AttributeKey<String> MESSAGE_ROLE = AttributeKey.stringKey("message.role");
    public static final AttributeKey<String> MESSAGE_CONTENT = AttributeKey.stringKey("message.content");
    public static final AttributeKey<String> MESSAGE_CONTENTS = AttributeKey.stringKey("message.contents");
    public static final AttributeKey<String> MESSAGE_NAME = AttributeKey.stringKey("message.name");
    public static final AttributeKey<String> MESSAGE_TOOL_CALLS = AttributeKey.stringKey("message.tool_calls");
    public static final AttributeKey<String> MESSAGE_FUNCTION_CALL_NAME =
            AttributeKey.stringKey("message.function_call_name");
    public static final AttributeKey<String> MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON =
            AttributeKey.stringKey("message.function_call_arguments_json");
    public static final AttributeKey<String> MESSAGE_TOOL_CALL_ID = AttributeKey.stringKey("message.tool_call_id");

    private MessageAttributes() {
        // Private constructor to prevent instantiation
    }
}

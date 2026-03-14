package com.arize.instrumentation.annotations;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.lang.reflect.Method;
import java.lang.reflect.Parameter;
import java.util.LinkedHashMap;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class SpanHelper {

    private static final Logger log = LoggerFactory.getLogger(SpanHelper.class);
    private static final ObjectMapper MAPPER = new ObjectMapper();

    public record SerializedValue(String value, boolean isJson) {}

    public static SerializedValue serialize(Object obj) {
        if (obj == null) {
            return null;
        }
        if (obj instanceof String s) {
            return new SerializedValue(s, false);
        }
        try {
            return new SerializedValue(MAPPER.writeValueAsString(obj), true);
        } catch (JsonProcessingException e) {
            log.warn(
                    "Failed to serialize object of type {}, falling back to toString",
                    obj.getClass().getName(),
                    e);
            return new SerializedValue(obj.toString(), false);
        }
    }

    public static Map<String, Object> buildInputMap(Method method, Object[] args) {
        Parameter[] params = method.getParameters();
        Map<String, Object> inputMap = new LinkedHashMap<>();

        for (int i = 0; i < params.length; i++) {
            if (params[i].isAnnotationPresent(SpanIgnore.class)) {
                continue;
            }
            String name = params[i].isNamePresent() ? params[i].getName() : "arg" + i;
            inputMap.put(name, args[i]);
        }

        return inputMap;
    }

    public static Object extractField(Object obj, String fieldPath) {
        if (obj == null || fieldPath == null || fieldPath.isEmpty()) {
            return null;
        }
        try {
            // Convert dot notation to JSON Pointer: usage.totalTokens -> /usage/totalTokens
            String jsonPointer = "/" + fieldPath.replace(".", "/");
            var tree = MAPPER.valueToTree(obj);
            var node = tree.at(jsonPointer);
            if (node.isMissingNode() || node.isNull()) {
                log.debug(
                        "Field path '{}' not found in object of type {}",
                        fieldPath,
                        obj.getClass().getName());
                return null;
            }
            if (node.isTextual()) return node.asText();
            if (node.isInt()) return node.asInt();
            if (node.isLong()) return node.asLong();
            if (node.isDouble()) return node.asDouble();
            if (node.isBoolean()) return node.asBoolean();
            return MAPPER.treeToValue(node, Object.class);
        } catch (Exception e) {
            log.warn(
                    "Failed to extract field '{}' from object of type {}",
                    fieldPath,
                    obj.getClass().getName(),
                    e);
            return null;
        }
    }
}

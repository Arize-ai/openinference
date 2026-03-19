package com.arize.instrumentation.annotation;

import java.lang.reflect.Method;
import java.lang.reflect.Parameter;
import java.util.LinkedHashMap;
import java.util.Map;

public class SpanHelper {

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
}

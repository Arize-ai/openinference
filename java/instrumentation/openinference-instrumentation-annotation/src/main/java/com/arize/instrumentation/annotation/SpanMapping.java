package com.arize.instrumentation.annotation;

import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;

@Retention(RetentionPolicy.RUNTIME)
public @interface SpanMapping {
    String param() default "";

    String field() default "";

    String attribute();
}

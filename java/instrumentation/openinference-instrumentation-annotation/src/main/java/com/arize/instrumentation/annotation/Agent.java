package com.arize.instrumentation.annotation;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.METHOD)
public @interface Agent {
    String name() default "";

    SpanMapping[] mapping() default {};

    SpanMapping[] outputMapping() default {};
}

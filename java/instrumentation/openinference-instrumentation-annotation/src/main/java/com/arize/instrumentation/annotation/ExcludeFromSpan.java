package com.arize.instrumentation.annotation;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Annotate a method parameter to exclude it from auto-captured span input.
 *
 * <p>By default, all parameters of an annotated method are captured as span input.
 * Use this annotation on parameters that should not be recorded (e.g., API keys,
 * internal context objects).
 */
@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.PARAMETER)
public @interface ExcludeFromSpan {}

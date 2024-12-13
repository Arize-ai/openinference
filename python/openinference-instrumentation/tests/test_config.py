import os
from contextlib import suppress
from random import random
from typing import Any, Dict, Optional

import pytest
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace import SpanLimits
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import TracerProvider, get_tracer, use_span
from opentelemetry.util.types import AttributeValue

from openinference.instrumentation import TraceConfig
from openinference.instrumentation.config import (
    _IMPORTANT_ATTRIBUTES,
    DEFAULT_BASE64_IMAGE_MAX_LENGTH,
    DEFAULT_HIDE_INPUT_IMAGES,
    DEFAULT_HIDE_INPUT_MESSAGES,
    DEFAULT_HIDE_INPUT_TEXT,
    DEFAULT_HIDE_INPUTS,
    DEFAULT_HIDE_LLM_INVOCATION_PARAMETERS,
    DEFAULT_HIDE_OUTPUT_MESSAGES,
    DEFAULT_HIDE_OUTPUT_TEXT,
    DEFAULT_HIDE_OUTPUTS,
    OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH,
    OPENINFERENCE_HIDE_INPUT_IMAGES,
    OPENINFERENCE_HIDE_INPUT_MESSAGES,
    OPENINFERENCE_HIDE_INPUT_TEXT,
    OPENINFERENCE_HIDE_INPUTS,
    OPENINFERENCE_HIDE_OUTPUT_MESSAGES,
    OPENINFERENCE_HIDE_OUTPUT_TEXT,
    OPENINFERENCE_HIDE_OUTPUTS,
    REDACTED_VALUE,
    OITracer,
)
from openinference.semconv.trace import SpanAttributes


def test_default_settings() -> None:
    config = TraceConfig()
    assert config.hide_llm_invocation_parameters == DEFAULT_HIDE_LLM_INVOCATION_PARAMETERS
    assert config.hide_inputs == DEFAULT_HIDE_INPUTS
    assert config.hide_outputs == DEFAULT_HIDE_OUTPUTS
    assert config.hide_input_messages == DEFAULT_HIDE_INPUT_MESSAGES
    assert config.hide_output_messages == DEFAULT_HIDE_OUTPUT_MESSAGES
    assert config.hide_input_images == DEFAULT_HIDE_INPUT_IMAGES
    assert config.hide_input_text == DEFAULT_HIDE_INPUT_TEXT
    assert config.hide_output_text == DEFAULT_HIDE_OUTPUT_TEXT
    assert config.base64_image_max_length == DEFAULT_BASE64_IMAGE_MAX_LENGTH


def test_oi_tracer(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    tracer = OITracer(tracer_provider.get_tracer(__name__), TraceConfig(hide_inputs=True))
    with tracer.start_as_current_span("a", attributes={"input.value": "c"}):
        tracer.start_span("b", attributes={"input.value": "d"}).end()
    assert len(spans := in_memory_span_exporter.get_finished_spans()) == 2
    for span in spans:
        assert (span.attributes or {}).get("input.value") == REDACTED_VALUE


@pytest.mark.parametrize("k", _IMPORTANT_ATTRIBUTES)
def test_attribute_priority(k: str, in_memory_span_exporter: InMemorySpanExporter) -> None:
    limit = 2
    tracer_provider = trace_sdk.TracerProvider(span_limits=SpanLimits(max_attributes=limit))
    tracer_provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))
    tracer = OITracer(tracer_provider.get_tracer(__name__), TraceConfig())
    v: AttributeValue = random()
    attributes: Dict[str, AttributeValue] = {k: v}
    extra_attributes: Dict[str, AttributeValue] = dict(zip("12345", "54321"))
    assert len(extra_attributes) > limit
    with tracer.start_as_current_span("0", attributes=attributes) as span0:
        span0.set_attributes(extra_attributes)
    with tracer.start_as_current_span("1") as span1:
        span1.set_attributes(extra_attributes)
        span1.set_attributes(attributes)
        span1.set_attributes(extra_attributes)
    span2 = tracer.start_span("2", attributes=attributes)
    span2.set_attributes(extra_attributes)
    span2.end()
    span3 = tracer.start_span("3")
    span3.set_attributes(extra_attributes)
    span3.set_attributes(attributes)
    span3.set_attributes(extra_attributes)
    span3.end()
    with suppress(RuntimeError):
        with tracer.start_as_current_span("4", attributes=attributes):
            span0.set_attributes(extra_attributes)
            raise RuntimeError
    with suppress(RuntimeError):
        with tracer.start_as_current_span("5") as span5:
            span5.set_attributes(extra_attributes)
            span5.set_attributes(attributes)
            span5.set_attributes(extra_attributes)
            raise RuntimeError
    with suppress(RuntimeError):
        with use_span(tracer.start_span("6", attributes=attributes), True) as span6:
            span6.set_attributes(extra_attributes)
            raise RuntimeError
    with suppress(RuntimeError):
        with use_span(tracer.start_span("7"), True) as span7:
            span7.set_attributes(extra_attributes)
            span7.set_attributes(attributes)
            span7.set_attributes(extra_attributes)
            raise RuntimeError
    assert len(spans := in_memory_span_exporter.get_finished_spans()) == 8
    for span in spans:
        assert (span.attributes or {}).get(k) == v


@pytest.mark.parametrize("hide_inputs", [False, True])
@pytest.mark.parametrize("hide_outputs", [False, True])
@pytest.mark.parametrize("hide_input_messages", [False, True])
@pytest.mark.parametrize("hide_output_messages", [False, True])
@pytest.mark.parametrize("hide_input_images", [False, True])
@pytest.mark.parametrize("hide_input_text", [False, True])
@pytest.mark.parametrize("hide_output_text", [False, True])
@pytest.mark.parametrize("base64_image_max_length", [10_000])
def test_settings_from_env_vars_and_code(
    hide_inputs: bool,
    hide_outputs: bool,
    hide_input_messages: bool,
    hide_output_messages: bool,
    hide_input_images: bool,
    hide_input_text: bool,
    hide_output_text: bool,
    base64_image_max_length: int,
) -> None:
    # First part of the test verifies that environment variables are read correctly
    os.environ[OPENINFERENCE_HIDE_INPUTS] = str(hide_inputs)
    os.environ[OPENINFERENCE_HIDE_OUTPUTS] = str(hide_outputs)
    os.environ[OPENINFERENCE_HIDE_INPUT_MESSAGES] = str(hide_input_messages)
    os.environ[OPENINFERENCE_HIDE_OUTPUT_MESSAGES] = str(hide_output_messages)
    os.environ[OPENINFERENCE_HIDE_INPUT_IMAGES] = str(hide_input_images)
    os.environ[OPENINFERENCE_HIDE_INPUT_TEXT] = str(hide_input_text)
    os.environ[OPENINFERENCE_HIDE_OUTPUT_TEXT] = str(hide_output_text)
    os.environ[OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH] = str(base64_image_max_length)

    config = TraceConfig()
    assert config.hide_inputs is parse_bool_from_env(OPENINFERENCE_HIDE_INPUTS)
    assert config.hide_outputs is parse_bool_from_env(OPENINFERENCE_HIDE_OUTPUTS)
    assert config.hide_input_messages is parse_bool_from_env(OPENINFERENCE_HIDE_INPUT_MESSAGES)
    assert config.hide_output_messages is parse_bool_from_env(OPENINFERENCE_HIDE_OUTPUT_MESSAGES)
    assert config.hide_input_images is parse_bool_from_env(OPENINFERENCE_HIDE_INPUT_IMAGES)
    assert config.hide_input_text is parse_bool_from_env(OPENINFERENCE_HIDE_INPUT_TEXT)
    assert config.hide_output_text is parse_bool_from_env(OPENINFERENCE_HIDE_OUTPUT_TEXT)
    assert config.base64_image_max_length == int(
        os.getenv(OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH, default=-1)
    )

    # This next part of the text verifies that the code specified values overwrite
    # the configuration from the environment variables
    new_base64_image_max_length = base64_image_max_length + 500
    new_hide_inputs = not hide_inputs
    new_hide_outputs = not hide_outputs
    new_hide_input_messages = not hide_input_messages
    new_hide_output_messages = not hide_output_messages
    new_hide_input_images = not hide_input_images
    new_hide_input_text = not hide_input_text
    new_hide_output_text = not hide_output_text
    config = TraceConfig(
        hide_inputs=new_hide_inputs,
        hide_outputs=new_hide_outputs,
        hide_input_messages=new_hide_input_messages,
        hide_output_messages=new_hide_output_messages,
        hide_input_images=new_hide_input_images,
        hide_input_text=new_hide_input_text,
        hide_output_text=new_hide_output_text,
        base64_image_max_length=new_base64_image_max_length,
    )
    assert config.hide_inputs is new_hide_inputs
    assert config.hide_outputs is new_hide_outputs
    assert config.hide_input_messages is new_hide_input_messages
    assert config.hide_output_messages is new_hide_output_messages
    assert config.hide_input_images is new_hide_input_images
    assert config.hide_input_text is new_hide_input_text
    assert config.hide_output_text is new_hide_output_text
    assert config.base64_image_max_length == new_base64_image_max_length


@pytest.mark.parametrize(
    "param,param_value,attr_key,attr_value,expected_value",
    [
        (
            "hide_llm_invocation_parameters",
            True,
            SpanAttributes.LLM_INVOCATION_PARAMETERS,
            "{api_key: '123'}",
            None,
        ),
        (
            "hide_llm_invocation_parameters",
            False,
            SpanAttributes.LLM_INVOCATION_PARAMETERS,
            "{api_key: '123'}",
            "{api_key: '123'}",
        ),
    ],
)
def test_hide_llm_invocation_parameters(
    param: str,
    param_value: bool,
    attr_key: str,
    attr_value: Any,
    expected_value: Any,
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    config = TraceConfig(**{param: param_value})
    tracer = OITracer(get_tracer(__name__, tracer_provider=tracer_provider), config=config)
    tracer.start_span("test", attributes={attr_key: attr_value}).end()
    span = in_memory_span_exporter.get_finished_spans()[0]
    assert span.attributes is not None
    if expected_value is None:
        assert span.attributes.get(attr_key) is None
    else:
        assert span.attributes.get(attr_key) == expected_value


def parse_bool_from_env(env_var: str) -> Optional[bool]:
    env_value = os.getenv(env_var)
    if isinstance(env_value, str) and env_value.lower() == "true":
        return True
    elif isinstance(env_value, str) and env_value.lower() == "false":
        return False
    else:
        return None

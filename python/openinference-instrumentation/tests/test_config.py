import os
from contextlib import suppress
from random import random
from typing import Any, Dict, List, Mapping, Optional

import pytest
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace import SpanLimits
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import TracerProvider, use_span
from opentelemetry.util.types import AttributeValue

from openinference.instrumentation import Message, OITracer, TraceConfig, get_llm_attributes
from openinference.instrumentation._spans import _IMPORTANT_ATTRIBUTES  # type:ignore[attr-defined]
from openinference.instrumentation.config import (
    DEFAULT_BASE64_IMAGE_MAX_LENGTH,
    DEFAULT_ENABLE_GENAI_SEMCONV,
    DEFAULT_HIDE_CHOICES,
    DEFAULT_HIDE_EMBEDDING_VECTORS,
    DEFAULT_HIDE_EMBEDDINGS_TEXT,
    DEFAULT_HIDE_EMBEDDINGS_VECTORS,
    DEFAULT_HIDE_INPUT_IMAGES,
    DEFAULT_HIDE_INPUT_MESSAGES,
    DEFAULT_HIDE_INPUT_TEXT,
    DEFAULT_HIDE_INPUTS,
    DEFAULT_HIDE_LLM_INVOCATION_PARAMETERS,
    DEFAULT_HIDE_LLM_TOOLS,
    DEFAULT_HIDE_OUTPUT_MESSAGES,
    DEFAULT_HIDE_OUTPUT_TEXT,
    DEFAULT_HIDE_OUTPUTS,
    DEFAULT_HIDE_PROMPTS,
    DEFAULT_HIDE_REASONING,
    DEFAULT_HIDE_RETRIEVAL_DOCUMENTS,
    OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH,
    OPENINFERENCE_ENABLE_GENAI_SEMCONV,
    OPENINFERENCE_HIDE_CHOICES,
    OPENINFERENCE_HIDE_EMBEDDING_VECTORS,
    OPENINFERENCE_HIDE_EMBEDDINGS_TEXT,
    OPENINFERENCE_HIDE_EMBEDDINGS_VECTORS,
    OPENINFERENCE_HIDE_INPUT_IMAGES,
    OPENINFERENCE_HIDE_INPUT_MESSAGES,
    OPENINFERENCE_HIDE_INPUT_TEXT,
    OPENINFERENCE_HIDE_INPUTS,
    OPENINFERENCE_HIDE_LLM_TOOLS,
    OPENINFERENCE_HIDE_OUTPUT_MESSAGES,
    OPENINFERENCE_HIDE_OUTPUT_TEXT,
    OPENINFERENCE_HIDE_OUTPUTS,
    OPENINFERENCE_HIDE_PROMPTS,
    OPENINFERENCE_HIDE_REASONING,
    OPENINFERENCE_HIDE_RETRIEVAL_DOCUMENTS,
    REDACTED_VALUE,
)
from openinference.semconv.trace import (
    DocumentAttributes,
    ImageAttributes,
    MessageAttributes,
    MessageContentAttributes,
    SpanAttributes,
    ToolAttributes,
)


def test_default_settings() -> None:
    config = TraceConfig()
    assert config.hide_llm_invocation_parameters == DEFAULT_HIDE_LLM_INVOCATION_PARAMETERS
    assert config.hide_llm_tools == DEFAULT_HIDE_LLM_TOOLS
    assert config.hide_inputs == DEFAULT_HIDE_INPUTS
    assert config.hide_outputs == DEFAULT_HIDE_OUTPUTS
    assert config.hide_input_messages == DEFAULT_HIDE_INPUT_MESSAGES
    assert config.hide_output_messages == DEFAULT_HIDE_OUTPUT_MESSAGES
    assert config.hide_input_images == DEFAULT_HIDE_INPUT_IMAGES
    assert config.hide_input_text == DEFAULT_HIDE_INPUT_TEXT
    assert config.hide_output_text == DEFAULT_HIDE_OUTPUT_TEXT
    assert config.hide_embedding_vectors == DEFAULT_HIDE_EMBEDDING_VECTORS
    assert config.hide_embeddings_vectors == DEFAULT_HIDE_EMBEDDINGS_VECTORS
    assert config.hide_embeddings_text == DEFAULT_HIDE_EMBEDDINGS_TEXT
    assert config.hide_prompts == DEFAULT_HIDE_PROMPTS
    assert config.hide_choices == DEFAULT_HIDE_CHOICES
    assert config.hide_retrieval_documents == DEFAULT_HIDE_RETRIEVAL_DOCUMENTS
    assert config.hide_reasoning == DEFAULT_HIDE_REASONING
    assert config.enable_genai_semconv == DEFAULT_ENABLE_GENAI_SEMCONV
    assert config.base64_image_max_length == DEFAULT_BASE64_IMAGE_MAX_LENGTH


def test_unparsable_bool_env_var_falls_back_to_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(OPENINFERENCE_HIDE_INPUTS, "yes")
    assert TraceConfig().hide_inputs == DEFAULT_HIDE_INPUTS


def test_unparsable_bool_env_var_falls_back_to_default_while_handling_an_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bare ``raise`` would re-raise the exception being handled here, so the
    fallback in ``_parse_value`` would never run."""
    monkeypatch.setenv(OPENINFERENCE_HIDE_INPUTS, "yes")
    try:
        raise KeyboardInterrupt("unrelated")
    except KeyboardInterrupt:
        assert TraceConfig().hide_inputs == DEFAULT_HIDE_INPUTS


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
@pytest.mark.parametrize("hide_embeddings_vectors", [False, True])
@pytest.mark.parametrize("hide_embeddings_text", [False, True])
@pytest.mark.parametrize("hide_prompts", [False, True])
@pytest.mark.parametrize("hide_choices", [False, True])
@pytest.mark.parametrize("hide_llm_tools", [False, True])
@pytest.mark.parametrize("enable_genai_semconv", [False, True])
@pytest.mark.parametrize("base64_image_max_length", [10_000])
def test_settings_from_env_vars_and_code(
    hide_inputs: bool,
    hide_outputs: bool,
    hide_input_messages: bool,
    hide_output_messages: bool,
    hide_input_images: bool,
    hide_input_text: bool,
    hide_output_text: bool,
    hide_embeddings_vectors: bool,
    hide_embeddings_text: bool,
    hide_prompts: bool,
    hide_choices: bool,
    hide_llm_tools: bool,
    enable_genai_semconv: bool,
    base64_image_max_length: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # First part of the test verifies that environment variables are read correctly
    monkeypatch.setenv(OPENINFERENCE_HIDE_INPUTS, str(hide_inputs))
    monkeypatch.setenv(OPENINFERENCE_HIDE_OUTPUTS, str(hide_outputs))
    monkeypatch.setenv(OPENINFERENCE_HIDE_INPUT_MESSAGES, str(hide_input_messages))
    monkeypatch.setenv(OPENINFERENCE_HIDE_OUTPUT_MESSAGES, str(hide_output_messages))
    monkeypatch.setenv(OPENINFERENCE_HIDE_INPUT_IMAGES, str(hide_input_images))
    monkeypatch.setenv(OPENINFERENCE_HIDE_PROMPTS, str(hide_prompts))
    monkeypatch.setenv(OPENINFERENCE_HIDE_CHOICES, str(hide_choices))
    monkeypatch.setenv(OPENINFERENCE_HIDE_RETRIEVAL_DOCUMENTS, str(hide_outputs))
    monkeypatch.setenv(OPENINFERENCE_HIDE_INPUT_TEXT, str(hide_input_text))
    monkeypatch.setenv(OPENINFERENCE_HIDE_OUTPUT_TEXT, str(hide_output_text))
    monkeypatch.setenv(OPENINFERENCE_HIDE_REASONING, str(hide_output_text))
    monkeypatch.setenv(OPENINFERENCE_HIDE_EMBEDDINGS_VECTORS, str(hide_embeddings_vectors))
    monkeypatch.setenv(OPENINFERENCE_HIDE_EMBEDDINGS_TEXT, str(hide_embeddings_text))
    monkeypatch.setenv(OPENINFERENCE_HIDE_LLM_TOOLS, str(hide_llm_tools))
    monkeypatch.setenv(OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH, str(base64_image_max_length))
    monkeypatch.setenv(OPENINFERENCE_ENABLE_GENAI_SEMCONV, str(enable_genai_semconv))

    config = TraceConfig()
    assert config.hide_inputs is parse_bool_from_env(OPENINFERENCE_HIDE_INPUTS)
    assert config.hide_outputs is parse_bool_from_env(OPENINFERENCE_HIDE_OUTPUTS)
    assert config.hide_input_messages is parse_bool_from_env(OPENINFERENCE_HIDE_INPUT_MESSAGES)
    assert config.hide_output_messages is parse_bool_from_env(OPENINFERENCE_HIDE_OUTPUT_MESSAGES)
    assert config.hide_input_images is parse_bool_from_env(OPENINFERENCE_HIDE_INPUT_IMAGES)
    assert config.hide_input_text is parse_bool_from_env(OPENINFERENCE_HIDE_INPUT_TEXT)
    assert config.hide_output_text is parse_bool_from_env(OPENINFERENCE_HIDE_OUTPUT_TEXT)
    assert config.hide_reasoning is parse_bool_from_env(OPENINFERENCE_HIDE_REASONING)
    assert config.hide_embeddings_vectors is parse_bool_from_env(
        OPENINFERENCE_HIDE_EMBEDDINGS_VECTORS
    )
    assert config.hide_embeddings_text is parse_bool_from_env(OPENINFERENCE_HIDE_EMBEDDINGS_TEXT)
    assert config.hide_prompts is parse_bool_from_env(OPENINFERENCE_HIDE_PROMPTS)
    assert config.hide_choices is parse_bool_from_env(OPENINFERENCE_HIDE_CHOICES)
    assert config.hide_retrieval_documents is parse_bool_from_env(
        OPENINFERENCE_HIDE_RETRIEVAL_DOCUMENTS
    )
    assert config.hide_llm_tools is parse_bool_from_env(OPENINFERENCE_HIDE_LLM_TOOLS)
    assert config.enable_genai_semconv is parse_bool_from_env(OPENINFERENCE_ENABLE_GENAI_SEMCONV)
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
    new_hide_reasoning = not hide_output_text
    new_hide_embeddings_vectors = not hide_embeddings_vectors
    new_hide_embeddings_text = not hide_embeddings_text
    new_hide_prompts = not hide_prompts
    new_hide_choices = not hide_choices
    new_hide_retrieval_documents = not hide_outputs
    new_hide_llm_tools = not hide_llm_tools
    new_enable_genai_semconv = not enable_genai_semconv
    config = TraceConfig(
        hide_inputs=new_hide_inputs,
        hide_outputs=new_hide_outputs,
        hide_input_messages=new_hide_input_messages,
        hide_output_messages=new_hide_output_messages,
        hide_input_images=new_hide_input_images,
        hide_input_text=new_hide_input_text,
        hide_output_text=new_hide_output_text,
        hide_reasoning=new_hide_reasoning,
        hide_embeddings_vectors=new_hide_embeddings_vectors,
        hide_embeddings_text=new_hide_embeddings_text,
        hide_prompts=new_hide_prompts,
        hide_choices=new_hide_choices,
        hide_retrieval_documents=new_hide_retrieval_documents,
        hide_llm_tools=new_hide_llm_tools,
        enable_genai_semconv=new_enable_genai_semconv,
        base64_image_max_length=new_base64_image_max_length,
    )
    assert config.hide_inputs is new_hide_inputs
    assert config.hide_outputs is new_hide_outputs
    assert config.hide_input_messages is new_hide_input_messages
    assert config.hide_output_messages is new_hide_output_messages
    assert config.hide_input_images is new_hide_input_images
    assert config.hide_input_text is new_hide_input_text
    assert config.hide_output_text is new_hide_output_text
    assert config.hide_reasoning is new_hide_reasoning
    assert config.hide_embeddings_vectors is new_hide_embeddings_vectors
    assert config.hide_embeddings_text is new_hide_embeddings_text
    assert config.hide_prompts is new_hide_prompts
    assert config.hide_choices is new_hide_choices
    assert config.hide_retrieval_documents is new_hide_retrieval_documents
    assert config.hide_llm_tools is new_hide_llm_tools
    assert config.enable_genai_semconv is new_enable_genai_semconv
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
            None,
            SpanAttributes.LLM_INVOCATION_PARAMETERS,
            "{api_key: '123'}",
            "{api_key: '123'}",
        ),
        (
            "hide_llm_tools",
            True,
            f"{SpanAttributes.LLM_TOOLS}.0.{ToolAttributes.TOOL_JSON_SCHEMA}",
            "{'type': 'function', 'function': {'name': 'get_weather'}}",
            None,
        ),
        (
            "hide_llm_tools",
            True,
            f"{SpanAttributes.LLM_TOOLS}.0.{ToolAttributes.TOOL_NAME}",
            "get_weather",
            None,
        ),
        (
            "hide_llm_tools",
            True,
            f"{SpanAttributes.LLM_TOOLS}.0.{ToolAttributes.TOOL_DESCRIPTION}",
            "Get the weather",
            None,
        ),
        (
            "hide_llm_tools",
            None,
            f"{SpanAttributes.LLM_TOOLS}.0.{ToolAttributes.TOOL_JSON_SCHEMA}",
            "{'type': 'function', 'function': {'name': 'get_weather'}}",
            "{'type': 'function', 'function': {'name': 'get_weather'}}",
        ),
        (
            "hide_inputs",
            True,
            f"{SpanAttributes.LLM_TOOLS}.0.{ToolAttributes.TOOL_JSON_SCHEMA}",
            "{'type': 'function', 'function': {'name': 'get_weather'}}",
            None,
        ),
        (
            "hide_inputs",
            True,
            f"{SpanAttributes.LLM_TOOLS}.0.{ToolAttributes.TOOL_NAME}",
            "get_weather",
            None,
        ),
        (
            "hide_inputs",
            True,
            f"{SpanAttributes.LLM_TOOLS}.0.{ToolAttributes.TOOL_DESCRIPTION}",
            "Get the weather",
            None,
        ),
        (
            "hide_retrieval_documents",
            True,
            f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.0.{DocumentAttributes.DOCUMENT_CONTENT}",
            "sensitive document text",
            REDACTED_VALUE,
        ),
        (
            "hide_retrieval_documents",
            True,
            f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.0.{DocumentAttributes.DOCUMENT_METADATA}",
            "{'source': 'secret.pdf'}",
            REDACTED_VALUE,
        ),
        (
            "hide_retrieval_documents",
            True,
            f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.0.{DocumentAttributes.DOCUMENT_ID}",
            "doc-1",
            "doc-1",
        ),
        (
            "hide_retrieval_documents",
            None,
            f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.0.{DocumentAttributes.DOCUMENT_CONTENT}",
            "sensitive document text",
            "sensitive document text",
        ),
        (
            "hide_outputs",
            True,
            f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.0.{DocumentAttributes.DOCUMENT_CONTENT}",
            "sensitive document text",
            REDACTED_VALUE,
        ),
    ],
)
def test_trace_config(
    param: str,
    param_value: Optional[bool],
    attr_key: str,
    attr_value: Any,
    expected_value: Any,
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    config = TraceConfig(**({} if param_value is None else {param: param_value}))  # type: ignore[arg-type]
    tracer = OITracer(tracer_provider.get_tracer(__name__), config=config)
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


@pytest.mark.parametrize(
    "use_deprecated",
    [True, False],
)
def test_embedding_vectors_backwards_compatibility(
    use_deprecated: bool,
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """Test that both hide_embedding_vectors (deprecated) and hide_embeddings_vectors work."""
    from openinference.semconv.trace import EmbeddingAttributes, SpanAttributes

    if use_deprecated:
        config = TraceConfig(hide_embedding_vectors=True)
    else:
        config = TraceConfig(hide_embeddings_vectors=True)

    tracer = OITracer(tracer_provider.get_tracer(__name__), config=config)
    tracer.start_span(
        "test",
        attributes={
            f"{SpanAttributes.EMBEDDING_EMBEDDINGS}.0.{EmbeddingAttributes.EMBEDDING_VECTOR}": (
                1.0,
                2.0,
                3.0,
            )
        },
    ).end()
    span = in_memory_span_exporter.get_finished_spans()[0]
    assert span.attributes is not None
    assert (
        span.attributes.get(
            f"{SpanAttributes.EMBEDDING_EMBEDDINGS}.0.{EmbeddingAttributes.EMBEDDING_VECTOR}"
        )
        == REDACTED_VALUE
    )


@pytest.mark.parametrize(
    "use_deprecated",
    [True, False],
)
def test_embedding_vectors_env_var_backwards_compatibility(
    use_deprecated: bool,
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that both OPENINFERENCE_HIDE_EMBEDDING_VECTORS (deprecated) and OPENINFERENCE_HIDE_EMBEDDINGS_VECTORS work."""
    from openinference.semconv.trace import EmbeddingAttributes, SpanAttributes

    if use_deprecated:
        monkeypatch.setenv(OPENINFERENCE_HIDE_EMBEDDING_VECTORS, "True")
    else:
        monkeypatch.setenv(OPENINFERENCE_HIDE_EMBEDDINGS_VECTORS, "True")

    config = TraceConfig()
    tracer = OITracer(tracer_provider.get_tracer(__name__), config=config)
    tracer.start_span(
        "test",
        attributes={
            f"{SpanAttributes.EMBEDDING_EMBEDDINGS}.0.{EmbeddingAttributes.EMBEDDING_VECTOR}": (
                1.0,
                2.0,
                3.0,
            )
        },
    ).end()
    span = in_memory_span_exporter.get_finished_spans()[0]
    assert span.attributes is not None
    assert (
        span.attributes.get(
            f"{SpanAttributes.EMBEDDING_EMBEDDINGS}.0.{EmbeddingAttributes.EMBEDDING_VECTOR}"
        )
        == REDACTED_VALUE
    )


@pytest.mark.parametrize(
    "messages_prefix",
    [SpanAttributes.LLM_INPUT_MESSAGES, SpanAttributes.LLM_OUTPUT_MESSAGES],
)
def test_base64_image_max_length_applies_to_input_and_output_messages(
    messages_prefix: str,
) -> None:
    """base64_image_max_length must redact oversized base64 image URLs in both
    input and output messages. Multimodal / image-generation models return
    base64 images in output messages; previously the truncation guard only
    matched LLM_INPUT_MESSAGES, so an oversized base64 image in an output
    message was stored in full, ignoring the configured limit."""
    config = TraceConfig(base64_image_max_length=100)
    key = (
        f"{messages_prefix}.0.message.contents.0."
        f"{MessageContentAttributes.MESSAGE_CONTENT_IMAGE}.{ImageAttributes.IMAGE_URL}"
    )

    over_limit = "data:image/png;base64," + "A" * 200
    under_limit = "data:image/png;base64," + "A" * 20

    assert config.mask(key, over_limit) == REDACTED_VALUE
    # A base64 image under the limit must pass through untouched.
    assert config.mask(key, under_limit) == under_limit


@pytest.mark.parametrize(
    ("images_prefix", "other_prefix", "hiding_config"),
    [
        (SpanAttributes.INPUT_IMAGES, SpanAttributes.OUTPUT_IMAGES, {"hide_inputs": True}),
        (SpanAttributes.INPUT_IMAGES, SpanAttributes.OUTPUT_IMAGES, {"hide_input_images": True}),
        (SpanAttributes.OUTPUT_IMAGES, SpanAttributes.INPUT_IMAGES, {"hide_outputs": True}),
    ],
)
def test_hiding_applies_to_span_level_images(
    images_prefix: str,
    other_prefix: str,
    hiding_config: Dict[str, Any],
) -> None:
    """The span-level image namespaces obey the same hiding controls as message
    images, so a configured hide setting cannot be bypassed by recording an image
    outside the LLM message structure."""
    config = TraceConfig(**hiding_config)
    url = "https://example.com/a.png"
    url_key = f"{images_prefix}.0.{ImageAttributes.IMAGE_URL}"

    assert config.mask(url_key, "data:image/png;base64,iVBORw0KGgo=") is None
    # The whole namespace is removed, not only the image.url leaves.
    assert config.mask(images_prefix, "[]") is None
    # The setting is scoped to its own side: the other namespace is untouched.
    other_key = f"{other_prefix}.0.{ImageAttributes.IMAGE_URL}"
    assert config.mask(other_key, url) == url
    # A sibling namespace that merely shares the prefix string is not matched.
    assert config.mask(f"{images_prefix}et.0.{ImageAttributes.IMAGE_URL}", url) == url
    # Without the hide setting the same keys pass through.
    assert TraceConfig().mask(url_key, url) == url


@pytest.mark.parametrize(
    "hiding_config",
    [{"hide_input_messages": True}, {"hide_output_messages": True}],
)
def test_hiding_messages_leaves_span_level_images_visible(
    hiding_config: Dict[str, Any],
) -> None:
    """input.images / output.images live outside the message structure, so the
    message-only hide settings do not reach them."""
    config = TraceConfig(**hiding_config)
    url = "https://example.com/a.png"
    for prefix in (SpanAttributes.INPUT_IMAGES, SpanAttributes.OUTPUT_IMAGES):
        assert config.mask(f"{prefix}.0.{ImageAttributes.IMAGE_URL}", url) == url


@pytest.mark.parametrize(
    "images_prefix",
    [SpanAttributes.INPUT_IMAGES, SpanAttributes.OUTPUT_IMAGES],
)
def test_base64_image_max_length_applies_to_span_level_images(
    images_prefix: str,
) -> None:
    """The size limit follows the image.url leaf, so an oversized base64 payload is
    redacted whether it sits under a message content item or at the span level."""
    config = TraceConfig(base64_image_max_length=100)
    key = f"{images_prefix}.0.{ImageAttributes.IMAGE_URL}"

    over_limit = "data:image/png;base64," + "A" * 200
    under_limit = "data:image/png;base64," + "A" * 20

    assert config.mask(key, over_limit) == REDACTED_VALUE
    assert config.mask(key, under_limit) == under_limit
    # A plain URI is never subject to the base64 budget.
    assert config.mask(key, "https://example.com/a.png") == "https://example.com/a.png"


_REASONING_INPUT_MESSAGES: List[Message] = [
    {"role": "user", "contents": [{"type": "text", "text": "What is 2+2?"}]},
    {
        # A prior assistant turn replayed as input, as reasoning models require.
        "role": "assistant",
        "contents": [
            {"type": "reasoning", "text": "earlier thoughts", "signature": "sig-0"},
            {"type": "text", "text": "4"},
        ],
    },
    {"role": "user", "contents": [{"type": "text", "text": "And 3+3?"}]},
]
_REASONING_OUTPUT_MESSAGES: List[Message] = [
    {
        "role": "assistant",
        "contents": [
            {"type": "reasoning", "text": "3+3 is six", "signature": "sig-1"},
            {"type": "text", "text": "6"},
            {"type": "reasoning", "data": "opaque-redacted-thinking"},
            {"type": "image", "image": {"url": "https://example.com/six.png"}},
            {"type": "reasoning", "encrypted_content": "gAAAA"},
        ],
    },
]


def _message_contents(
    attributes: Mapping[str, Any], messages_key: str, message_index: int
) -> List[Dict[str, Any]]:
    """Unflattens the content parts of one message, failing on index gaps."""
    prefix = f"{messages_key}.{message_index}.{MessageAttributes.MESSAGE_CONTENTS}."
    parts: Dict[int, Dict[str, Any]] = {}
    for key, value in attributes.items():
        if key.startswith(prefix):
            index, _, suffix = key[len(prefix) :].partition(".")
            parts.setdefault(int(index), {})[suffix] = value
    assert sorted(parts) == list(range(len(parts))), f"non-contiguous indices: {sorted(parts)}"
    return [parts[index] for index in range(len(parts))]


def _end_llm_span(
    config: TraceConfig,
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> Mapping[str, Any]:
    tracer = OITracer(tracer_provider.get_tracer(__name__), config=config)
    span = tracer.start_span("llm", openinference_span_kind="llm")
    span.set_attributes(
        get_llm_attributes(
            input_messages=_REASONING_INPUT_MESSAGES,
            output_messages=_REASONING_OUTPUT_MESSAGES,
        )
    )
    span.end()
    attributes = in_memory_span_exporter.get_finished_spans()[0].attributes
    assert attributes is not None
    return attributes


_TYPE = MessageContentAttributes.MESSAGE_CONTENT_TYPE
_TEXT = MessageContentAttributes.MESSAGE_CONTENT_TEXT
_IMAGE_URL = f"{MessageContentAttributes.MESSAGE_CONTENT_IMAGE}.{ImageAttributes.IMAGE_URL}"


def test_hide_reasoning_from_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(OPENINFERENCE_HIDE_REASONING, "true")
    assert TraceConfig().hide_reasoning is True
    # A value passed in code takes precedence over the environment variable.
    assert TraceConfig(hide_reasoning=False).hide_reasoning is False


def test_hide_reasoning_drops_reasoning_parts_and_keeps_the_answer(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """Reasoning parts are dropped from output messages and from prior turns replayed
    as input messages; the remaining parts are re-indexed without gaps."""
    attributes = _end_llm_span(
        TraceConfig(hide_reasoning=True), tracer_provider, in_memory_span_exporter
    )
    output_messages = SpanAttributes.LLM_OUTPUT_MESSAGES
    input_messages = SpanAttributes.LLM_INPUT_MESSAGES
    assert _message_contents(attributes, output_messages, 0) == [
        {_TYPE: "text", _TEXT: "6"},
        {_TYPE: "image", _IMAGE_URL: "https://example.com/six.png"},
    ]
    assert _message_contents(attributes, input_messages, 1) == [{_TYPE: "text", _TEXT: "4"}]
    assert _message_contents(attributes, input_messages, 0) == [
        {_TYPE: "text", _TEXT: "What is 2+2?"}
    ]
    assert _message_contents(attributes, input_messages, 2) == [{_TYPE: "text", _TEXT: "And 3+3?"}]
    assert attributes[f"{output_messages}.0.{MessageAttributes.MESSAGE_ROLE}"] == "assistant"
    assert "reasoning" not in attributes.values()
    for leaked in ("3+3 is six", "earlier thoughts", "opaque-redacted-thinking", "gAAAA"):
        assert leaked not in attributes.values()


@pytest.mark.parametrize("hide_outputs", [False, True])
def test_hide_reasoning_only_filters_message_contents(
    hide_outputs: bool,
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """Reasoning recorded outside the message contents, such as a raw provider response
    in output.value, is not filtered; hide_outputs is what removes it."""
    raw_response = '{"reasoning": "3+3 is six", "answer": "6"}'
    tracer = OITracer(
        tracer_provider.get_tracer(__name__),
        config=TraceConfig(hide_reasoning=True, hide_outputs=hide_outputs),
    )
    span = tracer.start_span("llm", openinference_span_kind="llm")
    span.set_attributes(
        {
            SpanAttributes.OUTPUT_VALUE: raw_response,
            SpanAttributes.OUTPUT_MIME_TYPE: "application/json",
        }
    )
    span.end()
    attributes = in_memory_span_exporter.get_finished_spans()[0].attributes
    assert attributes is not None
    if hide_outputs:
        assert attributes[SpanAttributes.OUTPUT_VALUE] == REDACTED_VALUE
    else:
        assert attributes[SpanAttributes.OUTPUT_VALUE] == raw_response


def test_hide_reasoning_is_off_by_default(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    attributes = _end_llm_span(TraceConfig(), tracer_provider, in_memory_span_exporter)
    assert dict(attributes) == {
        SpanAttributes.OPENINFERENCE_SPAN_KIND: "LLM",
        **get_llm_attributes(
            input_messages=_REASONING_INPUT_MESSAGES,
            output_messages=_REASONING_OUTPUT_MESSAGES,
        ),
    }


def test_hide_reasoning_does_not_depend_on_attribute_order(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """Instrumentors that flatten messages themselves may set a part's text before its
    type; the reasoning text must still be dropped."""
    tracer = OITracer(tracer_provider.get_tracer(__name__), TraceConfig(hide_reasoning=True))
    prefix = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENTS}"
    span = tracer.start_span("llm", openinference_span_kind="llm")
    span.set_attribute(f"{prefix}.0.{_TEXT}", "secret thoughts")
    span.set_attribute(f"{prefix}.1.{_TEXT}", "answer")
    span.set_attribute(f"{prefix}.1.{_TYPE}", "text")
    span.set_attribute(f"{prefix}.0.{_TYPE}", "reasoning")
    span.end()
    attributes = in_memory_span_exporter.get_finished_spans()[0].attributes or {}
    assert _message_contents(attributes, SpanAttributes.LLM_OUTPUT_MESSAGES, 0) == [
        {_TYPE: "text", _TEXT: "answer"}
    ]


def test_hide_reasoning_applies_to_start_span_attributes(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """Attributes passed to start_span reach the SDK span (and the sampler) directly,
    so they are filtered there too."""
    tracer = OITracer(tracer_provider.get_tracer(__name__), TraceConfig(hide_reasoning=True))
    tracer.start_span(
        "llm",
        attributes=dict(get_llm_attributes(output_messages=_REASONING_OUTPUT_MESSAGES)),
    ).end()
    attributes = in_memory_span_exporter.get_finished_spans()[0].attributes or {}
    assert _message_contents(attributes, SpanAttributes.LLM_OUTPUT_MESSAGES, 0) == [
        {_TYPE: "text", _TEXT: "6"},
        {_TYPE: "image", _IMAGE_URL: "https://example.com/six.png"},
    ]


@pytest.mark.parametrize(
    ("hiding_config", "expected_answer"),
    [
        ({"hide_output_text": True}, {_TYPE: "text", _TEXT: REDACTED_VALUE}),
        ({"hide_output_messages": True}, None),
        ({"hide_outputs": True}, None),
    ],
)
def test_hide_reasoning_composes_with_output_hiding(
    hiding_config: Dict[str, Any],
    expected_answer: Optional[Dict[str, Any]],
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """The other hide settings still apply to the parts that remain, at their new
    indices."""
    attributes = _end_llm_span(
        TraceConfig(hide_reasoning=True, **hiding_config),
        tracer_provider,
        in_memory_span_exporter,
    )
    contents = _message_contents(attributes, SpanAttributes.LLM_OUTPUT_MESSAGES, 0)
    if expected_answer is None:
        assert contents == []
    else:
        assert contents[0] == expected_answer

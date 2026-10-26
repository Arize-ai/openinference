import os
from contextlib import suppress
from random import random
from typing import Any, Dict, List, Optional

import pytest
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace import SpanLimits
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import TracerProvider, use_span
from opentelemetry.util.types import AttributeValue

from openinference.instrumentation import Blob, OITracer, TraceConfig
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
    OPENINFERENCE_HIDE_RETRIEVAL_DOCUMENTS,
    REDACTED_VALUE,
)
from openinference.semconv.trace import (
    DocumentAttributes,
    ImageAttributes,
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


def _message_document_key(messages_prefix: str, leaf: str) -> str:
    return (
        f"{messages_prefix}.0.message.contents.1."
        f"{MessageContentAttributes.MESSAGE_CONTENT_DOCUMENT}.{leaf}"
    )


_ALL_DOCUMENT_URL_KEYS = [
    _message_document_key(SpanAttributes.LLM_INPUT_MESSAGES, DocumentAttributes.DOCUMENT_URL),
    _message_document_key(SpanAttributes.LLM_OUTPUT_MESSAGES, DocumentAttributes.DOCUMENT_URL),
    f"{SpanAttributes.INPUT_DOCUMENTS}.0.{DocumentAttributes.DOCUMENT_URL}",
    f"{SpanAttributes.OUTPUT_DOCUMENTS}.0.{DocumentAttributes.DOCUMENT_URL}",
]


@pytest.mark.parametrize(
    ("documents_prefix", "other_prefix", "hiding_config"),
    [
        (SpanAttributes.INPUT_DOCUMENTS, SpanAttributes.OUTPUT_DOCUMENTS, {"hide_inputs": True}),
        (SpanAttributes.OUTPUT_DOCUMENTS, SpanAttributes.INPUT_DOCUMENTS, {"hide_outputs": True}),
    ],
)
def test_hiding_applies_to_span_level_documents(
    documents_prefix: str,
    other_prefix: str,
    hiding_config: Dict[str, Any],
) -> None:
    config = TraceConfig(**hiding_config)
    url = "https://example.com/report.pdf"
    for leaf, value in [
        (DocumentAttributes.DOCUMENT_URL, url),
        (DocumentAttributes.DOCUMENT_CONTENT, "quarterly numbers"),
        (DocumentAttributes.DOCUMENT_MIME_TYPE, "application/pdf"),
        (DocumentAttributes.DOCUMENT_NAME, "report.pdf"),
        (DocumentAttributes.DOCUMENT_ID, "doc-1"),
    ]:
        assert config.mask(f"{documents_prefix}.0.{leaf}", value) is None
    assert config.mask(documents_prefix, "[]") is None
    other_key = f"{other_prefix}.0.{DocumentAttributes.DOCUMENT_URL}"
    assert config.mask(other_key, url) == url
    assert config.mask(f"{documents_prefix}et.0.{DocumentAttributes.DOCUMENT_URL}", url) == url
    assert TraceConfig().mask(f"{documents_prefix}.0.{DocumentAttributes.DOCUMENT_URL}", url) == url


@pytest.mark.parametrize(
    ("messages_prefix", "hiding_config"),
    [
        (SpanAttributes.LLM_INPUT_MESSAGES, {"hide_inputs": True}),
        (SpanAttributes.LLM_INPUT_MESSAGES, {"hide_input_messages": True}),
        (SpanAttributes.LLM_OUTPUT_MESSAGES, {"hide_outputs": True}),
        (SpanAttributes.LLM_OUTPUT_MESSAGES, {"hide_output_messages": True}),
    ],
)
def test_hiding_messages_removes_message_content_documents(
    messages_prefix: str,
    hiding_config: Dict[str, Any],
) -> None:
    config = TraceConfig(**hiding_config)
    for leaf in (DocumentAttributes.DOCUMENT_URL, DocumentAttributes.DOCUMENT_CONTENT):
        assert config.mask(_message_document_key(messages_prefix, leaf), "x") is None


@pytest.mark.parametrize(
    ("messages_prefix", "other_prefix", "hiding_config"),
    [
        (
            SpanAttributes.LLM_INPUT_MESSAGES,
            SpanAttributes.LLM_OUTPUT_MESSAGES,
            {"hide_input_text": True},
        ),
        (
            SpanAttributes.LLM_OUTPUT_MESSAGES,
            SpanAttributes.LLM_INPUT_MESSAGES,
            {"hide_output_text": True},
        ),
    ],
)
def test_hide_text_redacts_message_document_content(
    messages_prefix: str,
    other_prefix: str,
    hiding_config: Dict[str, Any],
) -> None:
    config = TraceConfig(**hiding_config)
    content = "the contract says"
    content_key = _message_document_key(messages_prefix, DocumentAttributes.DOCUMENT_CONTENT)

    assert config.mask(content_key, content) == REDACTED_VALUE
    other_key = _message_document_key(other_prefix, DocumentAttributes.DOCUMENT_CONTENT)
    assert config.mask(other_key, content) == content
    name_key = _message_document_key(messages_prefix, DocumentAttributes.DOCUMENT_NAME)
    assert config.mask(name_key, "contract.txt") == "contract.txt"
    assert TraceConfig().mask(content_key, content) == content


class _RecordingUploader:
    def __init__(self, uri: str = "https://storage.example.com/blob") -> None:
        self.uri = uri
        self.uploads: List[Blob] = []

    def upload(self, blob: Blob) -> Optional[str]:
        self.uploads.append(blob)
        return self.uri

    def shutdown(self, timeout_sec: float = 10.0) -> None:
        pass


_OVERSIZED_PDF = "data:application/pdf;base64," + "A" * 200


@pytest.mark.parametrize("key", _ALL_DOCUMENT_URL_KEYS)
def test_oversized_document_url_is_externalized(key: str) -> None:
    uploader = _RecordingUploader()
    config = TraceConfig(base64_image_max_length=100, blob_uploader=uploader)

    assert config.mask(key, _OVERSIZED_PDF) == "https://storage.example.com/blob"
    assert len(uploader.uploads) == 1
    blob = uploader.uploads[0]
    assert blob.mime_type == "application/pdf"
    assert blob.modality == "document"
    assert blob.attribute_key == key


@pytest.mark.parametrize("key", _ALL_DOCUMENT_URL_KEYS)
def test_oversized_document_url_falls_back_to_redaction(key: str) -> None:
    assert TraceConfig(base64_image_max_length=100).mask(key, _OVERSIZED_PDF) == REDACTED_VALUE

    uploader = _RecordingUploader()
    config = TraceConfig(base64_image_max_length=100, blob_uploader=uploader)
    assert config.mask(key, _OVERSIZED_PDF, externalize=False) == REDACTED_VALUE
    assert uploader.uploads == []

    invalid = TraceConfig(
        base64_image_max_length=100, blob_uploader=_RecordingUploader(uri="not a uri")
    )
    assert invalid.mask(key, _OVERSIZED_PDF) == REDACTED_VALUE


@pytest.mark.parametrize("key", _ALL_DOCUMENT_URL_KEYS)
def test_small_or_plain_document_urls_are_not_uploaded(key: str) -> None:
    uploader = _RecordingUploader()
    config = TraceConfig(base64_image_max_length=100, blob_uploader=uploader)
    under_limit = "data:application/pdf;base64," + "A" * 20

    assert config.mask(key, under_limit) == under_limit
    for plain in (
        "https://example.com/" + "a" * 200 + ".pdf",
        "gs://bucket/" + "a" * 200 + ".pdf",
        "s3://bucket/" + "a" * 200 + ".pdf",
    ):
        assert config.mask(key, plain) == plain
    assert uploader.uploads == []


@pytest.mark.parametrize(
    ("key", "hiding_config"),
    [
        (
            f"{SpanAttributes.INPUT_DOCUMENTS}.0.{DocumentAttributes.DOCUMENT_URL}",
            {"hide_inputs": True},
        ),
        (
            _message_document_key(
                SpanAttributes.LLM_INPUT_MESSAGES, DocumentAttributes.DOCUMENT_URL
            ),
            {"hide_inputs": True},
        ),
        (
            _message_document_key(
                SpanAttributes.LLM_INPUT_MESSAGES, DocumentAttributes.DOCUMENT_URL
            ),
            {"hide_input_messages": True},
        ),
    ],
)
def test_hidden_document_is_not_uploaded(key: str, hiding_config: Dict[str, Any]) -> None:
    uploader = _RecordingUploader()
    config = TraceConfig(base64_image_max_length=100, blob_uploader=uploader, **hiding_config)

    assert config.mask(key, _OVERSIZED_PDF) is None
    assert uploader.uploads == []


def test_document_rules_leave_images_unchanged() -> None:
    uploader = _RecordingUploader()
    config = TraceConfig(base64_image_max_length=100, blob_uploader=uploader, hide_input_text=True)
    image_key = (
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.message.contents.0."
        f"{MessageContentAttributes.MESSAGE_CONTENT_IMAGE}.{ImageAttributes.IMAGE_URL}"
    )
    oversized_image = "data:image/png;base64," + "A" * 200
    non_image_data_uri = "data:application/pdf;base64," + "A" * 200

    assert config.mask(image_key, oversized_image) == "https://storage.example.com/blob"
    assert len(uploader.uploads) == 1
    assert config.mask(image_key, non_image_data_uri) == non_image_data_uri

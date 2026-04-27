import json
from typing import Any, Dict, Mapping

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import TracerProvider

from openinference.instrumentation import (
    REDACTED_VALUE,
    OITracer,
    TraceConfig,
    get_embedding_attributes,
    get_llm_attributes,
    get_retriever_attributes,
    using_session,
)
from openinference.instrumentation._genai_attributes import (
    GenAIAttributes,
    GenAIModalityValues,
    GenAIOperationNameValues,
    GenAIProviderNameValues,
    GenAIToolTypeValues,
)
from openinference.instrumentation._genai_conversion import (
    get_genai_attributes,
    get_genai_base_attributes,
)
from openinference.semconv.trace import (
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)


def _load_json_attribute(attributes: Mapping[str, Any], key: str) -> Any:
    value = attributes[key]
    assert isinstance(value, str)
    return json.loads(value)


@pytest.mark.parametrize(
    ("provider", "system", "expected"),
    [
        ("openai", "openai", GenAIProviderNameValues.OPENAI.value),
        ("azure", "openai", GenAIProviderNameValues.AZURE_AI_OPENAI.value),
        ("aws", "anthropic", GenAIProviderNameValues.AWS_BEDROCK.value),
        ("google", "vertexai", GenAIProviderNameValues.GCP_VERTEX_AI.value),
        ("mistralai", "mistralai", GenAIProviderNameValues.MISTRAL_AI.value),
        ("xai", "xai", GenAIProviderNameValues.X_AI.value),
    ],
)
def test_get_genai_base_attributes_maps_provider_names(
    provider: str,
    system: str,
    expected: str,
) -> None:
    genai_attributes = get_genai_base_attributes(
        {
            SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.LLM.value,
            SpanAttributes.LLM_PROVIDER: provider,
            SpanAttributes.LLM_SYSTEM: system,
        }
    )
    assert genai_attributes[GenAIAttributes.GEN_AI_PROVIDER_NAME] == expected


def test_get_genai_attributes_maps_llm_chat_attributes() -> None:
    openinference_attributes = {
        SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.LLM.value,
        SpanAttributes.LLM_FINISH_REASON: "tool_calls",
        SpanAttributes.SESSION_ID: "conversation-123",
        **get_llm_attributes(
            provider="azure",
            system="openai",
            model_name="gpt-4o-mini",
            invocation_parameters={
                "temperature": 0.5,
                "max_tokens": 128,
                "stop": ["DONE"],
                "n": 2,
                "stream": True,
            },
            input_messages=[
                {
                    "role": "user",
                    "content": "Describe this image",
                    "contents": [
                        {"type": "text", "text": "Describe this image"},
                        {"type": "image", "image": {"url": "https://example.com/cat.png"}},
                    ],
                }
            ],
            output_messages=[
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "Denver"}',
                            },
                        }
                    ],
                }
            ],
            token_count={
                "prompt": 10,
                "completion": 4,
                "total": 14,
                "prompt_details": {"cache_read": 2, "cache_write": 1},
            },
            tools=[
                {
                    "json_schema": {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get the weather",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"],
                            },
                        },
                    }
                }
            ],
        ),
    }

    genai_attributes = get_genai_attributes(openinference_attributes)

    assert (
        genai_attributes[GenAIAttributes.GEN_AI_OPERATION_NAME]
        == GenAIOperationNameValues.CHAT.value
    )
    assert (
        genai_attributes[GenAIAttributes.GEN_AI_PROVIDER_NAME]
        == GenAIProviderNameValues.AZURE_AI_OPENAI.value
    )
    assert genai_attributes[GenAIAttributes.GEN_AI_CONVERSATION_ID] == "conversation-123"
    assert genai_attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "gpt-4o-mini"
    assert genai_attributes[GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE] == 0.5
    assert genai_attributes[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 128
    assert genai_attributes[GenAIAttributes.GEN_AI_REQUEST_STOP_SEQUENCES] == ("DONE",)
    assert genai_attributes[GenAIAttributes.GEN_AI_REQUEST_CHOICE_COUNT] == 2
    assert genai_attributes[GenAIAttributes.GEN_AI_REQUEST_STREAM] is True
    assert genai_attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 10
    assert genai_attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 4
    assert genai_attributes[GenAIAttributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS] == 2
    assert genai_attributes[GenAIAttributes.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS] == 1
    assert genai_attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == ("tool_call",)

    input_messages = _load_json_attribute(genai_attributes, GenAIAttributes.GEN_AI_INPUT_MESSAGES)
    assert input_messages == [
        {
            "role": "user",
            "parts": [
                {"type": "text", "content": "Describe this image"},
                {
                    "type": "uri",
                    "modality": GenAIModalityValues.IMAGE.value,
                    "uri": "https://example.com/cat.png",
                },
            ],
        }
    ]

    output_messages = _load_json_attribute(genai_attributes, GenAIAttributes.GEN_AI_OUTPUT_MESSAGES)
    assert output_messages == [
        {
            "role": "assistant",
            "parts": [
                {
                    "type": "tool_call",
                    "id": "call-1",
                    "name": "get_weather",
                    "arguments": {"city": "Denver"},
                }
            ],
            "finish_reason": "tool_call",
        }
    ]

    tool_definitions = _load_json_attribute(
        genai_attributes, GenAIAttributes.GEN_AI_TOOL_DEFINITIONS
    )
    assert tool_definitions == [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get the weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
    ]


def test_get_genai_attributes_maps_text_completion_prompts_and_choices() -> None:
    openinference_attributes: Dict[str, Any] = {
        SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.LLM.value,
        SpanAttributes.LLM_FINISH_REASON: "stop",
        SpanAttributes.LLM_PROMPTS: ("What is OpenTelemetry?",),
        f"{SpanAttributes.LLM_CHOICES}.0.completion.text": "OpenTelemetry is an observability framework.",
    }

    genai_attributes = get_genai_attributes(openinference_attributes)

    assert (
        genai_attributes[GenAIAttributes.GEN_AI_OPERATION_NAME]
        == GenAIOperationNameValues.TEXT_COMPLETION.value
    )
    assert _load_json_attribute(genai_attributes, GenAIAttributes.GEN_AI_INPUT_MESSAGES) == [
        {
            "role": "user",
            "parts": [{"type": "text", "content": "What is OpenTelemetry?"}],
        }
    ]
    assert _load_json_attribute(genai_attributes, GenAIAttributes.GEN_AI_OUTPUT_MESSAGES) == [
        {
            "role": "assistant",
            "parts": [
                {
                    "type": "text",
                    "content": "OpenTelemetry is an observability framework.",
                }
            ],
            "finish_reason": "stop",
        }
    ]


def test_get_genai_attributes_maps_tool_span_attributes() -> None:
    openinference_attributes = {
        SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.TOOL.value,
        SpanAttributes.TOOL_NAME: "lookup_weather",
        SpanAttributes.TOOL_DESCRIPTION: "Look up the weather",
        SpanAttributes.TOOL_ID: "call-123",
        SpanAttributes.TOOL_PARAMETERS: '{"city": "Denver"}',
        SpanAttributes.OUTPUT_VALUE: '{"forecast": "sunny"}',
        SpanAttributes.OUTPUT_MIME_TYPE: OpenInferenceMimeTypeValues.JSON.value,
    }

    genai_attributes = get_genai_attributes(openinference_attributes)

    assert (
        genai_attributes[GenAIAttributes.GEN_AI_OPERATION_NAME]
        == GenAIOperationNameValues.EXECUTE_TOOL.value
    )
    assert genai_attributes[GenAIAttributes.GEN_AI_TOOL_NAME] == "lookup_weather"
    assert genai_attributes[GenAIAttributes.GEN_AI_TOOL_DESCRIPTION] == "Look up the weather"
    assert genai_attributes[GenAIAttributes.GEN_AI_TOOL_CALL_ID] == "call-123"
    assert genai_attributes[GenAIAttributes.GEN_AI_TOOL_TYPE] == GenAIToolTypeValues.FUNCTION.value
    assert _load_json_attribute(genai_attributes, GenAIAttributes.GEN_AI_TOOL_CALL_ARGUMENTS) == {
        "city": "Denver"
    }
    assert _load_json_attribute(genai_attributes, GenAIAttributes.GEN_AI_TOOL_CALL_RESULT) == {
        "forecast": "sunny"
    }


def test_get_genai_attributes_maps_retrieval_attributes() -> None:
    openinference_attributes = {
        SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.RETRIEVER.value,
        SpanAttributes.INPUT_VALUE: "what is oidc?",
        SpanAttributes.INPUT_MIME_TYPE: OpenInferenceMimeTypeValues.TEXT.value,
        **get_retriever_attributes(
            documents=[
                {
                    "id": "doc-1",
                    "content": "OIDC is an identity layer on top of OAuth 2.0.",
                    "metadata": {"source": "docs"},
                    "score": 0.9,
                }
            ]
        ),
    }

    genai_attributes = get_genai_attributes(openinference_attributes)

    assert (
        genai_attributes[GenAIAttributes.GEN_AI_OPERATION_NAME]
        == GenAIOperationNameValues.RETRIEVAL.value
    )
    assert genai_attributes[GenAIAttributes.GEN_AI_RETRIEVAL_QUERY_TEXT] == "what is oidc?"
    assert _load_json_attribute(genai_attributes, GenAIAttributes.GEN_AI_RETRIEVAL_DOCUMENTS) == [
        {
            "id": "doc-1",
            "content": "OIDC is an identity layer on top of OAuth 2.0.",
            "metadata": {"source": "docs"},
            "score": 0.9,
        }
    ]


def test_get_genai_attributes_maps_embedding_attributes() -> None:
    openinference_attributes = {
        SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.EMBEDDING.value,
        SpanAttributes.EMBEDDING_INVOCATION_PARAMETERS: json.dumps({"encoding_format": "float"}),
        **get_embedding_attributes(
            model_name="text-embedding-3-large",
            embeddings=[{"text": "hello", "vector": [0.1, 0.2, 0.3]}],
        ),
    }

    genai_attributes = get_genai_attributes(openinference_attributes)

    assert (
        genai_attributes[GenAIAttributes.GEN_AI_OPERATION_NAME]
        == GenAIOperationNameValues.EMBEDDINGS.value
    )
    assert genai_attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "text-embedding-3-large"
    assert genai_attributes[GenAIAttributes.GEN_AI_REQUEST_ENCODING_FORMATS] == ("float",)
    assert genai_attributes[GenAIAttributes.GEN_AI_EMBEDDINGS_DIMENSION_COUNT] == 3


def test_oi_tracer_emits_genai_semconv_when_enabled(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    tracer = OITracer(
        tracer_provider.get_tracer(__name__),
        TraceConfig(enable_genai_semconv=True),
    )

    with tracer.start_as_current_span(
        "llm-span",
        openinference_span_kind="llm",
    ) as span:
        span.set_attributes(
            get_llm_attributes(
                provider="openai",
                system="openai",
                model_name="gpt-4o-mini",
                input_messages=[{"role": "user", "content": "hello"}],
                output_messages=[{"role": "assistant", "content": "hi"}],
            )
        )
        span.set_attribute(SpanAttributes.LLM_FINISH_REASON, "stop")
        span.set_attribute(SpanAttributes.SESSION_ID, "session-123")

    exported_span = in_memory_span_exporter.get_finished_spans()[0]
    attributes = dict(exported_span.attributes or {})

    assert attributes[SpanAttributes.LLM_MODEL_NAME] == "gpt-4o-mini"
    assert attributes[GenAIAttributes.GEN_AI_OPERATION_NAME] == GenAIOperationNameValues.CHAT.value
    assert attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "gpt-4o-mini"
    assert attributes[GenAIAttributes.GEN_AI_CONVERSATION_ID] == "session-123"
    assert _load_json_attribute(attributes, GenAIAttributes.GEN_AI_INPUT_MESSAGES) == [
        {
            "role": "user",
            "parts": [{"type": "text", "content": "hello"}],
        }
    ]


def test_oi_tracer_does_not_emit_genai_semconv_when_disabled(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    tracer = OITracer(
        tracer_provider.get_tracer(__name__),
        TraceConfig(enable_genai_semconv=False),
    )

    with tracer.start_as_current_span(
        "llm-span",
        openinference_span_kind="llm",
    ) as span:
        span.set_attributes(
            get_llm_attributes(
                model_name="gpt-4o-mini",
                input_messages=[{"role": "user", "content": "hello"}],
            )
        )

    exported_span = in_memory_span_exporter.get_finished_spans()[0]
    attributes = dict(exported_span.attributes or {})
    assert not any(key.startswith("gen_ai.") for key in attributes)


def test_oi_tracer_preserves_user_defined_genai_attributes(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    tracer = OITracer(
        tracer_provider.get_tracer(__name__),
        TraceConfig(enable_genai_semconv=True),
    )

    tracer.start_span(
        "llm-span",
        openinference_span_kind="llm",
        attributes={
            SpanAttributes.LLM_MODEL_NAME: "gpt-4o-mini",
            GenAIAttributes.GEN_AI_REQUEST_MODEL: "custom-model",
        },
    ).end()

    exported_span = in_memory_span_exporter.get_finished_spans()[0]
    attributes = dict(exported_span.attributes or {})
    assert attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "custom-model"


def test_oi_tracer_derives_genai_from_masked_attributes(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    tracer = OITracer(
        tracer_provider.get_tracer(__name__),
        TraceConfig(enable_genai_semconv=True, hide_input_text=True),
    )

    with tracer.start_as_current_span(
        "masked-llm-span",
        openinference_span_kind="llm",
    ) as span:
        span.set_attributes(
            get_llm_attributes(input_messages=[{"role": "user", "content": "secret"}])
        )

    exported_span = in_memory_span_exporter.get_finished_spans()[0]
    attributes = dict(exported_span.attributes or {})
    input_messages = _load_json_attribute(attributes, GenAIAttributes.GEN_AI_INPUT_MESSAGES)
    assert input_messages == [
        {
            "role": "user",
            "parts": [{"type": "text", "content": REDACTED_VALUE}],
        }
    ]


def test_oi_tracer_preserves_initial_genai_attributes_after_set_attribute(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    tracer = OITracer(
        tracer_provider.get_tracer(__name__),
        TraceConfig(enable_genai_semconv=True),
    )

    span = tracer.start_span(
        "llm-span",
        openinference_span_kind="llm",
        attributes={
            SpanAttributes.LLM_MODEL_NAME: "gpt-4o",
            GenAIAttributes.GEN_AI_REQUEST_MODEL: "custom-model",
        },
    )
    span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, 10)
    span.end()

    exported_span = in_memory_span_exporter.get_finished_spans()[0]
    attributes = dict(exported_span.attributes or {})
    assert attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "custom-model"
    assert attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 10


def test_oi_tracer_propagates_context_attributes_to_genai_semconv(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    tracer = OITracer(
        tracer_provider.get_tracer(__name__),
        TraceConfig(enable_genai_semconv=True),
    )

    with using_session("session-abc"):
        with tracer.start_as_current_span(
            "llm-span",
            openinference_span_kind="llm",
        ):
            pass

    exported_span = in_memory_span_exporter.get_finished_spans()[0]
    attributes = dict(exported_span.attributes or {})
    assert attributes[SpanAttributes.SESSION_ID] == "session-abc"
    assert attributes[GenAIAttributes.GEN_AI_CONVERSATION_ID] == "session-abc"

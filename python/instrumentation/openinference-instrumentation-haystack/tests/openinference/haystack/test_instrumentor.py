import json
from datetime import datetime
from typing import Any, Dict, Generator, Optional, Sequence, Union
from unittest.mock import MagicMock

import pytest
from haystack import Document
from haystack.components.agents import Agent
from haystack.components.builders.chat_prompt_builder import ChatPromptBuilder
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.embedders.openai_document_embedder import OpenAIDocumentEmbedder
from haystack.components.generators.chat.openai import OpenAIChatGenerator
from haystack.components.generators.openai import OpenAIGenerator
from haystack.components.retrievers.in_memory.bm25_retriever import InMemoryBM25Retriever
from haystack.components.websearch.serper_dev import SerperDevWebSearch
from haystack.core.errors import PipelineRuntimeError
from haystack.core.pipeline.async_pipeline import AsyncPipeline
from haystack.core.pipeline.pipeline import Pipeline
from haystack.dataclasses.chat_message import ChatMessage
from haystack.document_stores.in_memory.document_store import InMemoryDocumentStore
from haystack.tools import Tool
from haystack_integrations.components.rankers.cohere import (
    CohereRanker,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from opentelemetry.util._importlib_metadata import entry_points
from typing_extensions import TypeGuard

from openinference.instrumentation import OITracer, suppress_tracing, using_attributes
from openinference.instrumentation.haystack import HaystackInstrumentor
from openinference.instrumentation.haystack._wrappers import (
    infer_llm_provider_from_class_name,
    infer_llm_system_from_model,
)
from openinference.semconv.trace import (
    DocumentAttributes,
    EmbeddingAttributes,
    MessageAttributes,
    OpenInferenceLLMProviderValues,
    OpenInferenceLLMSystemValues,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    RerankerAttributes,
    SpanAttributes,
    ToolCallAttributes,
)


def remove_all_vcr_request_headers(request: Any) -> Any:
    """
    Removes all request headers.

    Example:
    ```
    @pytest.mark.vcr(
        before_record_response=remove_all_vcr_request_headers
    )
    def test_openai() -> None:
        # make request to OpenAI
    """
    request.headers.clear()
    return request


def remove_all_vcr_response_headers(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Removes all response headers.

    Example:
    ```
    @pytest.mark.vcr(
        before_record_response=remove_all_vcr_response_headers
    )
    def test_openai() -> None:
        # make request to OpenAI
    """
    response["headers"] = {}
    return response


@pytest.fixture()
def in_memory_span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture()
def tracer_provider(in_memory_span_exporter: InMemorySpanExporter) -> TracerProvider:
    resource = Resource(attributes={})
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))
    return tracer_provider


@pytest.fixture()
def setup_haystack_instrumentation(
    tracer_provider: TracerProvider,
) -> Generator[None, None, None]:
    HaystackInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    HaystackInstrumentor().uninstrument()


class TestInstrumentor:
    def test_entrypoint_for_opentelemetry_instrument(self) -> None:
        (instrumentor_entrypoint,) = entry_points(  # type: ignore[no-untyped-call]
            group="opentelemetry_instrumentor", name="haystack"
        )
        instrumentor = instrumentor_entrypoint.load()()
        assert isinstance(instrumentor, HaystackInstrumentor)

    # Ensure we're using the common OITracer from common openinference-instrumentation pkg
    def test_oitracer(self, setup_haystack_instrumentation: Any) -> None:
        assert isinstance(HaystackInstrumentor()._tracer, OITracer)


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
async def test_async_pipeline_with_chat_prompt_builder_and_chat_generator_produces_expected_spans(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_haystack_instrumentation: Any,
    openai_api_key: str,
) -> None:
    pipe = AsyncPipeline()
    prompt_builder = ChatPromptBuilder()
    llm = OpenAIChatGenerator(model="gpt-4o")
    pipe.add_component("prompt_builder", prompt_builder)
    pipe.add_component("llm", llm)
    pipe.connect("prompt_builder.prompt", "llm.messages")
    location = "Berlin"
    messages = [
        ChatMessage.from_system("Answer concisely in one sentence."),
        ChatMessage.from_user("What country is {{location}} in?"),
    ]
    await pipe.run_async(
        data={
            "prompt_builder": {
                "template_variables": {"location": location},
                "template": messages,
            }
        }
    )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 4
    span = spans[0]
    assert span.status.is_ok
    assert not span.events
    assert span.name == "ChatPromptBuilder.run"
    attributes = dict(span.attributes or {})
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == CHAIN
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    assert isinstance(attributes.pop(INPUT_VALUE), str)
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
    assert isinstance(attributes.pop(OUTPUT_VALUE), str)
    assert not attributes

    span = spans[1]
    assert span.status.is_ok
    assert not span.events
    assert span.name == "OpenAIChatGenerator.run_async"
    attributes = dict(span.attributes or {})
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == LLM
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    assert isinstance(attributes.pop(INPUT_VALUE), str)
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
    assert isinstance(attributes.pop(OUTPUT_VALUE), str)
    assert isinstance(llm_model_name := attributes.pop(LLM_MODEL_NAME), str)
    assert "gpt-4o" in llm_model_name
    assert isinstance(llm_provider := attributes.pop(LLM_PROVIDER), str)
    assert OpenInferenceLLMProviderValues.OPENAI.value in llm_provider
    assert isinstance(llm_system := attributes.pop(LLM_SYSTEM), str)
    assert OpenInferenceLLMSystemValues.OPENAI.value in llm_system
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "system"
    assert (
        attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}")
        == "Answer concisely in one sentence."
    )
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_ROLE}") == "user"
    assert (
        attributes.pop(f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_CONTENT}") == "What country is Berlin in?"
    )
    assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
    assert isinstance(
        output_content := attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}"), str
    )
    assert "germany" in output_content.lower()
    assert isinstance(prompt_tokens := attributes.pop(LLM_TOKEN_COUNT_PROMPT), int)
    assert isinstance(completion_tokens := attributes.pop(LLM_TOKEN_COUNT_COMPLETION), int)
    assert isinstance(total_tokens := attributes.pop(LLM_TOKEN_COUNT_TOTAL), int)
    assert prompt_tokens + completion_tokens == total_tokens
    assert not attributes

    span = spans[2]
    assert span.status.is_ok
    assert not span.events
    assert span.name == "AsyncPipeline.run_async_generator"
    attributes = dict(span.attributes or {})
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == CHAIN
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    assert isinstance(attributes.pop(INPUT_VALUE), str)
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
    assert isinstance(attributes.pop(OUTPUT_VALUE), str)
    assert not attributes
    span = spans[3]
    assert span.status.is_ok
    assert not span.events
    assert span.name == "AsyncPipeline.run_async"
    attributes = dict(span.attributes or {})
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == CHAIN
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    assert isinstance(attributes.pop(INPUT_VALUE), str)
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
    assert isinstance(attributes.pop(OUTPUT_VALUE), str)
    assert not attributes


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
def test_pipeline_with_chat_prompt_builder_and_chat_generator_produces_expected_spans(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_haystack_instrumentation: Any,
    openai_api_key: str,
) -> None:
    pipe = Pipeline()
    prompt_builder = ChatPromptBuilder()
    llm = OpenAIChatGenerator(model="gpt-4o")
    pipe.add_component("prompt_builder", prompt_builder)
    pipe.add_component("llm", llm)
    pipe.connect("prompt_builder.prompt", "llm.messages")
    location = "Berlin"
    messages = [
        ChatMessage.from_system("Answer concisely in one sentence."),
        ChatMessage.from_user("What country is {{location}} in?"),
    ]
    pipe.run(
        data={
            "prompt_builder": {
                "template_variables": {"location": location},
                "template": messages,
            }
        }
    )
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 3

    span = spans[0]
    assert span.status.is_ok
    assert not span.events
    assert span.name == "ChatPromptBuilder.run"
    attributes = dict(span.attributes or {})
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == CHAIN
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    assert isinstance(attributes.pop(INPUT_VALUE), str)
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
    assert isinstance(attributes.pop(OUTPUT_VALUE), str)
    assert not attributes

    span = spans[1]
    assert span.status.is_ok
    assert not span.events
    assert span.name == "OpenAIChatGenerator.run"
    attributes = dict(span.attributes or {})
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == LLM
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    assert isinstance(attributes.pop(INPUT_VALUE), str)
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
    assert isinstance(attributes.pop(OUTPUT_VALUE), str)
    assert isinstance(llm_model_name := attributes.pop(LLM_MODEL_NAME), str)
    assert "gpt-4o" in llm_model_name
    assert isinstance(llm_provider := attributes.pop(LLM_PROVIDER), str)
    assert OpenInferenceLLMProviderValues.OPENAI.value in llm_provider
    assert isinstance(llm_system := attributes.pop(LLM_SYSTEM), str)
    assert OpenInferenceLLMSystemValues.OPENAI.value in llm_system
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "system"
    assert (
        attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}")
        == "Answer concisely in one sentence."
    )
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_ROLE}") == "user"
    assert (
        attributes.pop(f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_CONTENT}") == "What country is Berlin in?"
    )
    assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
    assert isinstance(
        output_content := attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}"), str
    )
    assert "germany" in output_content.lower()
    assert isinstance(prompt_tokens := attributes.pop(LLM_TOKEN_COUNT_PROMPT), int)
    assert isinstance(completion_tokens := attributes.pop(LLM_TOKEN_COUNT_COMPLETION), int)
    assert isinstance(total_tokens := attributes.pop(LLM_TOKEN_COUNT_TOTAL), int)
    assert prompt_tokens + completion_tokens == total_tokens
    assert not attributes

    span = spans[2]
    assert span.status.is_ok
    assert not span.events
    assert span.name == "Pipeline.run"
    attributes = dict(span.attributes or {})
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == CHAIN
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    assert isinstance(attributes.pop(INPUT_VALUE), str)
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
    assert isinstance(attributes.pop(OUTPUT_VALUE), str)
    assert not attributes


def test_haystack_instrumentation_filtering(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_haystack_instrumentation: Any,
) -> None:
    documents = [
        Document(
            content="Use pip to install a basic version of Haystack's latest release",
            meta={"version": 1.15, "date": datetime(2023, 3, 30)},
        ),
        Document(
            content="Use pip to install a basic version of Haystack's latest release: pip install",
            meta={"version": 1.22, "date": datetime(2023, 11, 7)},
        ),
        Document(
            content="Use pip to install only the Haystack 2.0 code: pip install haystack-ai",
            meta={"version": 2.0, "date": datetime(2023, 12, 4)},
        ),
    ]
    document_store = InMemoryDocumentStore(bm25_algorithm="BM25Plus")
    document_store.write_documents(documents=documents)

    pipeline = Pipeline()
    pipeline.add_component(
        instance=InMemoryBM25Retriever(document_store=document_store), name="retriever"
    )

    query = "Haystack installation"

    pipeline.run(
        data={
            "retriever": {
                "query": query,
                "filters": {"field": "meta.version", "operator": ">", "value": 1.21},
            }
        }
    )

    spans = in_memory_span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "InMemoryBM25Retriever.run",
        "Pipeline.run",
    ]

    assert [
        span.attributes.get("openinference.span.kind") for span in spans if span and span.attributes
    ] == [
        RETRIEVER,
        CHAIN,
    ]


async def test_haystack_instrumentation_async_pipeline_filtering(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_haystack_instrumentation: Any,
) -> None:
    documents = [
        Document(
            content="Use pip to install a basic version of Haystack's latest release",
            meta={"version": 1.15, "date": datetime(2023, 3, 30)},
        ),
        Document(
            content="Use pip to install a basic version of Haystack's latest release: pip install",
            meta={"version": 1.22, "date": datetime(2023, 11, 7)},
        ),
        Document(
            content="Use pip to install only the Haystack 2.0 code: pip install haystack-ai",
            meta={"version": 2.0, "date": datetime(2023, 12, 4)},
        ),
    ]
    document_store = InMemoryDocumentStore(bm25_algorithm="BM25Plus")
    document_store.write_documents(documents=documents)

    pipeline = AsyncPipeline()
    pipeline.add_component(
        instance=InMemoryBM25Retriever(document_store=document_store), name="retriever"
    )

    query = "Haystack installation"

    await pipeline.run_async(
        data={
            "retriever": {
                "query": query,
                "filters": {"field": "meta.version", "operator": ">", "value": 1.21},
            }
        }
    )

    spans = in_memory_span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "InMemoryBM25Retriever.run_async",
        "AsyncPipeline.run_async_generator",
        "AsyncPipeline.run_async",
    ]

    assert [
        span.attributes.get("openinference.span.kind") for span in spans if span and span.attributes
    ] == [RETRIEVER, CHAIN, CHAIN]


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
def test_tool_calling_llm_span_has_expected_attributes(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_haystack_instrumentation: Any,
    openai_api_key: str,
) -> None:
    chat_generator = OpenAIChatGenerator(model="gpt-4o")
    pipe = Pipeline()
    pipe.add_component("llm", chat_generator)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        }
                    },
                    "required": ["location"],
                },
            },
        },
    ]
    response = pipe.run(
        {
            "llm": {
                "messages": [
                    ChatMessage.from_user("What is the weather in Berlin"),
                ],
                "generation_kwargs": {"tools": tools},
            }
        }
    )
    chat_message = response["llm"]["replies"][0]
    tool_call = chat_message.tool_calls[0]
    assert tool_call.tool_name == "get_current_weather"

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 2
    span = spans[0]
    assert span.name == "OpenAIChatGenerator.run"
    assert span.status.is_ok
    assert not span.events
    attributes = dict(span.attributes or {})
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == "LLM"
    assert isinstance(llm_model_name := attributes.pop(LLM_MODEL_NAME), str)
    assert "gpt-4o" in llm_model_name
    assert isinstance(llm_provider := attributes.pop(LLM_PROVIDER), str)
    assert OpenInferenceLLMProviderValues.OPENAI.value in llm_provider
    assert isinstance(llm_system := attributes.pop(LLM_SYSTEM), str)
    assert OpenInferenceLLMSystemValues.OPENAI.value in llm_system
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    assert isinstance(input_value := attributes.pop(INPUT_VALUE), str)
    assert "What is the weather in Berlin" in input_value
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
    assert isinstance(attributes.pop(OUTPUT_VALUE), str)
    assert (
        attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}")
        == "What is the weather in Berlin"
    )
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "user"
    assert (
        attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_TOOL_CALLS}.0.{TOOL_CALL_FUNCTION_NAME}")
        == "get_current_weather"
    )
    assert isinstance(
        tool_call_arguments := attributes.pop(
            f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_TOOL_CALLS}.0.{TOOL_CALL_FUNCTION_ARGUMENTS_JSON}"
        ),
        str,
    )
    assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
    assert json.loads(tool_call_arguments) == {"location": "Berlin"}
    assert isinstance(prompt_tokens := attributes.pop(LLM_TOKEN_COUNT_PROMPT), int)
    assert isinstance(completion_tokens := attributes.pop(LLM_TOKEN_COUNT_COMPLETION), int)
    assert isinstance(total_tokens := attributes.pop(LLM_TOKEN_COUNT_TOTAL), int)
    assert prompt_tokens + completion_tokens == total_tokens
    assert not attributes


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
def test_async_pipeline_tool_calling_llm_span_has_expected_attributes(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_haystack_instrumentation: Any,
    openai_api_key: str,
) -> None:
    chat_generator = OpenAIChatGenerator(model="gpt-4o")
    pipe = AsyncPipeline()
    pipe.add_component("llm", chat_generator)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        }
                    },
                    "required": ["location"],
                },
            },
        },
    ]
    response = pipe.run(
        {
            "llm": {
                "messages": [
                    ChatMessage.from_user("What is the weather in Berlin"),
                ],
                "generation_kwargs": {"tools": tools},
            }
        }
    )
    chat_message = response["llm"]["replies"][0]
    tool_call = chat_message.tool_calls[0]
    assert tool_call.tool_name == "get_current_weather"

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 4
    assert [span.name for span in spans] == [
        "OpenAIChatGenerator.run_async",
        "AsyncPipeline.run_async_generator",
        "AsyncPipeline.run_async",
        "AsyncPipeline.run",
    ]
    span = spans[0]
    assert span.name == "OpenAIChatGenerator.run_async"
    assert span.status.is_ok
    assert not span.events
    attributes = dict(span.attributes or {})
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == "LLM"
    assert isinstance(llm_model_name := attributes.pop(LLM_MODEL_NAME), str)
    assert "gpt-4o" in llm_model_name
    assert isinstance(llm_provider := attributes.pop(LLM_PROVIDER), str)
    assert OpenInferenceLLMProviderValues.OPENAI.value in llm_provider
    assert isinstance(llm_system := attributes.pop(LLM_SYSTEM), str)
    assert OpenInferenceLLMSystemValues.OPENAI.value in llm_system
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    assert isinstance(input_value := attributes.pop(INPUT_VALUE), str)
    assert "What is the weather in Berlin" in input_value
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
    assert isinstance(attributes.pop(OUTPUT_VALUE), str)
    assert (
        attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}")
        == "What is the weather in Berlin"
    )
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "user"
    assert (
        attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_TOOL_CALLS}.0.{TOOL_CALL_FUNCTION_NAME}")
        == "get_current_weather"
    )
    assert isinstance(
        tool_call_arguments := attributes.pop(
            f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_TOOL_CALLS}.0.{TOOL_CALL_FUNCTION_ARGUMENTS_JSON}"
        ),
        str,
    )
    assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
    assert json.loads(tool_call_arguments) == {"location": "Berlin, Germany"}
    assert isinstance(prompt_tokens := attributes.pop(LLM_TOKEN_COUNT_PROMPT), int)
    assert isinstance(completion_tokens := attributes.pop(LLM_TOKEN_COUNT_COMPLETION), int)
    assert isinstance(total_tokens := attributes.pop(LLM_TOKEN_COUNT_TOTAL), int)
    assert prompt_tokens + completion_tokens == total_tokens
    assert not attributes


def test_instrument_and_uninstrument_methods_wrap_and_unwrap_expected_methods(
    tracer_provider: TracerProvider,
) -> None:
    HaystackInstrumentor().instrument(tracer_provider=tracer_provider)

    assert hasattr(Pipeline.run, "__wrapped__")
    assert hasattr(Pipeline._run_component, "__wrapped__")
    assert hasattr(AsyncPipeline.run, "__wrapped__")
    assert hasattr(AsyncPipeline.run_async, "__wrapped__")
    assert hasattr(AsyncPipeline.run_async_generator, "__wrapped__")
    assert hasattr(AsyncPipeline._run_component_async, "__wrapped__")

    HaystackInstrumentor().uninstrument()

    assert not hasattr(Pipeline.run, "__wrapped__")
    assert not hasattr(Pipeline._run_component, "__wrapped__")
    assert not hasattr(AsyncPipeline.run, "__wrapped__")
    assert not hasattr(AsyncPipeline.run_async, "__wrapped__")
    assert not hasattr(AsyncPipeline.run_async_generator, "__wrapped__")
    assert not hasattr(AsyncPipeline._run_component_async, "__wrapped__")


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
def test_openai_chat_generator_llm_span_has_expected_attributes(
    openai_api_key: str,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_haystack_instrumentation: Any,
) -> None:
    pipe = Pipeline()
    llm = OpenAIChatGenerator(model="gpt-4o")
    pipe.add_component("llm", llm)
    response = pipe.run(
        {
            "llm": {
                "messages": [
                    ChatMessage.from_system("Answer user questions succinctly"),
                    ChatMessage.from_assistant("What can I help you with?"),
                    ChatMessage.from_user("Who won the World Cup in 2022? Answer in one word."),
                ]
            }
        }
    )
    assert "argentina" in response["llm"]["replies"][0].text.lower()
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 2
    span = spans[0]
    assert span.status.is_ok
    assert not span.events
    assert span.name == "OpenAIChatGenerator.run"
    attributes = dict(span.attributes or {})
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == "LLM"
    assert (
        attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}")
        == "Answer user questions succinctly"
    )
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "system"
    assert (
        attributes.pop(f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_CONTENT}") == "What can I help you with?"
    )
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_ROLE}") == "assistant"

    assert (
        attributes.pop(f"{LLM_INPUT_MESSAGES}.2.{MESSAGE_CONTENT}")
        == "Who won the World Cup in 2022? Answer in one word."
    )
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.2.{MESSAGE_ROLE}") == "user"
    assert isinstance(
        (output_message_content := attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}")),
        str,
    )
    assert "argentina" in output_message_content.lower()
    assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
    assert isinstance(prompt_tokens := attributes.pop(LLM_TOKEN_COUNT_PROMPT), int)
    assert isinstance(completion_tokens := attributes.pop(LLM_TOKEN_COUNT_COMPLETION), int)
    assert isinstance(total_tokens := attributes.pop(LLM_TOKEN_COUNT_TOTAL), int)
    assert prompt_tokens + completion_tokens == total_tokens
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    assert isinstance(attributes.pop(INPUT_VALUE), str)
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
    assert isinstance(attributes.pop(OUTPUT_VALUE), str)
    assert isinstance(llm_model_name := attributes.pop(LLM_MODEL_NAME), str)
    assert "gpt-4o" in llm_model_name
    assert isinstance(llm_provider := attributes.pop(LLM_PROVIDER), str)
    assert OpenInferenceLLMProviderValues.OPENAI.value in llm_provider
    assert isinstance(llm_system := attributes.pop(LLM_SYSTEM), str)
    assert OpenInferenceLLMSystemValues.OPENAI.value in llm_system
    assert not attributes


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
async def test_async_pipeline_openai_chat_generator_llm_span_has_expected_attributes(
    openai_api_key: str,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_haystack_instrumentation: Any,
) -> None:
    async_pipe = AsyncPipeline()
    llm = OpenAIChatGenerator(model="gpt-4o")
    async_pipe.add_component("llm", llm)
    responses = [
        item
        async for item in async_pipe.run_async_generator(
            {
                "llm": {
                    "messages": [
                        ChatMessage.from_system("Answer user questions succinctly"),
                        ChatMessage.from_assistant("What can I help you with?"),
                        ChatMessage.from_user("Who won the World Cup in 2022? Answer in one word."),
                    ]
                }
            }
        )
    ]
    assert "argentina" in responses[0]["llm"]["replies"][0].text.lower()
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 2
    assert [span.name for span in spans] == [
        "OpenAIChatGenerator.run_async",
        "AsyncPipeline.run_async_generator",
    ]
    span = spans[0]
    assert span.status.is_ok
    assert not span.events
    assert span.name == "OpenAIChatGenerator.run_async"
    attributes = dict(span.attributes or {})
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == "LLM"
    assert (
        attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}")
        == "Answer user questions succinctly"
    )
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "system"
    assert (
        attributes.pop(f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_CONTENT}") == "What can I help you with?"
    )
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_ROLE}") == "assistant"

    assert (
        attributes.pop(f"{LLM_INPUT_MESSAGES}.2.{MESSAGE_CONTENT}")
        == "Who won the World Cup in 2022? Answer in one word."
    )
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.2.{MESSAGE_ROLE}") == "user"
    assert isinstance(
        (output_message_content := attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}")),
        str,
    )
    assert "argentina" in output_message_content.lower()
    assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
    assert isinstance(prompt_tokens := attributes.pop(LLM_TOKEN_COUNT_PROMPT), int)
    assert isinstance(completion_tokens := attributes.pop(LLM_TOKEN_COUNT_COMPLETION), int)
    assert isinstance(total_tokens := attributes.pop(LLM_TOKEN_COUNT_TOTAL), int)
    assert prompt_tokens + completion_tokens == total_tokens
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    assert isinstance(attributes.pop(INPUT_VALUE), str)
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
    assert isinstance(attributes.pop(OUTPUT_VALUE), str)
    assert isinstance(llm_model_name := attributes.pop(LLM_MODEL_NAME), str)
    assert "gpt-4o" in llm_model_name
    assert isinstance(llm_provider := attributes.pop(LLM_PROVIDER), str)
    assert OpenInferenceLLMProviderValues.OPENAI.value in llm_provider
    assert isinstance(llm_system := attributes.pop(LLM_SYSTEM), str)
    assert OpenInferenceLLMSystemValues.OPENAI.value in llm_system
    assert not attributes


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
def test_openai_generator_llm_span_has_expected_attributes(
    openai_api_key: str,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_haystack_instrumentation: Any,
) -> None:
    pipe = Pipeline()
    llm = OpenAIGenerator(model="gpt-4o")
    pipe.add_component("llm", llm)
    response = pipe.run(
        {
            "llm": {
                "prompt": "Who won the World Cup in 2022? Answer in one word.",
            }
        }
    )
    assert "argentina" in response["llm"]["replies"][0].lower()
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 2
    span = spans[0]
    assert span.name == "OpenAIGenerator.run"
    assert span.status.is_ok
    assert not span.events
    attributes = dict(span.attributes or {})
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == "LLM"
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    assert isinstance(input_value := attributes.pop(INPUT_VALUE), str)
    input_value_data = json.loads(input_value)
    assert input_value_data.get("prompt") == "Who won the World Cup in 2022? Answer in one word."
    assert isinstance(llm_model_name := attributes.pop(LLM_MODEL_NAME), str)
    assert "gpt-4o" in llm_model_name
    assert isinstance(llm_provider := attributes.pop(LLM_PROVIDER), str)
    assert OpenInferenceLLMProviderValues.OPENAI.value in llm_provider
    assert isinstance(llm_system := attributes.pop(LLM_SYSTEM), str)
    assert OpenInferenceLLMSystemValues.OPENAI.value in llm_system
    assert (
        attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}")
        == "Who won the World Cup in 2022? Answer in one word."
    )
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "user"
    assert isinstance(
        (output_message_content := attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}")),
        str,
    )
    assert "argentina" in output_message_content.lower()
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
    assert isinstance(output_value := attributes.pop(OUTPUT_VALUE), str)
    output_value_data = json.loads(output_value)
    assert len(replies := output_value_data["replies"]) == 1
    assert "argentina" in replies[0].lower()
    assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
    assert isinstance(prompt_tokens := attributes.pop(LLM_TOKEN_COUNT_PROMPT), int)
    assert isinstance(completion_tokens := attributes.pop(LLM_TOKEN_COUNT_COMPLETION), int)
    assert isinstance(total_tokens := attributes.pop(LLM_TOKEN_COUNT_TOTAL), int)
    assert prompt_tokens + completion_tokens == total_tokens
    assert not attributes


@pytest.mark.parametrize(
    "default_template, prompt_builder_inputs",
    [
        pytest.param(
            "Where is {{ city }}?",
            {"template_variables": {"city": "Munich"}},
            id="default-template",
        ),
        pytest.param(
            "What is the weather in {{ city }}?",
            {
                "template": "Where is {{ city }}?",  # overrides default template
                "template_variables": {"city": "Munich"},
            },
            id="input-template-overrides-default-template",
        ),
        pytest.param(
            "Where is {{ city }}?",
            {
                "city": "Munich",
            },
            id="input-kwarg-recorded-as-template-variable",
        ),
        pytest.param(
            "Where is {{ city }}?",
            {
                "template_variables": {"city": "Munich"},  # overrides kwarg
                "city": "Berlin",
            },
            id="input-template-variables-overrides-input-kwarg",
        ),
    ],
)
def test_prompt_builder_llm_span_has_expected_attributes(
    default_template: Optional[str],
    prompt_builder_inputs: Dict[str, Any],
    in_memory_span_exporter: InMemorySpanExporter,
    setup_haystack_instrumentation: Any,
) -> None:
    prompt_builder = PromptBuilder(template=default_template or "")
    pipe = Pipeline()
    pipe.add_component("prompt_builder", prompt_builder)
    output = pipe.run({"prompt_builder": prompt_builder_inputs})
    assert output == {"prompt_builder": {"prompt": "Where is Munich?"}}
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 2
    span = spans[0]
    assert span.name == "PromptBuilder.run"
    assert span.status.is_ok
    assert not span.events
    attributes = dict(span.attributes or {})
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == "LLM"
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    assert isinstance(attributes.pop(INPUT_VALUE), str)
    assert attributes.pop(LLM_PROMPT_TEMPLATE) == "Where is {{ city }}?"
    assert isinstance(
        prompt_template_variables_json := attributes.pop(LLM_PROMPT_TEMPLATE_VARIABLES), str
    )
    assert json.loads(prompt_template_variables_json) == {"city": "Munich"}
    assert attributes.pop(OUTPUT_MIME_TYPE) == TEXT
    assert attributes.pop(OUTPUT_VALUE) == "Where is Munich?"
    assert not attributes


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
def test_cohere_reranker_span_has_expected_attributes(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_haystack_instrumentation: Any,
    cohere_api_key: str,
) -> None:
    ranker = CohereRanker()
    pipe = Pipeline()
    pipe.add_component("ranker", ranker)
    response = pipe.run(
        {
            "ranker": {
                "query": "Who won the World Cup in 2022?",
                "documents": [
                    Document(
                        content="Paul Graham is the founder of Y Combinator.",
                    ),
                    Document(
                        content=(
                            "Lionel Messi, captain of the Argentinian national team, "
                            " won his first World Cup in 2022."
                        ),
                    ),
                    Document(
                        content="France lost the 2022 World Cup.",
                    ),  # Cohere consistently ranks this document last
                ],
                "top_k": 2,
            }
        }
    )
    ranker_response = response["ranker"]
    assert len(response_documents := ranker_response["documents"]) == 2
    assert "Lionel Messi" in response_documents[0].content
    assert "Paul Graham" in response_documents[1].content

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 2
    span = spans[0]
    assert span.name == "CohereRanker.run"
    assert span.status.is_ok
    assert not span.events
    attributes = dict(span.attributes or {})
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == RERANKER
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    assert isinstance(attributes.pop(INPUT_VALUE), str)
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
    assert isinstance(attributes.pop(OUTPUT_VALUE), str)
    assert attributes.pop(RERANKER_QUERY) == "Who won the World Cup in 2022?"
    assert attributes.pop(RERANKER_TOP_K) == 2
    assert isinstance(attributes.pop(RERANKER_MODEL_NAME), str)
    assert isinstance(
        in_doc0 := attributes.pop(f"{RERANKER_INPUT_DOCUMENTS}.0.{DOCUMENT_CONTENT}"), str
    )
    assert "Paul Graham" in in_doc0
    assert isinstance(
        in_doc1 := attributes.pop(f"{RERANKER_INPUT_DOCUMENTS}.1.{DOCUMENT_CONTENT}"), str
    )
    assert "Lionel Messi" in in_doc1
    assert isinstance(
        in_doc2 := attributes.pop(f"{RERANKER_INPUT_DOCUMENTS}.2.{DOCUMENT_CONTENT}"), str
    )
    assert "France" in in_doc2
    assert isinstance(attributes.pop(f"{RERANKER_INPUT_DOCUMENTS}.0.{DOCUMENT_ID}"), str)
    assert isinstance(attributes.pop(f"{RERANKER_INPUT_DOCUMENTS}.1.{DOCUMENT_ID}"), str)
    assert isinstance(attributes.pop(f"{RERANKER_INPUT_DOCUMENTS}.2.{DOCUMENT_ID}"), str)
    assert isinstance(
        out_doc0 := attributes.pop(f"{RERANKER_OUTPUT_DOCUMENTS}.0.{DOCUMENT_CONTENT}"), str
    )
    assert "Lionel Messi" in out_doc0
    assert isinstance(
        out_doc1 := attributes.pop(f"{RERANKER_OUTPUT_DOCUMENTS}.1.{DOCUMENT_CONTENT}"), str
    )
    assert "Paul Graham" in out_doc1
    assert isinstance(attributes.pop(f"{RERANKER_OUTPUT_DOCUMENTS}.0.{DOCUMENT_ID}"), str)
    assert isinstance(attributes.pop(f"{RERANKER_OUTPUT_DOCUMENTS}.1.{DOCUMENT_ID}"), str)
    assert isinstance(attributes.pop(f"{RERANKER_OUTPUT_DOCUMENTS}.0.{DOCUMENT_SCORE}"), float)
    assert isinstance(attributes.pop(f"{RERANKER_OUTPUT_DOCUMENTS}.1.{DOCUMENT_SCORE}"), float)
    assert not attributes


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
def test_serperdev_websearch_retriever_span_has_expected_attributes(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_haystack_instrumentation: Any,
    serperdev_api_key: str,
) -> None:
    # To run this test without `vcrpy`, create an account and an API key at
    # https://serper.dev/.
    k = 2
    web_search = SerperDevWebSearch(top_k=k)
    pipe = Pipeline()
    pipe.add_component("websearch", web_search)
    output = pipe.run({"websearch": {"query": "Who won the World Cup in 2022?"}})
    assert "websearch" in output
    assert len(output["websearch"]) == k
    assert (documents := output["websearch"].get("documents")) is not None
    assert len(documents) == k
    assert (links := output["websearch"].get("links")) is not None
    assert len(links) == k

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == k
    span = spans[0]
    assert span.name == "SerperDevWebSearch.run"
    assert span.status.is_ok
    assert not span.events
    attributes = dict(span.attributes or {})
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == RETRIEVER
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    assert isinstance(input_value := attributes.pop(INPUT_VALUE), str)
    assert json.loads(input_value) == {"query": "Who won the World Cup in 2022?"}
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
    assert isinstance(output_value := attributes.pop(OUTPUT_VALUE), str)
    output_value_data = json.loads(output_value)
    assert len(output_value_data.get("documents")) == k
    assert output_value_data.get("links") == output["websearch"]["links"]
    for document_index in range(k):
        output_document = output["websearch"]["documents"][document_index]
        assert (
            attributes.pop(f"{RETRIEVAL_DOCUMENTS}.{document_index}.{DOCUMENT_CONTENT}")
            == output_document.content
        )
        assert (
            attributes.pop(f"{RETRIEVAL_DOCUMENTS}.{document_index}.{DOCUMENT_ID}")
            == output_document.id
        )
        assert isinstance(
            document_metadata := attributes.pop(
                f"{RETRIEVAL_DOCUMENTS}.{document_index}.{DOCUMENT_METADATA}"
            ),
            str,
        )
        assert json.loads(document_metadata) == output_document.meta
    assert not attributes


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
def test_openai_document_embedder_embedding_span_has_expected_attributes(
    openai_api_key: str,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_haystack_instrumentation: Any,
) -> None:
    pipe = Pipeline()
    embedder = OpenAIDocumentEmbedder(model="text-embedding-3-small")
    pipe.add_component("embedder", embedder)
    response = pipe.run(
        {
            "embedder": {
                "documents": [
                    Document(content="Argentina won the World Cup in 2022."),
                    Document(content="France won the World Cup in 2018."),
                ]
            }
        }
    )
    assert (response_documents := response["embedder"].get("documents")) is not None
    assert len(response_documents) == 2
    assert "Argentina won the World Cup in 2022." == response_documents[0].content
    assert response_documents[0].embedding is not None
    assert "France won the World Cup in 2018." == response_documents[1].content
    assert response_documents[1].embedding is not None

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 2
    span = spans[0]
    assert span.name == "CreateEmbeddings"
    assert span.status.is_ok
    assert not span.events
    attributes = dict(span.attributes or {})
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == "EMBEDDING"
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    assert isinstance(input_value := attributes.pop(INPUT_VALUE), str)
    input_value_data = json.loads(input_value)
    assert len(input_value_data) == 1
    assert (input_documents := input_value_data.get("documents")) is not None
    assert len(input_documents) == 2
    assert "Argentina won the World Cup in 2022." in input_documents[0]
    assert "France won the World Cup in 2018." in input_documents[1]
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
    assert isinstance(output_value := attributes.pop(OUTPUT_VALUE), str)
    output_value_data = json.loads(output_value)
    assert len(output_value_data) == 2
    assert (output_documents := output_value_data.get("documents")) is not None
    assert len(output_documents) == 2
    assert "Argentina won the World Cup in 2022." in output_documents[0]
    assert "France won the World Cup in 2018." in output_documents[1]
    assert attributes.pop(EMBEDDING_MODEL_NAME) == "text-embedding-3-small"
    assert (
        attributes.pop(f"{EMBEDDING_EMBEDDINGS}.0.{EMBEDDING_TEXT}")
        == "Argentina won the World Cup in 2022."
    )
    assert _is_vector(attributes.pop(f"{EMBEDDING_EMBEDDINGS}.0.{EMBEDDING_VECTOR}"))
    assert (
        attributes.pop(f"{EMBEDDING_EMBEDDINGS}.1.{EMBEDDING_TEXT}")
        == "France won the World Cup in 2018."
    )
    assert _is_vector(attributes.pop(f"{EMBEDDING_EMBEDDINGS}.1.{EMBEDDING_VECTOR}"))
    invocation_params_raw = attributes.pop(EMBEDDING_INVOCATION_PARAMETERS, None)
    if invocation_params_raw is not None and isinstance(invocation_params_raw, str):
        invocation_params = json.loads(invocation_params_raw)
        assert isinstance(invocation_params, dict)
    assert not attributes


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
def test_pipelines_and_components_produce_no_tracing_with_suppress_tracing(
    openai_api_key: str,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_haystack_instrumentation: Any,
) -> None:
    pipe = Pipeline()
    llm = OpenAIGenerator(model="gpt-4o")
    pipe.add_component("llm", llm)
    with suppress_tracing():
        response = pipe.run(
            {
                "llm": {
                    "prompt": "Who won the World Cup in 2022? Answer in one word.",
                }
            }
        )
    assert "argentina" in response["llm"]["replies"][0].lower()
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 0


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
def test_error_status_code_and_exception_events_with_invalid_api_key(
    openai_api_key: str,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_haystack_instrumentation: Any,
) -> None:
    pipe = Pipeline()
    llm = OpenAIGenerator(model="gpt-4o")
    pipe.add_component("llm", llm)
    with pytest.raises(PipelineRuntimeError):
        pipe.run(
            {
                "llm": {
                    "prompt": "Who won the World Cup in 2022? Answer in one word.",
                }
            }
        )
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 2
    for span in spans:
        assert span.status.status_code is StatusCode.ERROR
        assert len(span.events) == 1
        event = span.events[0]
        assert event.name == "exception"
        event_attributes = dict(event.attributes or {})
        assert isinstance(exception_message := event_attributes["exception.message"], str)
        assert "401" in exception_message
        assert "api key" in exception_message.lower()


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
def test_pipeline_and_component_spans_contain_context_attributes(
    openai_api_key: str,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_haystack_instrumentation: Any,
) -> None:
    pipe = Pipeline()
    llm = OpenAIGenerator(model="gpt-4o")
    pipe.add_component("llm", llm)
    with using_attributes(
        session_id="session-id",
        user_id="user-id",
        metadata={"metadata-key": "metadata-value"},
        tags=["tag"],
        prompt_template="template with {var_name}",
        prompt_template_version="prompt-template-version",
        prompt_template_variables={"var_name": "var-value"},
    ):
        response = pipe.run(
            {
                "llm": {
                    "prompt": "Who won the World Cup in 2022? Answer in one word.",
                }
            }
        )
    assert "argentina" in response["llm"]["replies"][0].lower()
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 2
    for span in spans:
        attributes = dict(span.attributes or {})
        assert attributes.get(SESSION_ID, "session-id")
        assert attributes.get(USER_ID, "user-id")
        assert attributes.get(METADATA, '{"metadata-key": "metadata-value"}')
        assert attributes.get(TAG_TAGS, ["tag"])
        assert attributes.get(LLM_PROMPT_TEMPLATE, "tempate with {var_name}")
        assert attributes.get(LLM_PROMPT_TEMPLATE_VERSION, "prompt-template-version")
        assert attributes.get(LLM_PROMPT_TEMPLATE_VARIABLES, '{"var_name": "var-value"}')


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
    record_mode="once",
)
async def test_agent_run_component_spans(
    openai_api_key: str,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_haystack_instrumentation: Any,
    use_async: bool,
) -> None:
    run_method = "run_async" if use_async else "run"

    def search_documents(query: str, user_context: str) -> Dict[str, Any]:
        """Search documents using query and user context."""
        return {"results": [f"Found results for '{query}' (user: {user_context})"]}

    search_tool = Tool(
        name="search",
        description="Search documents",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}, "user_context": {"type": "string"}},
            "required": ["query"],
        },
        function=search_documents,
        inputs_from_state={"user_name": "user_context"},
    )
    agent = Agent(
        chat_generator=OpenAIChatGenerator(),
        tools=[search_tool],
        state_schema={"user_name": {"type": str}, "search_results": {"type": list}},
    )
    if use_async:
        result = await agent.run_async(
            messages=[ChatMessage.from_user("Search for Python tutorials")], user_name="Alice"
        )
    else:
        result = agent.run(
            messages=[ChatMessage.from_user("Search for Python tutorials")], user_name="Alice"
        )
    last_message = result["last_message"]
    assert last_message.role.name == "ASSISTANT"
    assert last_message.text == (
        'I found some results for "Python tutorials." Would you like'
        " more specific information about them?"
    )
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 4
    openai_span = spans[0]
    assert openai_span.name == f"OpenAIChatGenerator.{run_method}"
    assert openai_span.status.is_ok
    attributes = dict(openai_span.attributes or {})
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == "LLM"
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    assert isinstance(attributes.pop(INPUT_VALUE), str)
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
    assert isinstance(attributes.pop(OUTPUT_VALUE), str)
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "user"
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}") == (
        "Search for Python tutorials"
    )
    assert isinstance(llm_model_name := attributes.pop(LLM_MODEL_NAME), str)
    assert "gpt-4o" in llm_model_name
    assert isinstance(llm_provider := attributes.pop(LLM_PROVIDER), str)
    assert OpenInferenceLLMProviderValues.OPENAI.value in llm_provider
    assert isinstance(llm_system := attributes.pop(LLM_SYSTEM), str)
    assert OpenInferenceLLMSystemValues.OPENAI.value in llm_system
    tool_prefix = f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_TOOL_CALLS}.0"
    assert attributes.pop(f"{tool_prefix}.{TOOL_CALL_FUNCTION_NAME}") == "search"
    assert isinstance(
        tool_call_arguments := attributes.pop(f"{tool_prefix}.{TOOL_CALL_FUNCTION_ARGUMENTS_JSON}"),
        str,
    )
    assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
    assert json.loads(tool_call_arguments) == {"query": "Python tutorials"}
    assert isinstance(prompt_tokens := attributes.pop(LLM_TOKEN_COUNT_PROMPT), int)
    assert isinstance(completion_tokens := attributes.pop(LLM_TOKEN_COUNT_COMPLETION), int)
    assert isinstance(total_tokens := attributes.pop(LLM_TOKEN_COUNT_TOTAL), int)
    assert prompt_tokens + completion_tokens == total_tokens
    assert not attributes
    tool_invoker_span = spans[1]
    assert tool_invoker_span.name == f"ToolInvoker.{run_method}"
    assert tool_invoker_span.status.is_ok
    attributes = dict(tool_invoker_span.attributes or {})
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == "CHAIN"
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    assert isinstance(attributes.pop(INPUT_VALUE), str)
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
    assert isinstance(attributes.pop(OUTPUT_VALUE), str)
    assert not attributes
    openai_span = spans[2]
    assert openai_span.name == f"OpenAIChatGenerator.{run_method}"
    assert openai_span.status.is_ok
    attributes = dict(openai_span.attributes or {})
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == "LLM"
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    assert isinstance(attributes.pop(INPUT_VALUE), str)
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
    assert isinstance(attributes.pop(OUTPUT_VALUE), str)
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "user"
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_ROLE}") == "assistant"
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.2.{MESSAGE_ROLE}") == "tool"
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}") == (
        "Search for Python tutorials"
    )
    assert isinstance(llm_model_name := attributes.pop(LLM_MODEL_NAME), str)
    assert "gpt-4o" in llm_model_name
    assert isinstance(llm_provider := attributes.pop(LLM_PROVIDER), str)
    assert OpenInferenceLLMProviderValues.OPENAI.value in llm_provider
    assert isinstance(llm_system := attributes.pop(LLM_SYSTEM), str)
    assert OpenInferenceLLMSystemValues.OPENAI.value in llm_system
    assert isinstance(attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}"), str)
    assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
    assert isinstance(prompt_tokens := attributes.pop(LLM_TOKEN_COUNT_PROMPT), int)
    assert isinstance(completion_tokens := attributes.pop(LLM_TOKEN_COUNT_COMPLETION), int)
    assert isinstance(total_tokens := attributes.pop(LLM_TOKEN_COUNT_TOTAL), int)
    assert prompt_tokens + completion_tokens == total_tokens
    assert not attributes
    agent_run_span = spans[3]  # root span
    assert agent_run_span.name == f"Agent.{run_method}"
    assert agent_run_span.status.is_ok
    attributes = dict(agent_run_span.attributes or {})
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == "CHAIN"
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    assert isinstance(attributes.pop(INPUT_VALUE), str)
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
    assert isinstance(attributes.pop(OUTPUT_VALUE), str)
    assert not attributes


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
def test_individual_component_without_child_components(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_haystack_instrumentation: Any,
) -> None:
    document_store = InMemoryDocumentStore()
    documents = [
        Document(content="There are over 7,000 languages spoken around the world today."),
        Document(
            content="Elephants have been observed to behave in a way that indicates "
            "a high level of self-awareness, such as recognizing themselves "
            "in mirrors."
        ),
        Document(
            content="In certain parts of the world, like the Maldives, Puerto Rico, "
            "and San Diego, you can witness the phenomenon of bioluminescent"
            " waves."
        ),
    ]
    document_store.write_documents(documents=documents)

    retriever = InMemoryBM25Retriever(document_store=document_store)
    results = retriever.run(query="How many languages are spoken around the world today?")
    assert results.get("documents") is not None
    assert len(results["documents"]) == 3
    for document in results["documents"]:
        assert isinstance(document, Document)
        assert document.id is not None
        assert document.content_type == "text"
        assert isinstance(document.content, str)
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    retriever_span = spans[0]
    assert retriever_span.name == "InMemoryBM25Retriever.run"
    assert retriever_span.status.is_ok
    attributes = dict(retriever_span.attributes or {})
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == "RETRIEVER"
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    assert isinstance(attributes.pop(INPUT_VALUE), str)
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
    assert isinstance(attributes.pop(OUTPUT_VALUE), str)
    for i, document in enumerate(results["documents"]):
        prefix = f"{RETRIEVAL_DOCUMENTS}.{i}"
        assert isinstance(content := attributes.pop(f"{prefix}.{DOCUMENT_CONTENT}"), str)
        assert content == document.content
        assert isinstance(doc_id := attributes.pop(f"{prefix}.{DOCUMENT_ID}"), str)
        assert doc_id == document.id
        assert isinstance(score := attributes.pop(f"{prefix}.{DOCUMENT_SCORE}"), float)
        assert score == document.score
        assert isinstance(attributes.pop(f"{prefix}.{DOCUMENT_METADATA}"), str)
    assert not attributes


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
async def test_individual_component_run_async_without_child_components(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_haystack_instrumentation: Any,
) -> None:
    document_store = InMemoryDocumentStore()
    documents = [
        Document(content="There are over 7,000 languages spoken around the world today."),
        Document(
            content="Elephants have been observed to behave in a way that indicates "
            "a high level of self-awareness, such as recognizing themselves "
            "in mirrors."
        ),
        Document(
            content="In certain parts of the world, like the Maldives, Puerto Rico, "
            "and San Diego, you can witness the phenomenon of bioluminescent"
            " waves."
        ),
    ]
    document_store.write_documents(documents=documents)

    retriever = InMemoryBM25Retriever(document_store=document_store)
    results = await retriever.run_async(
        query="How many languages are spoken around the world today?"
    )
    assert results.get("documents") is not None
    assert len(results["documents"]) == 3
    for document in results["documents"]:
        assert isinstance(document, Document)
        assert document.id is not None
        assert document.content_type == "text"
        assert isinstance(document.content, str)
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    retriever_span = spans[0]
    assert retriever_span.name == "InMemoryBM25Retriever.run_async"
    assert retriever_span.status.is_ok
    attributes = dict(retriever_span.attributes or {})
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == "RETRIEVER"
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    assert isinstance(attributes.pop(INPUT_VALUE), str)
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
    assert isinstance(attributes.pop(OUTPUT_VALUE), str)
    for i, document in enumerate(results["documents"]):
        prefix = f"{RETRIEVAL_DOCUMENTS}.{i}"
        assert isinstance(content := attributes.pop(f"{prefix}.{DOCUMENT_CONTENT}"), str)
        assert content == document.content
        assert isinstance(doc_id := attributes.pop(f"{prefix}.{DOCUMENT_ID}"), str)
        assert doc_id == document.id
        assert isinstance(score := attributes.pop(f"{prefix}.{DOCUMENT_SCORE}"), float)
        assert score == document.score
        assert isinstance(attributes.pop(f"{prefix}.{DOCUMENT_METADATA}"), str)
    assert not attributes


@pytest.fixture
def openai_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-")


@pytest.fixture
def serperdev_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SERPERDEV_API_KEY", "sk-")


@pytest.fixture
def cohere_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COHERE_API_KEY", "sk-")


def _is_vector(
    value: Any,
) -> TypeGuard[Sequence[Union[int, float]]]:
    """
    Checks for sequences of numbers.
    """

    is_sequence_of_numbers = isinstance(value, Sequence) and all(
        map(lambda x: isinstance(x, (int, float)), value)
    )
    return is_sequence_of_numbers


class TestInferLLMProviderFromClassName:
    def test_returns_none_when_instance_is_none(self) -> None:
        result = infer_llm_provider_from_class_name(None)
        assert result is None

    @pytest.mark.parametrize(
        "class_name, model_id, expected",
        [
            (
                "HuggingFaceAPIGenerator",
                "anthropic/claude-3-opus",
                OpenInferenceLLMProviderValues.ANTHROPIC,
            ),
            ("HuggingFaceAPIGenerator", "openai/gpt-4", OpenInferenceLLMProviderValues.OPENAI),
            ("HuggingFaceAPIGenerator", "azure/gpt-4", OpenInferenceLLMProviderValues.AZURE),
            ("HuggingFaceAPIGenerator", "cohere/command-r", OpenInferenceLLMProviderValues.COHERE),
            (
                "HuggingFaceAPIChatGenerator",
                "anthropic/claude-3",
                OpenInferenceLLMProviderValues.ANTHROPIC,
            ),
            (
                "HuggingFaceAPIChatGenerator",
                "openai/gpt-3.5",
                OpenInferenceLLMProviderValues.OPENAI,
            ),
        ],
    )
    def test_litellm_models_with_valid_provider_prefix(
        self, class_name: str, model_id: str, expected: OpenInferenceLLMProviderValues
    ) -> None:
        mock_instance = MagicMock()
        mock_instance.__class__.__name__ = class_name
        mock_instance.api_params = {"model": model_id}

        result = infer_llm_provider_from_class_name(mock_instance)
        assert result == expected

    @pytest.mark.parametrize(
        "class_name, model_id",
        [
            ("HuggingFaceAPIGenerator", "invalid_provider/some-model"),
            ("HuggingFaceAPIGenerator", "gpt-4"),
            ("HuggingFaceAPIChatGenerator", "unknown/model"),
        ],
    )
    def test_litellm_models_with_invalid_model_id_returns_none(
        self, class_name: str, model_id: str
    ) -> None:
        mock_instance = MagicMock()
        mock_instance.__class__.__name__ = class_name
        mock_instance.api_params = {"model": model_id}

        result = infer_llm_provider_from_class_name(mock_instance)
        assert result is None

    @pytest.mark.parametrize(
        "model_id",
        [None, 12345, [], {}],
    )
    def test_litellm_model_with_invalid_model_id_type(self, model_id: Any) -> None:
        mock_instance = MagicMock()
        mock_instance.__class__.__name__ = "HuggingFaceAPIGenerator"
        mock_instance.api_params = {"model": model_id}

        result = infer_llm_provider_from_class_name(mock_instance)
        assert result is None

    def test_litellm_model_with_missing_model_id_attribute(self) -> None:
        mock_instance = MagicMock()
        mock_instance.__class__.__name__ = "HuggingFaceAPIGenerator"
        del mock_instance.api_params

        result = infer_llm_provider_from_class_name(mock_instance)
        assert result is None

    @pytest.mark.parametrize(
        "class_name, expected",
        [
            ("OpenAIGenerator", OpenInferenceLLMProviderValues.OPENAI),
            ("DALLEImageGenerator", OpenInferenceLLMProviderValues.OPENAI),
            ("OpenAIChatGenerator", OpenInferenceLLMProviderValues.OPENAI),
            ("AzureOpenAIGenerator", OpenInferenceLLMProviderValues.AZURE),
            ("AzureOpenAIChatGenerator", OpenInferenceLLMProviderValues.AZURE),
        ],
    )
    def test_known_server_models_return_expected_provider(
        self, class_name: str, expected: OpenInferenceLLMProviderValues
    ) -> None:
        mock_instance = MagicMock()
        mock_instance.__class__.__name__ = class_name

        result = infer_llm_provider_from_class_name(mock_instance)
        assert result == expected

    @pytest.mark.parametrize(
        "class_name",
        ["InferenceClientModel", "UnknownModelClass", "CustomModel"],
    )
    def test_unknown_or_special_class_names_return_none(self, class_name: str) -> None:
        mock_instance = MagicMock()
        mock_instance.__class__.__name__ = class_name

        result = infer_llm_provider_from_class_name(mock_instance)
        assert result is None


class TestInferLLMSystemFromModel:
    @pytest.mark.parametrize(
        "model_name",
        [None, ""],
    )
    def test_returns_none_for_invalid_input(self, model_name: Optional[str]) -> None:
        result = infer_llm_system_from_model(model_name)
        assert result is None

    @pytest.mark.parametrize(
        "model_name, expected",
        [
            # OpenAI
            ("gpt-4", OpenInferenceLLMSystemValues.OPENAI),
            ("gpt-4-turbo-preview", OpenInferenceLLMSystemValues.OPENAI),
            ("gpt.3.5.turbo", OpenInferenceLLMSystemValues.OPENAI),
            ("o1-preview", OpenInferenceLLMSystemValues.OPENAI),
            ("o3-mini", OpenInferenceLLMSystemValues.OPENAI),
            ("o4-turbo", OpenInferenceLLMSystemValues.OPENAI),
            ("text-embedding-ada-002", OpenInferenceLLMSystemValues.OPENAI),
            ("davinci-002", OpenInferenceLLMSystemValues.OPENAI),
            ("curie", OpenInferenceLLMSystemValues.OPENAI),
            ("babbage", OpenInferenceLLMSystemValues.OPENAI),
            ("ada", OpenInferenceLLMSystemValues.OPENAI),
            ("azure_openai/gpt-4", OpenInferenceLLMSystemValues.OPENAI),
            ("azure_ai/some-model", OpenInferenceLLMSystemValues.OPENAI),
            ("azure/deployment-name", OpenInferenceLLMSystemValues.OPENAI),
            # Anthropic
            ("anthropic.claude-v2", OpenInferenceLLMSystemValues.ANTHROPIC),
            ("anthropic/claude-3-opus", OpenInferenceLLMSystemValues.ANTHROPIC),
            ("claude-3-sonnet", OpenInferenceLLMSystemValues.ANTHROPIC),
            ("claude-3-opus-20240229", OpenInferenceLLMSystemValues.ANTHROPIC),
            ("google_anthropic_vertex/claude-3", OpenInferenceLLMSystemValues.ANTHROPIC),
            # Cohere
            ("cohere.command-r", OpenInferenceLLMSystemValues.COHERE),
            ("command-r-plus", OpenInferenceLLMSystemValues.COHERE),
            ("cohere/embed-english-v3", OpenInferenceLLMSystemValues.COHERE),
            # Mistral
            ("mistralai/mistral-large", OpenInferenceLLMSystemValues.MISTRALAI),
            ("mixtral-8x7b", OpenInferenceLLMSystemValues.MISTRALAI),
            ("mistral-small", OpenInferenceLLMSystemValues.MISTRALAI),
            ("pixtral-12b", OpenInferenceLLMSystemValues.MISTRALAI),
            # VertexAI
            ("google_vertexai/gemini-pro", OpenInferenceLLMSystemValues.VERTEXAI),
            ("google_genai/gemini-pro", OpenInferenceLLMSystemValues.VERTEXAI),
            ("vertexai/gemini-ultra", OpenInferenceLLMSystemValues.VERTEXAI),
            ("vertex_ai/palm-2", OpenInferenceLLMSystemValues.VERTEXAI),
            ("vertex/bison", OpenInferenceLLMSystemValues.VERTEXAI),
            ("gemini-1.5-pro", OpenInferenceLLMSystemValues.VERTEXAI),
            ("google/palm-2", OpenInferenceLLMSystemValues.VERTEXAI),
        ],
    )
    def test_known_model_names_return_expected_system(
        self, model_name: str, expected: OpenInferenceLLMSystemValues
    ) -> None:
        result = infer_llm_system_from_model(model_name)
        assert result == expected

    @pytest.mark.parametrize(
        "model_name",
        [
            "unknown-model-xyz",
            "custom-llm-v1",
            "my-gpt-4-custom",
        ],
    )
    def test_unknown_model_names_return_none(self, model_name: str) -> None:
        result = infer_llm_system_from_model(model_name)
        assert result is None

    def test_case_insensitive_matching(self) -> None:
        result = infer_llm_system_from_model("GPT-4-Turbo")
        assert result == OpenInferenceLLMSystemValues.OPENAI


CHAIN = OpenInferenceSpanKindValues.CHAIN.value
EMBEDDING = OpenInferenceSpanKindValues.EMBEDDING.value
LLM = OpenInferenceSpanKindValues.LLM.value
RERANKER = OpenInferenceSpanKindValues.RERANKER.value
RETRIEVER = OpenInferenceSpanKindValues.RETRIEVER.value

JSON = OpenInferenceMimeTypeValues.JSON.value
TEXT = OpenInferenceMimeTypeValues.TEXT.value

DOCUMENT_CONTENT = DocumentAttributes.DOCUMENT_CONTENT
DOCUMENT_ID = DocumentAttributes.DOCUMENT_ID
DOCUMENT_METADATA = DocumentAttributes.DOCUMENT_METADATA
DOCUMENT_SCORE = DocumentAttributes.DOCUMENT_SCORE
EMBEDDING_EMBEDDINGS = SpanAttributes.EMBEDDING_EMBEDDINGS
EMBEDDING_INVOCATION_PARAMETERS = SpanAttributes.EMBEDDING_INVOCATION_PARAMETERS
EMBEDDING_MODEL_NAME = SpanAttributes.EMBEDDING_MODEL_NAME
EMBEDDING_TEXT = EmbeddingAttributes.EMBEDDING_TEXT
EMBEDDING_VECTOR = EmbeddingAttributes.EMBEDDING_VECTOR
INPUT_MIME_TYPE = SpanAttributes.INPUT_MIME_TYPE
INPUT_VALUE = SpanAttributes.INPUT_VALUE
LLM_INPUT_MESSAGES = SpanAttributes.LLM_INPUT_MESSAGES
LLM_INVOCATION_PARAMETERS = SpanAttributes.LLM_INVOCATION_PARAMETERS
LLM_MODEL_NAME = SpanAttributes.LLM_MODEL_NAME
LLM_PROVIDER = SpanAttributes.LLM_PROVIDER
LLM_SYSTEM = SpanAttributes.LLM_SYSTEM
LLM_OUTPUT_MESSAGES = SpanAttributes.LLM_OUTPUT_MESSAGES
LLM_PROMPTS = SpanAttributes.LLM_PROMPTS
LLM_PROMPT_TEMPLATE = SpanAttributes.LLM_PROMPT_TEMPLATE
LLM_PROMPT_TEMPLATE_VARIABLES = SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES
LLM_PROMPT_TEMPLATE_VERSION = SpanAttributes.LLM_PROMPT_TEMPLATE_VERSION
LLM_TOKEN_COUNT_COMPLETION = SpanAttributes.LLM_TOKEN_COUNT_COMPLETION
LLM_TOKEN_COUNT_PROMPT = SpanAttributes.LLM_TOKEN_COUNT_PROMPT
LLM_TOKEN_COUNT_TOTAL = SpanAttributes.LLM_TOKEN_COUNT_TOTAL
MESSAGE_CONTENT = MessageAttributes.MESSAGE_CONTENT
MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON = MessageAttributes.MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON
MESSAGE_FUNCTION_CALL_NAME = MessageAttributes.MESSAGE_FUNCTION_CALL_NAME
MESSAGE_ROLE = MessageAttributes.MESSAGE_ROLE
MESSAGE_TOOL_CALLS = MessageAttributes.MESSAGE_TOOL_CALLS
METADATA = SpanAttributes.METADATA
OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
OUTPUT_MIME_TYPE = SpanAttributes.OUTPUT_MIME_TYPE
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
RERANKER_INPUT_DOCUMENTS = RerankerAttributes.RERANKER_INPUT_DOCUMENTS
RERANKER_MODEL_NAME = RerankerAttributes.RERANKER_MODEL_NAME
RERANKER_OUTPUT_DOCUMENTS = RerankerAttributes.RERANKER_OUTPUT_DOCUMENTS
RERANKER_QUERY = RerankerAttributes.RERANKER_QUERY
RERANKER_TOP_K = RerankerAttributes.RERANKER_TOP_K
RETRIEVAL_DOCUMENTS = SpanAttributes.RETRIEVAL_DOCUMENTS
SESSION_ID = SpanAttributes.SESSION_ID
TAG_TAGS = SpanAttributes.TAG_TAGS
TOOL_CALL_FUNCTION_ARGUMENTS_JSON = ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON
TOOL_CALL_FUNCTION_NAME = ToolCallAttributes.TOOL_CALL_FUNCTION_NAME
LLM_PROMPT_TEMPLATE = SpanAttributes.LLM_PROMPT_TEMPLATE
LLM_PROMPT_TEMPLATE_VARIABLES = SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES
USER_ID = SpanAttributes.USER_ID

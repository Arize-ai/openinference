import json
from datetime import datetime
from typing import Any, Dict, Generator, List, Optional, Sequence, Union

import pytest
from haystack import Document
from haystack.components.builders import ChatPromptBuilder
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.embedders import (
    OpenAIDocumentEmbedder,
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.components.generators import OpenAIGenerator
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.retrievers.in_memory import (
    InMemoryBM25Retriever,
    InMemoryEmbeddingRetriever,
)
from haystack.components.websearch.serper_dev import SerperDevWebSearch
from haystack.core.pipeline.pipeline import Pipeline
from haystack.dataclasses import ChatMessage, ChatRole
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.utils.auth import Secret
from haystack_integrations.components.rankers.cohere import (  # type: ignore[import-untyped]
    CohereRanker,
)
from openai import AuthenticationError
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from typing_extensions import TypeGuard

from openinference.instrumentation import OITracer, suppress_tracing, using_attributes
from openinference.instrumentation.haystack import HaystackInstrumentor
from openinference.semconv.trace import (
    DocumentAttributes,
    EmbeddingAttributes,
    MessageAttributes,
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


def fake_OpenAIGenerator_run(
    self: Any, prompt: str, generation_kwargs: Optional[Dict[str, Any]] = None, **kwargs: Any
) -> Dict[str, List[Union[str, Dict[str, Any]]]]:
    return {
        "replies": ["sorry, i have zero clue"],
        "meta": [
            {
                "model": "fake_model",
                "usage": {"completion_tokens": 10, "prompt_tokens": 5, "total_tokens": 15},
                "awesome_attribute": True,
                "another_attribute": 7,
            }
        ],
    }


def fake_OpenAIGenerator_run_chat(
    self: Any, messages: List[ChatMessage], generation_kwargs: Optional[Dict[str, Any]] = None
) -> Dict[str, List[ChatMessage]]:
    return {
        "replies": [
            ChatMessage(
                content="idk, sorry!",
                name=None,
                role=ChatRole.ASSISTANT,
                meta={
                    "model": "gpt-3.5-turbo-0613",
                    "index": 0,
                    "finish_reason": "stop",
                    "usage": {"prompt_tokens": 16, "completion_tokens": 61, "total_tokens": 77},
                },
            )
        ]
    }


def fake_SentenceTransformersTextEmbedder_run(
    self: Any, text: str, **kwargs: Any
) -> Dict[str, Any]:
    return {"embedding": [0.1, 0.2, 0.3]}


def fake_SentenceTransformersDocumentEmbedder_run(
    self: Any, documents: List[Document], **kwargs: Any
) -> Dict[str, Any]:
    return {"documents": [Document(content="I love pizza!", embedding=[0.1, 0.2, 0.3])]}


def fake_SentenceTransformersEmbedder_warm_up(self: Any) -> None:
    self.embedding_backend = None


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


def test_haystack_instrumentation(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_haystack_instrumentation: Any,
) -> None:
    # Configure document store and load dataset
    document_store = InMemoryDocumentStore()

    dataset = [
        {
            "content": "The first wonder of the world is the Great Pyramid of Giza",
            "meta": {"id": 1},
        },
        {
            "content": "The second wonder of the world is the Hanging Gardens of Babylon",
            "meta": {"id": 2},
        },
        {
            "content": "The third wonder of the world is the Statue of Zeus at Olympia",
            "meta": {"id": 3},
        },
    ]

    docs = [Document(content=doc["content"], meta=doc["meta"]) for doc in dataset]

    # Configure document embedder and store documents from dataset
    doc_embedder = SentenceTransformersDocumentEmbedder(model="fake_model")
    doc_embedder.warm_up = fake_SentenceTransformersEmbedder_warm_up.__get__(
        doc_embedder, SentenceTransformersDocumentEmbedder
    )

    doc_embedder.warm_up()
    doc_embedder.run = fake_SentenceTransformersDocumentEmbedder_run.__get__(
        doc_embedder, SentenceTransformersDocumentEmbedder
    )
    docs_with_embeddings = doc_embedder.run(docs)

    document_store.write_documents(docs_with_embeddings["documents"])

    # Configure text embedder (prompt)
    text_embedder = SentenceTransformersTextEmbedder(model="fake_model")

    text_embedder.warm_up = fake_SentenceTransformersEmbedder_warm_up.__get__(
        text_embedder, SentenceTransformersDocumentEmbedder
    )
    text_embedder.run = fake_SentenceTransformersTextEmbedder_run.__get__(
        text_embedder, SentenceTransformersTextEmbedder
    )

    # Configure retriever
    retriever = InMemoryEmbeddingRetriever(document_store)

    template = """
    Given the following information, answer the question.
    Context:
    {% for document in documents %}
        {{ document.content }}
    {% endfor %}
    Question: {{question}}
    Answer:
    """

    prompt_builder = PromptBuilder(template=template)
    llm = OpenAIGenerator(api_key=Secret.from_token("TOTALLY_REAL_API_KEY"))

    llm.run = fake_OpenAIGenerator_run.__get__(llm, OpenAIGenerator)

    basic_rag_pipeline = Pipeline()
    basic_rag_pipeline.add_component("text_embedder", text_embedder)
    basic_rag_pipeline.add_component("retriever", retriever)
    basic_rag_pipeline.add_component("prompt_builder", prompt_builder)
    basic_rag_pipeline.add_component("llm", llm)

    # Connect necessary pipeline components to each other
    basic_rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    basic_rag_pipeline.connect("retriever", "prompt_builder.documents")
    basic_rag_pipeline.connect("prompt_builder", "llm")

    q = "What's Natural Language Processing? Be brief."

    basic_rag_pipeline.run(
        {
            "text_embedder": {"text": q},
            "prompt_builder": {"question": q},
        }
    )

    spans = in_memory_span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "SentenceTransformersTextEmbedder (text_embedder)",
        "InMemoryEmbeddingRetriever (retriever)",
        "PromptBuilder (prompt_builder)",
        "OpenAIGenerator (llm)",
        "Pipeline",
    ]

    assert [
        span.attributes.get("openinference.span.kind") for span in spans if span and span.attributes
    ] == [
        EMBEDDING,
        RETRIEVER,
        LLM,
        LLM,
        CHAIN,
    ]


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
    assert span.name == "ChatPromptBuilder (prompt_builder)"
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
    assert span.name == "OpenAIChatGenerator (llm)"
    attributes = dict(span.attributes or {})
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == LLM
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    assert isinstance(attributes.pop(INPUT_VALUE), str)
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
    assert isinstance(attributes.pop(OUTPUT_VALUE), str)
    assert isinstance(llm_model_name := attributes.pop(LLM_MODEL_NAME), str)
    assert "gpt-4o" in llm_model_name
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
    assert span.name == "Pipeline"
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
        "InMemoryBM25Retriever (retriever)",
        "Pipeline",
    ]

    assert [
        span.attributes.get("openinference.span.kind") for span in spans if span and span.attributes
    ] == [
        RETRIEVER,
        CHAIN,
    ]


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
    assert span.name == "OpenAIChatGenerator (llm)"
    assert span.status.is_ok
    assert not span.events
    attributes = dict(span.attributes or {})
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == "LLM"
    assert isinstance(llm_model_name := attributes.pop(LLM_MODEL_NAME), str)
    assert "gpt-4o" in llm_model_name
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


def test_instrument_and_uninstrument_methods_wrap_and_unwrap_expected_methods(
    tracer_provider: TracerProvider,
) -> None:
    HaystackInstrumentor().instrument(tracer_provider=tracer_provider)

    assert hasattr(Pipeline.run, "__wrapped__")
    assert hasattr(Pipeline._run_component, "__wrapped__")

    HaystackInstrumentor().uninstrument()

    assert not hasattr(Pipeline.run, "__wrapped__")
    assert not hasattr(Pipeline._run_component, "__wrapped__")


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
    assert span.name == "OpenAIChatGenerator (llm)"
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
    assert isinstance(model_name := attributes.pop(LLM_MODEL_NAME), str)
    assert "gpt-4o" in model_name
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
    assert span.name == "OpenAIGenerator (llm)"
    assert span.status.is_ok
    assert not span.events
    attributes = dict(span.attributes or {})
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == "LLM"
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    assert isinstance(input_value := attributes.pop(INPUT_VALUE), str)
    input_value_data = json.loads(input_value)
    assert input_value_data.get("prompt") == "Who won the World Cup in 2022? Answer in one word."
    assert isinstance(model_name := attributes.pop(LLM_MODEL_NAME), str)
    assert "gpt-4o" in model_name
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
    prompt_builder = PromptBuilder(template=default_template)
    pipe = Pipeline()
    pipe.add_component("prompt_builder", prompt_builder)
    output = pipe.run({"prompt_builder": prompt_builder_inputs})
    assert output == {"prompt_builder": {"prompt": "Where is Munich?"}}
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 2
    span = spans[0]
    assert span.name == "PromptBuilder (prompt_builder)"
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
    assert span.name == "CohereRanker (ranker)"
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
    assert span.name == "SerperDevWebSearch (websearch)"
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
    assert span.name == "OpenAIDocumentEmbedder (embedder)"
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
    with pytest.raises(AuthenticationError):
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


def test_instrumentor_uses_oitracer(
    setup_haystack_instrumentation: Any,
) -> None:
    assert isinstance(HaystackInstrumentor()._tracer, OITracer)


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
EMBEDDING_MODEL_NAME = SpanAttributes.EMBEDDING_MODEL_NAME
EMBEDDING_TEXT = EmbeddingAttributes.EMBEDDING_TEXT
EMBEDDING_VECTOR = EmbeddingAttributes.EMBEDDING_VECTOR
INPUT_MIME_TYPE = SpanAttributes.INPUT_MIME_TYPE
INPUT_VALUE = SpanAttributes.INPUT_VALUE
LLM_INPUT_MESSAGES = SpanAttributes.LLM_INPUT_MESSAGES
LLM_INVOCATION_PARAMETERS = SpanAttributes.LLM_INVOCATION_PARAMETERS
LLM_MODEL_NAME = SpanAttributes.LLM_MODEL_NAME
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

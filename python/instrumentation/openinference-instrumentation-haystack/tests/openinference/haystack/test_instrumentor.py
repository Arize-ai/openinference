from datetime import datetime
from typing import Any, Dict, Generator, List, Optional, Union

import pytest
from haystack import Document
from haystack.components.builders import ChatPromptBuilder
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.components.generators import OpenAIGenerator
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.retrievers.in_memory import (
    InMemoryBM25Retriever,
    InMemoryEmbeddingRetriever,
)
from haystack.components.routers import ConditionalRouter
from haystack.core.pipeline.pipeline import Pipeline
from haystack.dataclasses import ChatMessage, ChatRole
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.utils.auth import Secret
from openinference.instrumentation import OITracer, using_attributes
from openinference.instrumentation.haystack import HaystackInstrumentor
from openinference.semconv.trace import (
    DocumentAttributes,
    EmbeddingAttributes,
    MessageAttributes,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolCallAttributes,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


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
    self: Any, prompt: str, generation_kwargs: Optional[Dict[str, Any]] = None
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
def session_id() -> str:
    return "my-test-session-id"


@pytest.fixture()
def user_id() -> str:
    return "my-test-user-id"


@pytest.fixture()
def metadata() -> Dict[str, Any]:
    return {
        "test-int": 1,
        "test-str": "string",
        "test-list": [1, 2, 3],
        "test-dict": {
            "key-1": "val-1",
            "key-2": "val-2",
        },
    }


@pytest.fixture()
def tags() -> List[str]:
    return ["tag-1", "tag-2"]


@pytest.fixture
def prompt_template() -> str:
    return (
        "This is a test prompt template with int {var_int}, "
        "string {var_string}, and list {var_list}"
    )


@pytest.fixture
def prompt_template_version() -> str:
    return "v1.0"


@pytest.fixture
def prompt_template_variables() -> Dict[str, Any]:
    return {
        "var_int": 1,
        "var_str": "2",
        "var_list": [1, 2, 3],
    }


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


def _check_context_attributes(
    attributes: Any,
) -> None:
    assert attributes.get(SESSION_ID, None)
    assert attributes.get(USER_ID, None)
    assert attributes.get(METADATA, None)
    assert attributes.get(TAG_TAGS, None)
    assert attributes.get(LLM_PROMPT_TEMPLATE, None)
    assert attributes.get(LLM_PROMPT_TEMPLATE_VERSION, None)
    assert attributes.get(LLM_PROMPT_TEMPLATE_VARIABLES, None)


def test_haystack_instrumentation(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_haystack_instrumentation: Any,
    session_id: str,
    user_id: str,
    metadata: Dict[str, Any],
    tags: List[str],
    prompt_template: str,
    prompt_template_version: str,
    prompt_template_variables: Dict[str, Any],
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

    with using_attributes(
        session_id=session_id,
        user_id=user_id,
        metadata=metadata,
        tags=tags,
        prompt_template=prompt_template,
        prompt_template_version=prompt_template_version,
        prompt_template_variables=prompt_template_variables,
    ):
        basic_rag_pipeline.run(
            {
                "text_embedder": {"text": q},
                "prompt_builder": {"question": q},
            }
        )

    spans = in_memory_span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "SentenceTransformersTextEmbedder",
        "InMemoryEmbeddingRetriever",
        "PromptBuilder",
        "OpenAIGenerator",
        "Pipeline",
    ]

    assert [
        span.attributes.get("openinference.span.kind") for span in spans if span and span.attributes
    ] == [
        "EMBEDDING",
        "RETRIEVER",
        "CHAIN",
        "LLM",
        "CHAIN",
    ]

    for span in spans:
        att = span.attributes
        _check_context_attributes(
            att,
        )


def test_haystack_instrumentation_chat(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_haystack_instrumentation: Any,
    session_id: str,
    user_id: str,
    metadata: Dict[str, Any],
    tags: List[str],
    prompt_template: str,
    prompt_template_version: str,
    prompt_template_variables: Dict[str, Any],
) -> None:
    prompt_builder = ChatPromptBuilder()

    llm = OpenAIChatGenerator(api_key=Secret.from_token("fake_key"), model="fake_model")
    llm.run = fake_OpenAIGenerator_run_chat.__get__(llm, OpenAIChatGenerator)

    pipe = Pipeline()

    pipe.add_component("prompt_builder", prompt_builder)
    pipe.add_component("llm", llm)

    pipe.connect("prompt_builder.prompt", "llm.messages")

    location = "Berlin"
    messages = [
        ChatMessage.from_system("Try and be super useful."),
        ChatMessage.from_user("Tell me about {{location}}"),
    ]

    with using_attributes(
        session_id=session_id,
        user_id=user_id,
        metadata=metadata,
        tags=tags,
        prompt_template=prompt_template,
        prompt_template_version=prompt_template_version,
        prompt_template_variables=prompt_template_variables,
    ):
        pipe.run(
            data={
                "prompt_builder": {
                    "template_variables": {"location": location},
                    "template": messages,
                }
            }
        )

    spans = in_memory_span_exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        "ChatPromptBuilder",
        "OpenAIChatGenerator",
        "Pipeline",
    ]

    assert [
        span.attributes.get("openinference.span.kind") for span in spans if span and span.attributes
    ] == [
        "CHAIN",
        "LLM",
        "CHAIN",
    ]

    for span in spans:
        att = span.attributes
        _check_context_attributes(
            att,
        )


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
        "InMemoryBM25Retriever",
        "Pipeline",
    ]

    assert [
        span.attributes.get("openinference.span.kind") for span in spans if span and span.attributes
    ] == [
        "RETRIEVER",
        "CHAIN",
    ]


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
def test_haystack_tool_calling_llm_span_has_expected_attributes(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_haystack_instrumentation: Any,
    openai_api_key: str,
) -> None:
    WEATHER_INFO = {
        "Berlin": {"weather": "mostly sunny", "temperature": 7, "unit": "celsius"},
        "Paris": {"weather": "mostly cloudy", "temperature": 8, "unit": "celsius"},
        "Rome": {"weather": "sunny", "temperature": 14, "unit": "celsius"},
        "Madrid": {"weather": "sunny", "temperature": 10, "unit": "celsius"},
        "London": {"weather": "cloudy", "temperature": 9, "unit": "celsius"},
    }

    def get_current_weather(location: str) -> Any:
        if location in WEATHER_INFO:
            return WEATHER_INFO[location]

        # fallback data
        else:
            return {"weather": "sunny", "temperature": 21.8, "unit": "fahrenheit"}

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

    messages = [
        ChatMessage.from_user("What is the weather in Berlin"),
    ]

    chat_generator = OpenAIChatGenerator(model="gpt-3.5-turbo")

    pipe = Pipeline()
    pipe.add_component("llm", chat_generator)
    pipe.run({"llm": {"messages": messages, "generation_kwargs": {"tools": tools}}})

    spans = in_memory_span_exporter.get_finished_spans()

    assert spans


def test_haystack_uninstrumentation(
    tracer_provider: TracerProvider,
) -> None:
    # Instrumenting Haystack
    HaystackInstrumentor().instrument(tracer_provider=tracer_provider)

    # Ensure methods are wrapped
    assert hasattr(Pipeline.run, "__wrapped__")
    assert hasattr(Pipeline._run_component, "__wrapped__")

    # Uninstrumenting Haystack
    HaystackInstrumentor().uninstrument()

    # Ensure methods are not wrapped
    assert not hasattr(Pipeline.run, "__wrapped__")
    assert not hasattr(Pipeline._run_component, "__wrapped__")


def test_haystack_conditional_router(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_haystack_instrumentation: Any,
) -> None:
    routes = [
        {
            "condition": "{{query|length > 10}}",
            "output": "{{query}}",
            "output_name": "ok_query",
            "output_type": str,
        },
        {
            "condition": "{{query|length <= 10}}",
            "output": "query is too short: {{query}}",
            "output_name": "too_short_query",
            "output_type": str,
        },
    ]
    router = ConditionalRouter(routes=routes)

    llm = OpenAIGenerator(api_key=Secret.from_token("TOTALLY_REAL_API_KEY"))
    llm.run = fake_OpenAIGenerator_run.__get__(llm, OpenAIGenerator)

    pipe = Pipeline()
    pipe.add_component("router", router)
    pipe.add_component("prompt_builder", PromptBuilder("Answer the following query. {{query}}"))
    pipe.add_component("generator", llm)
    pipe.connect("router.ok_query", "prompt_builder.query")
    pipe.connect("prompt_builder", "generator")

    pipe.run(data={"router": {"query": "Berlin"}})

    pipe.run(data={"router": {"query": "What is the capital of Italy?"}})

    spans = in_memory_span_exporter.get_finished_spans()

    assert spans


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
    assert "argentina" in response["llm"]["replies"][0].content.lower()
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 2
    span = spans[0]
    attributes = dict(span.attributes or {})
    assert attributes.get(OPENINFERENCE_SPAN_KIND) == "LLM"
    assert (
        attributes.get(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}")
        == "Answer user questions succinctly"
    )
    assert attributes.get(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "system"
    assert (
        attributes.get(f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_CONTENT}") == "What can I help you with?"
    )
    assert attributes.get(f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_ROLE}") == "assistant"

    assert (
        attributes.get(f"{LLM_INPUT_MESSAGES}.2.{MESSAGE_CONTENT}")
        == "Who won the World Cup in 2022? Answer in one word."
    )
    assert attributes.get(f"{LLM_INPUT_MESSAGES}.2.{MESSAGE_ROLE}") == "user"
    assert isinstance(
        (output_message_content := attributes.get(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}")),
        str,
    )
    assert "argentina" in output_message_content.lower()
    assert attributes.get(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
    assert isinstance(attributes.get(LLM_TOKEN_COUNT_PROMPT), int)
    assert isinstance(attributes.get(LLM_TOKEN_COUNT_COMPLETION), int)
    assert isinstance(attributes.get(LLM_TOKEN_COUNT_TOTAL), int)


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
    attributes = dict(span.attributes or {})
    assert attributes.get(OPENINFERENCE_SPAN_KIND) == "LLM"
    assert (
        attributes.get(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}")
        == "Who won the World Cup in 2022? Answer in one word."
    )
    assert attributes.get(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "user"
    assert isinstance(
        (output_message_content := attributes.get(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}")),
        str,
    )
    assert "argentina" in output_message_content.lower()
    assert attributes.get(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
    assert isinstance(attributes.get(LLM_TOKEN_COUNT_PROMPT), int)
    assert isinstance(attributes.get(LLM_TOKEN_COUNT_COMPLETION), int)
    assert isinstance(attributes.get(LLM_TOKEN_COUNT_TOTAL), int)


# Ensure we're using the common OITracer from common openinference-instrumentation pkg
def test_oitracer(
    setup_haystack_instrumentation: Any,
) -> None:
    assert isinstance(HaystackInstrumentor()._tracer, OITracer)


@pytest.fixture
def openai_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-")


CHAIN = OpenInferenceSpanKindValues.CHAIN
LLM = OpenInferenceSpanKindValues.LLM
RETRIEVER = OpenInferenceSpanKindValues.RETRIEVER

JSON = OpenInferenceMimeTypeValues.JSON

DOCUMENT_CONTENT = DocumentAttributes.DOCUMENT_CONTENT
DOCUMENT_ID = DocumentAttributes.DOCUMENT_ID
DOCUMENT_METADATA = DocumentAttributes.DOCUMENT_METADATA
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
RETRIEVAL_DOCUMENTS = SpanAttributes.RETRIEVAL_DOCUMENTS
SESSION_ID = SpanAttributes.SESSION_ID
TAG_TAGS = SpanAttributes.TAG_TAGS
TOOL_CALL_FUNCTION_ARGUMENTS_JSON = ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON
TOOL_CALL_FUNCTION_NAME = ToolCallAttributes.TOOL_CALL_FUNCTION_NAME
LLM_PROMPT_TEMPLATE = SpanAttributes.LLM_PROMPT_TEMPLATE
LLM_PROMPT_TEMPLATE_VARIABLES = SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES
USER_ID = SpanAttributes.USER_ID

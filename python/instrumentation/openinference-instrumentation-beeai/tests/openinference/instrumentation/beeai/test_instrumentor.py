import os
from typing import Any, Generator

import pytest
from beeai_framework.adapters.openai import OpenAIChatModel
from beeai_framework.backend import SystemMessage, UserMessage
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool
from beeai_framework.tools.weather import OpenMeteoTool
from opentelemetry import trace as trace_api
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from openinference.instrumentation.beeai import BeeAIInstrumentor
from openinference.semconv.trace import (
    DocumentAttributes,
    EmbeddingAttributes,
    MessageAttributes,
    MessageContentAttributes,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    RerankerAttributes,
    SpanAttributes,
    ToolAttributes,
    ToolCallAttributes,
)

os.environ["OPENAI_API_KEY"] = "sk-proj-ubPK"


@pytest.fixture(scope="module")
def vcr_config() -> dict[str, Any]:
    return {
        "record_mode": "once",
        "filter_headers": [
            "authorization",
            "api-key",
            "x-api-key",
        ],
        # Match requests on these attributes
        "match_on": ["method", "scheme", "host", "port", "path", "query"],
        # Decode compressed responses
        "decode_compressed_response": True,
        # Allow recording of requests
        "allow_playback_repeats": True,
    }


@pytest.fixture
def openai_api_key(monkeypatch: pytest.MonkeyPatch) -> str:
    api_key = "sk-fake-key"
    monkeypatch.setenv("OPENAI_API_KEY", api_key)
    return api_key


@pytest.fixture
def serperdev_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SERPERDEV_API_KEY", "sk-fake-key")


@pytest.fixture
def cohere_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COHERE_API_KEY", "sk-fake-key")


@pytest.fixture()
def in_memory_span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture(scope="session", autouse=True)
def rebuild_beeai_pydantic_models() -> None:
    from beeai_framework.backend.chat import _ChatModelKwargsAdapter

    # Required for Python 3.14 + Pydantic v2
    _ChatModelKwargsAdapter.rebuild(force=True)


@pytest.fixture()
async def tracer_provider(
    in_memory_span_exporter: InMemorySpanExporter,
) -> trace_api.TracerProvider:
    resource = Resource(attributes={})
    tracer_provider = trace_sdk.TracerProvider(resource=resource)
    span_processor = SimpleSpanProcessor(span_exporter=in_memory_span_exporter)
    tracer_provider.add_span_processor(span_processor=span_processor)
    return tracer_provider


@pytest.fixture(autouse=True)
def instrument(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> Generator[None, None, None]:
    BeeAIInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    BeeAIInstrumentor().uninstrument()
    in_memory_span_exporter.clear()


@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_instrumentor(
    in_memory_span_exporter: InMemorySpanExporter,
    openai_api_key: str,
) -> None:
    prompt = "Hello, How are you?"
    llm = OpenAIChatModel(model_id="gpt-4o-mini")
    await llm.run([UserMessage(prompt)], stream=False, max_tokens=10)
    spans = list(in_memory_span_exporter.get_finished_spans())
    spans.sort(key=lambda span: span.start_time or 0)
    assert len(spans) == 1
    open_ai_span = [span for span in spans if span.name == "OpenAIChatModel"][0]
    attrs: dict[str, Any] = dict(open_ai_span.attributes or {})
    assert attrs.pop(INPUT_MIME_TYPE) == JSON
    assert attrs.pop(LLM_TOKEN_COUNT_COMPLETION) == 10
    assert attrs.pop(LLM_TOKEN_COUNT_PROMPT) == 13
    assert attrs.pop(LLM_TOKEN_COUNT_TOTAL) == 23
    assert attrs.pop(SpanAttributes.LLM_PROVIDER) == "openai"
    assert attrs.pop(SpanAttributes.LLM_COST_COMPLETION) == 0.0001
    assert attrs.pop(SpanAttributes.LLM_COST_PROMPT) == 3.2500000000000004e-05
    assert attrs.pop(SpanAttributes.LLM_COST_TOTAL) == 0.00013250000000000002

    assert attrs.pop(OPENINFERENCE_SPAN_KIND) == LLM
    assert attrs.pop(OUTPUT_MIME_TYPE) == TEXT
    assert attrs.pop(LLM_MODEL_NAME) == "gpt-4o-mini"
    input_messages = f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0"
    assert attrs.pop(f"{input_messages}.{MESSAGE_CONTENT_TEXT}") == "Hello, How are you?"
    assert attrs.pop(f"{input_messages}.{MESSAGE_CONTENT_TYPE}") == "text"
    assert attrs.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "user"
    output_messages = f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0"
    assert attrs.pop(f"{output_messages}.{MESSAGE_CONTENT_TEXT}").startswith(
        "Hello! I'm just a computer program"
    )
    assert attrs.pop(f"{output_messages}.{MESSAGE_CONTENT_TYPE}") == "text"
    assert attrs.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
    assert attrs.pop(INPUT_VALUE).startswith(
        '{"input": [{"role": "user", "content": [{"type": "text"'
    )
    assert attrs.pop(LLM_INVOCATION_PARAMETERS).startswith('{"temperature": 0.0, "max_tokens": 10')
    assert attrs.pop(OUTPUT_VALUE).startswith("Hello! I'm just a computer program")
    assert attrs.pop(f"{METADATA}.chunks_count") == 1
    assert attrs.pop(f"{METADATA}.class_name") == "OpenAIChatModel"
    assert attrs.pop(f"{METADATA}.usage.completion_tokens") == "10"
    assert attrs.pop(f"{METADATA}.usage.prompt_tokens") == "13"
    assert attrs.pop(f"{METADATA}.usage.total_tokens") == "23"
    assert attrs.pop(f"{METADATA}.usage.cached_creation_tokens") == "0"
    assert attrs.pop(f"{METADATA}.usage.cached_prompt_tokens") == "0"
    assert attrs == {}, f"Unexpected keys found: {attrs}"


@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_llm_with_tools(
    in_memory_span_exporter: InMemorySpanExporter,
    openai_api_key: str,
) -> None:
    prompt = "What's the current weather in Las Vegas?"
    llm = OpenAIChatModel(model_id="gpt-4o-mini")
    response = await llm.run(
        [
            SystemMessage("Answer the user's question using the available tools."),
            UserMessage(prompt),
        ],
        tools=[DuckDuckGoSearchTool(), OpenMeteoTool()],
        tool_choice="auto",
        stream=False,
        max_tokens=100,
    )
    assert response.get_text_content() is not None
    spans = list(in_memory_span_exporter.get_finished_spans())
    spans.sort(key=lambda span: span.start_time or 0)
    assert len(spans) == 1
    open_ai_span = [span for span in spans if span.name == "OpenAIChatModel"][0]
    attrs: dict[str, Any] = dict(open_ai_span.attributes or {})
    assert attrs.pop(INPUT_MIME_TYPE) == JSON
    assert attrs.pop(OUTPUT_MIME_TYPE) == TEXT
    assert attrs.pop(SpanAttributes.LLM_PROVIDER) == "openai"
    assert attrs.pop(LLM_MODEL_NAME) == "gpt-4o-mini"
    assert attrs.pop(LLM_TOKEN_COUNT_COMPLETION) == 39
    assert attrs.pop(LLM_TOKEN_COUNT_PROMPT) == 252
    assert attrs.pop(LLM_TOKEN_COUNT_TOTAL) == 291
    assert attrs.pop(SpanAttributes.LLM_COST_COMPLETION) == 0.00039000000000000005
    assert attrs.pop(SpanAttributes.LLM_COST_PROMPT) == 0.00063
    assert attrs.pop(SpanAttributes.LLM_COST_TOTAL) == 0.00102

    assert attrs.pop(OPENINFERENCE_SPAN_KIND) == LLM

    assert attrs.pop(LLM_INVOCATION_PARAMETERS).startswith('{"temperature": 0.0, "max_tokens": 100')
    assert attrs.pop(INPUT_VALUE).startswith('{"input": [{"role": "system", "content": "Answer the')
    input_msg_0 = f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0"
    assert attrs.pop(f"{input_msg_0}.{MESSAGE_CONTENT_TEXT}") == (
        "Answer the user's question using the available tools."
    )
    assert attrs.pop(f"{input_msg_0}.{MESSAGE_CONTENT_TYPE}") == "text"
    assert attrs.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "system"

    input_msg_1 = f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_CONTENTS}.0"
    assert attrs.pop(f"{input_msg_1}.{MESSAGE_CONTENT_TEXT}") == (
        "What's the current weather in Las Vegas?"
    )
    assert attrs.pop(f"{input_msg_1}.{MESSAGE_CONTENT_TYPE}") == "text"
    assert attrs.pop(f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_ROLE}") == "user"
    assert attrs.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
    tool_call_base = f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_TOOL_CALLS}.0"
    assert attrs.pop(f"{tool_call_base}.{TOOL_CALL_FUNCTION_NAME}") == "OpenMeteoTool"
    assert attrs.pop(f"{tool_call_base}.{TOOL_CALL_FUNCTION_ARGUMENTS_JSON}") == (
        '{"location_name":"Las Vegas","country":"USA","start_date":null,'
        '"end_date":null,"temperature_unit":"celsius"}'
    )
    assert attrs.pop(f"{METADATA}.chunks_count") == 1
    assert attrs.pop(f"{METADATA}.class_name") == "OpenAIChatModel"
    assert attrs.pop(f"{METADATA}.usage.completion_tokens") == "39"
    assert attrs.pop(f"{METADATA}.usage.cached_creation_tokens") == "0"
    assert attrs.pop(f"{METADATA}.usage.cached_prompt_tokens") == "0"
    assert attrs.pop(f"{METADATA}.usage.prompt_tokens") == "252"
    assert attrs.pop(f"{METADATA}.usage.total_tokens") == "291"
    assert attrs.pop(OUTPUT_VALUE) == ""
    tool_call = f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_TOOL_CALLS}.0"
    assert (
        attrs.pop(f"{tool_call}.{ToolCallAttributes.TOOL_CALL_ID}")
        == "call_l3hCeqcnWalcG3C9gI5t9p1w"
    )
    assert attrs.pop(f"{LLM_TOOLS}.0.{TOOL_JSON_SCHEMA}").startswith(
        '{"type": "function", "function": {"name": "DuckDuckGo"'
    )
    assert attrs.pop(f"{LLM_TOOLS}.1.{TOOL_JSON_SCHEMA}").startswith(
        '{"type": "function", "function": {"name": "OpenMeteoTool"'
    )
    assert attrs == {}, f"Unexpected keys found: {attrs}"


@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_agent(
    in_memory_span_exporter: InMemorySpanExporter,
    openai_api_key: str,
) -> None:
    llm = OpenAIChatModel(model_id="gpt-4o-mini")
    await llm.run(
        [
            SystemMessage("Answer the user's question using the available tools."),
            UserMessage("What's the current weather in Las Vegas?"),
        ],
        tools=[DuckDuckGoSearchTool(), OpenMeteoTool()],
        stream=False,
    )
    spans = list(in_memory_span_exporter.get_finished_spans())
    spans.sort(key=lambda span: span.start_time or 0)
    assert len(spans) == 1
    attrs: dict[str, Any] = dict(spans[0].attributes or {})
    assert attrs.pop(INPUT_MIME_TYPE) == JSON
    assert attrs.pop(OUTPUT_MIME_TYPE) == TEXT
    assert attrs.pop(INPUT_VALUE).startswith(
        '{"input": [{"role": "system", "content": "Answer the user\'s question'
    )
    assert attrs.pop(OUTPUT_VALUE) == ""
    assert attrs.pop(LLM_MODEL_NAME) == "gpt-4o-mini"
    assert attrs.pop(SpanAttributes.LLM_PROVIDER) == "openai"
    assert attrs.pop(OPENINFERENCE_SPAN_KIND) == LLM
    assert attrs.pop(LLM_TOKEN_COUNT_PROMPT) == 252
    assert attrs.pop(LLM_TOKEN_COUNT_COMPLETION) == 39
    assert attrs.pop(LLM_TOKEN_COUNT_TOTAL) == 291
    assert attrs.pop(SpanAttributes.LLM_COST_PROMPT) == 3.78e-05
    assert attrs.pop(SpanAttributes.LLM_COST_COMPLETION) == 2.34e-05
    assert attrs.pop(SpanAttributes.LLM_COST_TOTAL) == 6.12e-05
    assert attrs.pop(LLM_INVOCATION_PARAMETERS).startswith('{"temperature": 0.0')
    sys_msg = f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0"
    assert attrs.pop(f"{sys_msg}.{MESSAGE_CONTENT_TYPE}") == "text"
    assert attrs.pop(f"{sys_msg}.{MESSAGE_CONTENT_TEXT}") == (
        "Answer the user's question using the available tools."
    )
    assert attrs.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "system"

    user_msg = f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_CONTENTS}.0"
    assert attrs.pop(f"{user_msg}.{MESSAGE_CONTENT_TYPE}") == "text"
    assert attrs.pop(f"{user_msg}.{MESSAGE_CONTENT_TEXT}") == (
        "What's the current weather in Las Vegas?"
    )
    assert attrs.pop(f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_ROLE}") == "user"
    assert attrs.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
    tool_call = f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_TOOL_CALLS}.0"
    assert attrs.pop(f"{tool_call}.{ToolCallAttributes.TOOL_CALL_ID}").startswith("call_")
    assert attrs.pop(f"{tool_call}.{TOOL_CALL_FUNCTION_NAME}") == "OpenMeteoTool"
    assert attrs.pop(f"{tool_call}.{TOOL_CALL_FUNCTION_ARGUMENTS_JSON}") == (
        '{"location_name":"Las Vegas","country":"USA","start_date":null,'
        '"end_date":null,"temperature_unit":"celsius"}'
    )
    assert attrs.pop(f"{LLM_TOOLS}.0.{TOOL_JSON_SCHEMA}").startswith(
        '{"type": "function", "function": {"name": "DuckDuckGo"'
    )

    assert attrs.pop(f"{LLM_TOOLS}.1.{TOOL_JSON_SCHEMA}").startswith(
        '{"type": "function", "function": {"name": "OpenMeteoTool"'
    )
    assert attrs.pop(f"{METADATA}.class_name") == "OpenAIChatModel"
    assert attrs.pop(f"{METADATA}.chunks_count") == 1
    assert attrs.pop(f"{METADATA}.usage.prompt_tokens") == "252"
    assert attrs.pop(f"{METADATA}.usage.completion_tokens") == "39"
    assert attrs.pop(f"{METADATA}.usage.total_tokens") == "291"
    assert attrs.pop(f"{METADATA}.usage.cached_creation_tokens") == "0"
    assert attrs.pop(f"{METADATA}.usage.cached_prompt_tokens") == "0"
    assert attrs == {}, f"Unexpected keys found: {attrs}"


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
MESSAGE_CONTENTS = MessageAttributes.MESSAGE_CONTENTS
MESSAGE_CONTENT_TEXT = MessageContentAttributes.MESSAGE_CONTENT_TEXT
MESSAGE_CONTENT_TYPE = MessageContentAttributes.MESSAGE_CONTENT_TYPE
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
USER_ID = SpanAttributes.USER_ID
LLM_TOOLS = SpanAttributes.LLM_TOOLS
TOOL_JSON_SCHEMA = ToolAttributes.TOOL_JSON_SCHEMA

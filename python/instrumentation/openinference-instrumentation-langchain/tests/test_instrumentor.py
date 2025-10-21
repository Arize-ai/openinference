import asyncio
import json
import logging
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from importlib.metadata import version
from secrets import token_hex
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)
from unittest.mock import patch

import openai
import pytest
import vcr  # type: ignore
from google.auth.credentials import AnonymousCredentials
from httpx import AsyncByteStream, Response, SyncByteStream
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableLambda, RunnableSerializable
from langchain_google_vertexai import VertexAI
from langchain_openai import ChatOpenAI
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Span
from opentelemetry.util._importlib_metadata import entry_points
from respx import MockRouter
from typing_extensions import assert_never

from openinference.instrumentation.langchain import (
    LangChainInstrumentor,
    get_ancestor_spans,
    get_current_span,
)
from openinference.semconv.trace import (
    DocumentAttributes,
    EmbeddingAttributes,
    MessageAttributes,
    MessageContentAttributes,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolAttributes,
    ToolCallAttributes,
)

for name, logger in logging.root.manager.loggerDict.items():
    if name.startswith("openinference.") and isinstance(logger, logging.Logger):
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()
        logger.addHandler(logging.StreamHandler())

LANGCHAIN_VERSION = tuple(map(int, version("langchain-core").split(".")[:3]))
LANGCHAIN_OPENAI_VERSION = tuple(map(int, version("langchain-openai").split(".")[:3]))
SUPPORTS_TEMPLATES = LANGCHAIN_VERSION < (0, 3, 0)


class TestInstrumentor:
    def test_entrypoint_for_opentelemetry_instrument(self) -> None:
        (instrumentor_entrypoint,) = entry_points(  # type: ignore[no-untyped-call]
            group="opentelemetry_instrumentor", name="langchain"
        )
        instrumentor = instrumentor_entrypoint.load()()
        assert isinstance(instrumentor, LangChainInstrumentor)


@pytest.mark.parametrize("is_async", [False, True])
async def test_get_current_span(
    in_memory_span_exporter: InMemorySpanExporter,
    is_async: bool,
) -> None:
    if is_async and sys.version_info < (3, 11):
        pytest.xfail("async test fails in older Python")
    n = 10
    loop = asyncio.get_running_loop()
    if is_async:

        async def f(_: Any) -> Optional[Span]:
            await asyncio.sleep(0.001)
            return get_current_span()

        results = await asyncio.gather(*(RunnableLambda(f).ainvoke(...) for _ in range(n)))  # type: ignore[arg-type]
    else:
        results = await asyncio.gather(
            *(
                loop.run_in_executor(None, RunnableLambda(lambda _: get_current_span()).invoke, ...)
                for _ in range(n)
            )
        )
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == n
    assert {id(span.get_span_context()) for span in results if isinstance(span, Span)} == {
        id(span.get_span_context())  # type: ignore[no-untyped-call]
        for span in spans
    }


def test_get_current_span_when_there_is_no_tracer() -> None:
    instrumentor = LangChainInstrumentor()
    instrumentor.uninstrument()
    del instrumentor._tracer
    assert RunnableLambda(lambda _: (get_current_span(), get_ancestor_spans())).invoke(0) == (
        None,
        [],
    )


async def test_get_ancestor_spans(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """Test retrieving the current chain root span during RunnableLambda execution."""
    n = 10  # Number of concurrent runs
    loop = asyncio.get_running_loop()

    ancestors_during_execution = []

    def f(x: int) -> int:
        current_span = get_current_span()
        root_spans = get_ancestor_spans()
        assert root_spans is not None, "Ancestor should not be None during execution (async)"
        assert len(root_spans) == 1, "Only get ancestor spans"
        assert current_span is not root_spans[0], "Ancestor is distinct from the current span"
        ancestors_during_execution.append(root_spans[0])
        assert (
            root_spans[0].name == "RunnableSequence"  # type: ignore[attr-defined, unused-ignore]
        ), "RunnableSequence should be the outermost ancestor"
        return x + 1

    sequence: RunnableSerializable[int, int] = RunnableLambda[int, int](f) | RunnableLambda[
        int, int
    ](f)

    with ThreadPoolExecutor() as executor:
        tasks = [loop.run_in_executor(executor, sequence.invoke, 1) for _ in range(n)]
        await asyncio.gather(*tasks)

    ancestors_after_execution = get_ancestor_spans()
    assert ancestors_after_execution == [], "No ancestors after execution"

    assert len(ancestors_during_execution) == 2 * n, (
        "Did not capture all ancestors during execution"
    )

    assert len(set(id(span) for span in ancestors_during_execution)) == n, (
        "Both Lambdas share the same ancestor"
    )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 3 * n, f"Expected {3 * n} spans, but found {len(spans)}"


async def test_get_ancestor_spans_async(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """Test retrieving the current chain root span during RunnableLambda execution."""
    if sys.version_info < (3, 11):
        pytest.xfail("Async test may fail on Python versions below 3.11")
    n = 10  # Number of concurrent runs

    ancestors_during_execution = []

    async def f(x: int) -> int:
        current_span = get_current_span()
        root_spans = get_ancestor_spans()
        assert root_spans is not None, "Ancestor should not be None during execution (async)"
        assert len(root_spans) == 1, "Only get ancestor spans"
        assert current_span is not root_spans[0], "Ancestor is distinct from the current span"
        ancestors_during_execution.append(root_spans[0])
        assert (
            root_spans[0].name == "RunnableSequence"  # type: ignore[attr-defined, unused-ignore]
        ), "RunnableSequence should be the outermost ancestor"
        await asyncio.sleep(0.01)
        return x + 1

    sequence: RunnableSerializable[int, int] = RunnableLambda[int, int](f) | RunnableLambda[
        int, int
    ](f)

    await asyncio.gather(*(sequence.ainvoke(1) for _ in range(n)))

    ancestors_after_execution = get_ancestor_spans()
    assert ancestors_after_execution == [], "No ancestors after execution"

    assert len(ancestors_during_execution) == 2 * n, (
        "Did not capture all ancestors during execution"
    )

    assert len(set(id(span) for span in ancestors_during_execution)) == n, (
        "Both Lambdas share the same ancestor"
    )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 3 * n, f"Expected {3 * n} spans, but found {len(spans)}"


@pytest.mark.parametrize(
    "message",
    [
        SystemMessage(name=token_hex(8), content=token_hex(8)),
        SystemMessage(content=[token_hex(8), token_hex(8)]),
        HumanMessage(name=token_hex(8), content=token_hex(8)),
        HumanMessage(content=[token_hex(8), token_hex(8)]),
        AIMessage(name=token_hex(8), content=token_hex(8)),
        AIMessage(content=[token_hex(8), token_hex(8)]),
        ToolMessage(name=token_hex(8), tool_call_id=token_hex(8), content=token_hex(8)),
        ToolMessage(tool_call_id=token_hex(8), content=[token_hex(8), token_hex(8)]),
    ],
)
def test_input_messages(
    message: Union[SystemMessage, HumanMessage, AIMessage, ToolMessage],
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    n = 2
    url = "https://api.openai.com/v1/chat/completions"
    respx_mock.post(url).mock(Response(400))
    with suppress(openai.BadRequestError):
        ChatOpenAI().invoke([message] * n)
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attributes = dict(span.attributes or {})
    for i in range(n):
        prefix = f"{LLM_INPUT_MESSAGES}.{i}"
        key = f"{prefix}.{MESSAGE_ROLE}"
        if isinstance(message, SystemMessage):
            assert attributes[key] == "system"
        elif isinstance(message, HumanMessage):
            assert attributes[key] == "user"
        elif isinstance(message, AIMessage):
            assert attributes[key] == "assistant"
        elif isinstance(message, ToolMessage):
            assert attributes[key] == "tool"
        else:
            assert_never(message)
        if message.name:
            key = f"{prefix}.{MESSAGE_NAME}"
            assert attributes[key] == message.name
        if isinstance(message.content, str):
            key = f"{prefix}.{MESSAGE_CONTENT}"
            assert attributes[key] == message.content
        elif isinstance(message.content, list):
            for j in range(len(message.content)):
                key = f"{prefix}.{MESSAGE_CONTENTS}.{j}.{MESSAGE_CONTENT_TEXT}"
                assert attributes[key] == message.content[j]
        else:
            assert_never(message.content)
        if isinstance(message, ToolMessage):
            key = f"{prefix}.{MESSAGE_TOOL_CALL_ID}"
            assert attributes[key] == message.tool_call_id


def test_anthropic_token_counts(
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
    anthropic_api_key: str,
) -> None:
    langchain_anthropic = pytest.importorskip(
        "langchain_anthropic", reason="`langchain-anthropic` is not installed"
    )  # langchain-anthropic is not in pyproject.toml because it conflicts with pinned test deps

    respx_mock.post("https://api.anthropic.com/v1/messages").mock(
        return_value=Response(
            status_code=200,
            json={
                "id": "msg_015kYHnmPtpzZbXpwMmziqju",
                "type": "message",
                "role": "assistant",
                "model": "claude-3-5-sonnet-20240620",
                "content": [{"type": "text", "text": "Argentina."}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 22,
                    "output_tokens": 5,
                    "cache_read_input_tokens": 9,
                    "cache_creation_input_tokens": 2,
                },
            },
        )
    )
    model = langchain_anthropic.ChatAnthropic(model="claude-3-5-sonnet-20240620")
    model.invoke("Who won the World Cup in 2022? Answer in one word.")
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    llm_attributes = dict(span.attributes or {})
    assert llm_attributes.pop(OPENINFERENCE_SPAN_KIND, None) == LLM.value
    assert llm_attributes.pop(LLM_TOKEN_COUNT_PROMPT, None) == 22
    assert llm_attributes.pop(LLM_TOKEN_COUNT_COMPLETION, None) == 5
    assert llm_attributes.pop(LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE) == 2
    assert llm_attributes.pop(LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ) == 9


@pytest.mark.parametrize(
    "streaming, cassette_name",
    [
        (True, "test_gemini_token_counts_streaming"),
        (False, "test_gemini_token_counts"),
    ],
)
@pytest.mark.vcr(
    match_on=["method", "uri"],
)
def test_gemini_token_counts_streaming(
    streaming: bool,
    cassette_name: str,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    with patch("google.auth.default", return_value=(AnonymousCredentials(), "test-project")) as _:  # type: ignore
        # patching with AnonymousCredentials in order to avoid a check on CI where a metadata
        # request will be made because the job is running in GCE.
        # ignoring types until https://github.com/googleapis/google-cloud-python/issues/10540
        with vcr.use_cassette(path=f"tests/cassettes/test_instrumentor/{cassette_name}.yaml"):
            llm = VertexAI(
                api_transport="rest",
                project="test-project",
                model_name="gemini-pro",
                streaming=streaming,
            )
            llm.invoke("Tell me a funny joke, a one-liner.")
            spans = in_memory_span_exporter.get_finished_spans()
            assert len(spans) == 1
            span = spans[0]
            attributes = dict(span.attributes or {})
            assert attributes.pop(OPENINFERENCE_SPAN_KIND, None) == LLM.value
            assert isinstance(attributes.pop(LLM_TOKEN_COUNT_PROMPT, None), int)
            assert isinstance(attributes.pop(LLM_TOKEN_COUNT_COMPLETION, None), int)
            assert isinstance(attributes.pop(LLM_TOKEN_COUNT_TOTAL, None), int)


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


@pytest.mark.skipif(
    condition=LANGCHAIN_OPENAI_VERSION < (0, 1, 9),
    reason="The stream_usage parameter was introduced in langchain-openai==0.1.9",
    # https://github.com/langchain-ai/langchain/releases/tag/langchain-openai%3D%3D0.1.9
)
@pytest.mark.vcr(
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
def test_records_token_counts_for_streaming_openai_llm(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    llm = ChatOpenAI(streaming=True, stream_usage=True)  # type: ignore[call-arg,unused-ignore]
    llm.invoke("Tell me a funny joke, a one-liner.")
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attributes = dict(span.attributes or {})
    assert attributes.pop(OPENINFERENCE_SPAN_KIND, None) == LLM.value
    assert isinstance(attributes.pop(LLM_TOKEN_COUNT_PROMPT, None), int)
    assert isinstance(attributes.pop(LLM_TOKEN_COUNT_COMPLETION, None), int)
    assert isinstance(attributes.pop(LLM_TOKEN_COUNT_TOTAL, None), int)


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
def test_token_counts(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    from langchain.chat_models import init_chat_model
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = init_chat_model("gpt-4o-mini", model_provider="openai")
    messages = [
        SystemMessage("Translate the following from English into Italian"),
        HumanMessage("hi!"),
    ]
    llm.invoke(messages)

    # The token counts in the mocked responses in the file
    # "cassettes/test_instrumentor/test_token_counts.yaml"
    # are not accurate representations of the actual token counts fro API calls.
    # They were manually altered/hard coded for test assertions.

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attr = dict(span.attributes or {})
    assert attr.pop(LLM_TOKEN_COUNT_PROMPT) == 20
    assert attr.pop(LLM_TOKEN_COUNT_COMPLETION) == 4
    assert attr.pop(LLM_TOKEN_COUNT_COMPLETION_DETAILS_AUDIO) == 4
    assert attr.pop(LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING) == 3
    assert attr.pop(LLM_TOKEN_COUNT_PROMPT_DETAILS_AUDIO) == 2
    assert attr.pop(LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ) == 1
    assert attr.pop(LLM_TOKEN_COUNT_TOTAL) == 24


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
    cassette_library_dir="tests/cassettes/test_instrumentor",  # Explicitly set the directory
)
def test_tool_call_with_function(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    from langchain.chat_models import init_chat_model
    from langchain_core.messages import AIMessage
    from langchain_core.tools import tool

    @tool
    def add(a: int, b: int) -> int:
        """Adds a and b."""
        return a + b

    @tool
    def multiply(a: int, b: int) -> int:
        """Multiplies a and b."""
        return a * b

    tools = [add, multiply]

    llm = init_chat_model(
        "gpt-4o-mini",
        model_provider="openai",
        api_key="sk-fake-key",
    )
    llm_with_tools = llm.bind_tools(tools)
    query = "What is 3 * 12? Also, what is 11 + 49?"
    result = llm_with_tools.invoke(query)
    assert isinstance(result, AIMessage)
    _ = result.tool_calls
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attributes = dict(span.attributes or {})

    # Test input message
    assert attributes.pop("llm.input_messages.0.message.role") == "user"
    assert attributes.pop("llm.input_messages.0.message.content") == query

    # Test output message and tool calls
    assert attributes.pop("llm.output_messages.0.message.role") == "assistant"

    # Test first tool call (multiply)
    assert (
        attributes.pop("llm.output_messages.0.message.tool_calls.0.tool_call.function.name")
        == "multiply"
    )
    multiply_args = attributes.pop(
        "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments"
    )
    assert isinstance(multiply_args, str)
    assert json.loads(multiply_args) == {"a": 3, "b": 12}

    # Test second tool call (add)
    assert (
        attributes.pop("llm.output_messages.0.message.tool_calls.1.tool_call.function.name")
        == "add"
    )
    add_args = attributes.pop(
        "llm.output_messages.0.message.tool_calls.1.tool_call.function.arguments"
    )
    assert isinstance(add_args, str)
    assert json.loads(add_args) == {"a": 11, "b": 49}

    # Test tool schemas
    tool1_schema = attributes.pop(f"{LLM_TOOLS}.0.{TOOL_JSON_SCHEMA}", None)
    tool2_schema = attributes.pop(f"{LLM_TOOLS}.1.{TOOL_JSON_SCHEMA}", None)
    assert tool1_schema is not None
    assert tool2_schema is not None
    assert isinstance(tool1_schema, str)
    assert isinstance(tool2_schema, str)

    tool1_schema_dict = json.loads(tool1_schema)
    assert tool1_schema_dict["type"] == "function"
    assert tool1_schema_dict["function"]["name"] == "add"
    assert tool1_schema_dict["function"]["description"] == "Adds a and b."
    assert tool1_schema_dict["function"]["parameters"]["properties"]["a"]["type"] == "integer"
    assert tool1_schema_dict["function"]["parameters"]["properties"]["b"]["type"] == "integer"

    tool2_schema_dict = json.loads(tool2_schema)
    assert tool2_schema_dict["type"] == "function"
    assert tool2_schema_dict["function"]["name"] == "multiply"
    assert tool2_schema_dict["function"]["description"] == "Multiplies a and b."
    assert tool2_schema_dict["function"]["parameters"]["properties"]["a"]["type"] == "integer"
    assert tool2_schema_dict["function"]["parameters"]["properties"]["b"]["type"] == "integer"


def _check_context_attributes(
    attributes: Dict[str, Any],
    session_id: Optional[str],
    user_id: Optional[str],
    metadata: Optional[Dict[str, Any]],
    tags: Optional[List[str]],
    prompt_template: Optional[str],
    prompt_template_version: Optional[str],
    prompt_template_variables: Optional[Dict[str, Any]],
) -> None:
    if session_id is not None:
        assert attributes.pop(SESSION_ID, None) == session_id
    if user_id is not None:
        assert attributes.pop(USER_ID, None) == user_id
    if metadata is not None:
        attr_metadata = attributes.pop(METADATA, None)
        assert attr_metadata is not None
        assert isinstance(attr_metadata, str)  # must be json string
        metadata_dict = json.loads(attr_metadata)
        assert metadata_dict == metadata
    if tags is not None:
        attr_tags = attributes.pop(TAG_TAGS, None)
        assert attr_tags is not None
        assert len(attr_tags) == len(tags)
        assert list(attr_tags) == tags
    if prompt_template is not None:
        assert attributes.pop(SpanAttributes.LLM_PROMPT_TEMPLATE, None) == prompt_template
    if prompt_template_version:
        assert (
            attributes.pop(SpanAttributes.LLM_PROMPT_TEMPLATE_VERSION, None)
            == prompt_template_version
        )
    if prompt_template_variables:
        x = attributes.pop(SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES, None)
        assert x


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


@pytest.fixture
def documents() -> List[str]:
    return [randstr(), randstr()]


@pytest.fixture
def chat_completion_mock_stream() -> Tuple[List[bytes], List[Dict[str, Any]]]:
    return (
        [
            b'data: {"choices": [{"delta": {"role": "assistant"}, "index": 0}]}\n\n',
            b'data: {"choices": [{"delta": {"content": "A"}, "index": 0}]}\n\n',
            b'data: {"choices": [{"delta": {"content": "B"}, "index": 0}]}\n\n',
            b'data: {"choices": [{"delta": {"content": "C"}, "index": 0}]}\n\n',
            b"data: [DONE]\n",
        ],
        [{"role": "assistant", "content": "ABC"}],
    )


@pytest.fixture
def completion_usage() -> Dict[str, Any]:
    prompt_tokens = random.randint(1, 1000)
    completion_tokens = random.randint(1, 1000)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


@pytest.fixture
def model_name() -> str:
    return randstr()


@pytest.fixture
def anthropic_api_key(monkeypatch: pytest.MonkeyPatch) -> str:
    api_key = "sk-1234567890"
    monkeypatch.setenv("ANTHROPIC_API_KEY", api_key)
    return api_key


def randstr() -> str:
    return str(random.random())


class MockByteStream(SyncByteStream, AsyncByteStream):
    def __init__(self, byte_stream: Iterable[bytes]):
        self._byte_stream = byte_stream

    def __iter__(self) -> Iterator[bytes]:
        for byte_string in self._byte_stream:
            yield byte_string

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for byte_string in self._byte_stream:
            yield byte_string


LANGCHAIN_SESSION_ID = "session_id"
LANGCHAIN_CONVERSATION_ID = "conversation_id"
LANGCHAIN_THREAD_ID = "thread_id"

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
LLM_TOKEN_COUNT_COMPLETION = SpanAttributes.LLM_TOKEN_COUNT_COMPLETION
LLM_TOKEN_COUNT_COMPLETION_DETAILS_AUDIO = SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_AUDIO
LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING = (
    SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING
)
LLM_TOKEN_COUNT_PROMPT = SpanAttributes.LLM_TOKEN_COUNT_PROMPT
LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE = (
    SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE
)
LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ = SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ
LLM_TOKEN_COUNT_PROMPT_DETAILS_AUDIO = SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_AUDIO
LLM_TOKEN_COUNT_TOTAL = SpanAttributes.LLM_TOKEN_COUNT_TOTAL
MESSAGE_CONTENT = MessageAttributes.MESSAGE_CONTENT
MESSAGE_CONTENTS = MessageAttributes.MESSAGE_CONTENTS
MESSAGE_CONTENT_TEXT = MessageContentAttributes.MESSAGE_CONTENT_TEXT
MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON = MessageAttributes.MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON
MESSAGE_FUNCTION_CALL_NAME = MessageAttributes.MESSAGE_FUNCTION_CALL_NAME
MESSAGE_NAME = MessageAttributes.MESSAGE_NAME
MESSAGE_ROLE = MessageAttributes.MESSAGE_ROLE
MESSAGE_TOOL_CALLS = MessageAttributes.MESSAGE_TOOL_CALLS
MESSAGE_TOOL_CALL_ID = MessageAttributes.MESSAGE_TOOL_CALL_ID
METADATA = SpanAttributes.METADATA
OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
OUTPUT_MIME_TYPE = SpanAttributes.OUTPUT_MIME_TYPE
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
RETRIEVAL_DOCUMENTS = SpanAttributes.RETRIEVAL_DOCUMENTS
TOOL_CALL_FUNCTION_ARGUMENTS_JSON = ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON
TOOL_CALL_FUNCTION_NAME = ToolCallAttributes.TOOL_CALL_FUNCTION_NAME
TOOL_JSON_SCHEMA = ToolAttributes.TOOL_JSON_SCHEMA
LLM_TOOLS = SpanAttributes.LLM_TOOLS

CHAIN = OpenInferenceSpanKindValues.CHAIN
LLM = OpenInferenceSpanKindValues.LLM
RETRIEVER = OpenInferenceSpanKindValues.RETRIEVER

JSON = OpenInferenceMimeTypeValues.JSON
SESSION_ID = SpanAttributes.SESSION_ID
USER_ID = SpanAttributes.USER_ID
TAG_TAGS = SpanAttributes.TAG_TAGS

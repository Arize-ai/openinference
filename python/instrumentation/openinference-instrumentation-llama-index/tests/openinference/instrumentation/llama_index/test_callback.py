import asyncio
import json
import logging
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from importlib.metadata import version
from itertools import count
from typing import (
    Any,
    AsyncIterator,
    DefaultDict,
    Dict,
    Generator,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    cast,
)

import openai
import pytest
from httpx import AsyncByteStream, Response, SyncByteStream
from llama_index.core import Document, ListIndex, Settings
from llama_index.core.base.response.schema import StreamingResponse
from llama_index.core.callbacks import CallbackManager
from llama_index.core.schema import TextNode
from llama_index.llms.openai import OpenAI  # type: ignore
from openinference.instrumentation import using_attributes
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from openinference.semconv.trace import (
    DocumentAttributes,
    EmbeddingAttributes,
    MessageAttributes,
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolCallAttributes,
)
from opentelemetry import trace as trace_api
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from respx import MockRouter
from tenacity import wait_none

for k in dir(OpenAI):
    v = getattr(OpenAI, k)
    if callable(v) and hasattr(v, "retry") and hasattr(v.retry, "wait"):
        v.retry.wait = wait_none()


for name, logger in logging.root.manager.loggerDict.items():
    if name.startswith("openinference.") and isinstance(logger, logging.Logger):
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()
        logger.addHandler(logging.StreamHandler())

LLAMA_INDEX_VERSION = tuple(map(int, version("llama-index-core").split(".")[:3]))


@pytest.mark.parametrize("is_stream", [False, True])
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("status_code", [200, 400])
@pytest.mark.parametrize("use_context_attributes", [False, True])
def test_callback_llm(
    is_async: bool,
    is_stream: bool,
    status_code: int,
    use_context_attributes: bool,
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
    nodes: List[Document],
    chat_completion_mock_stream: Tuple[List[bytes], List[Dict[str, Any]]],
    session_id: str,
    user_id: str,
    metadata: Dict[str, Any],
    tags: List[str],
) -> None:
    if is_stream and is_async:
        pytest.xfail("not supported")
    if status_code == 400 and is_stream and LLAMA_INDEX_VERSION < (0, 10, 44):
        pytest.xfail("streaming errors can't be detected")
    n = 10  # number of concurrent queries
    questions = {randstr() for _ in range(n)}
    answer = chat_completion_mock_stream[1][0]["content"] if is_stream else randstr()
    callback_manager = CallbackManager()
    Settings.callback_manager = callback_manager
    Settings.llm = OpenAI()
    query_engine = ListIndex(nodes).as_query_engine(use_async=is_async, streaming=is_stream)
    respx_kwargs: Dict[str, Any] = (
        {
            "stream": MockAsyncByteStream(chat_completion_mock_stream[0])
            if is_async
            else MockSyncByteStream(chat_completion_mock_stream[0])
        }
        if is_stream
        else {
            "json": {
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": answer},
                        "finish_reason": "stop",
                    }
                ],
            }
        }
    )
    url = "https://api.openai.com/v1/chat/completions"
    respx_mock.post(url).mock(return_value=Response(status_code=status_code, **respx_kwargs))

    async def task() -> None:
        await asyncio.gather(
            *(query_engine.aquery(question) for question in questions),
            return_exceptions=True,
        )

    def threaded_query(question: str) -> None:
        response = query_engine.query(question)
        (list(cast(StreamingResponse, response).response_gen) if is_stream else None,)

    def threaded_query_with_attributes(question: str) -> None:
        # This context manager must be inside this function definition so
        # there's a different instantiation per thread. This allows to use
        # a different context per thread, as desired
        with using_attributes(
            session_id=session_id,
            user_id=user_id,
            metadata=metadata,
            tags=tags,
        ):
            response = query_engine.query(question)
            (list(cast(StreamingResponse, response).response_gen) if is_stream else None,)

    with suppress(openai.BadRequestError):
        if use_context_attributes:
            if is_async:
                with using_attributes(
                    session_id=session_id,
                    user_id=user_id,
                    metadata=metadata,
                    tags=tags,
                ):
                    asyncio.run(task())
            else:
                with ThreadPoolExecutor(max_workers=n) as executor:
                    executor.map(
                        threaded_query_with_attributes,
                        questions,
                    )
        else:
            if is_async:
                asyncio.run(task())
            else:
                with ThreadPoolExecutor(max_workers=n) as executor:
                    executor.map(
                        threaded_query,
                        questions,
                    )

    spans = in_memory_span_exporter.get_finished_spans()
    traces: DefaultDict[int, Dict[str, ReadableSpan]] = defaultdict(dict)
    for span in spans:
        traces[span.context.trace_id][span.name] = span

    assert len(traces) == n
    for spans_by_name in traces.values():
        assert (query_span := spans_by_name.pop("query", None)) is not None
        assert query_span.parent is None
        query_attributes = dict(query_span.attributes or {})
        assert query_attributes.pop(OPENINFERENCE_SPAN_KIND, None) == CHAIN.value
        question = cast(Optional[str], query_attributes.pop(INPUT_VALUE, None))
        assert question is not None
        assert question in questions
        questions.remove(question)
        if status_code == 200:
            assert query_span.status.status_code == trace_api.StatusCode.OK
            assert not query_span.status.description
            if not (is_async and is_stream):
                assert query_attributes.pop(OUTPUT_VALUE, None) == answer

        if use_context_attributes:
            _check_context_attributes(query_attributes, session_id, user_id, metadata, tags)
        assert query_attributes == {}  # all attributes should be accounted for

        assert (synthesize_span := spans_by_name.pop("synthesize", None)) is not None
        assert synthesize_span.parent is not None
        assert synthesize_span.parent.span_id == query_span.context.span_id
        assert synthesize_span.context.trace_id == query_span.context.trace_id
        synthesize_attributes = dict(synthesize_span.attributes or {})
        assert synthesize_attributes.pop(OPENINFERENCE_SPAN_KIND, None) == CHAIN.value
        assert synthesize_attributes.pop(INPUT_VALUE, None) == question
        if status_code == 200:
            assert synthesize_span.status.status_code == trace_api.StatusCode.OK
            assert not synthesize_span.status.description
            if not (is_async and is_stream):
                assert synthesize_attributes.pop(OUTPUT_VALUE, None) == answer
        else:
            assert synthesize_span.status.status_code == trace_api.StatusCode.ERROR
            assert (
                synthesize_span.status.description
                and synthesize_span.status.description.startswith(
                    openai.BadRequestError.__name__,
                )
            )
        if use_context_attributes:
            _check_context_attributes(synthesize_attributes, session_id, user_id, metadata, tags)
        assert synthesize_attributes == {}  # all attributes should be accounted for

        assert (retrieve_span := spans_by_name.pop("retrieve", None)) is not None
        assert retrieve_span.parent is not None
        assert retrieve_span.parent.span_id == query_span.context.span_id
        assert retrieve_span.context.trace_id == query_span.context.trace_id
        retrieve_attributes = dict(retrieve_span.attributes or {})
        assert retrieve_attributes.pop(OPENINFERENCE_SPAN_KIND, None) == RETRIEVER.value
        assert retrieve_attributes.pop(INPUT_VALUE, None) == question
        retrieve_attributes.pop(f"{RETRIEVAL_DOCUMENTS}.0.{DOCUMENT_ID}", None)
        retrieve_attributes.pop(f"{RETRIEVAL_DOCUMENTS}.1.{DOCUMENT_ID}", None)
        assert (
            retrieve_attributes.pop(f"{RETRIEVAL_DOCUMENTS}.0.{DOCUMENT_CONTENT}", None)
            == nodes[0].text
        )
        assert retrieve_attributes.pop(
            f"{RETRIEVAL_DOCUMENTS}.0.{DOCUMENT_METADATA}", None
        ) == json.dumps(nodes[0].metadata)
        assert (
            retrieve_attributes.pop(f"{RETRIEVAL_DOCUMENTS}.1.{DOCUMENT_CONTENT}", None)
            == nodes[1].text
        )
        if use_context_attributes:
            _check_context_attributes(retrieve_attributes, session_id, user_id, metadata, tags)
        assert retrieve_attributes == {}  # all attributes should be accounted for

        assert (llm_span := spans_by_name.pop("llm", None)) is not None
        assert llm_span.parent is not None
        assert llm_span.parent.span_id == synthesize_span.context.span_id
        assert llm_span.context.trace_id == synthesize_span.context.trace_id
        llm_attributes = dict(llm_span.attributes or {})
        assert llm_attributes.pop(OPENINFERENCE_SPAN_KIND, None) == LLM.value
        assert llm_attributes.pop(LLM_MODEL_NAME, None) is not None
        assert llm_attributes.pop(LLM_INVOCATION_PARAMETERS, None) is not None
        assert llm_attributes.pop(LLM_PROMPT_TEMPLATE, None) is not None
        template_variables = json.loads(
            cast(str, llm_attributes.pop(f"{LLM_PROMPT_TEMPLATE_VARIABLES}", None))
        )
        assert template_variables.keys() == {"context_str", "query_str"}
        assert template_variables["query_str"] == question
        assert llm_attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}", None) is not None
        assert llm_attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}", None) is not None
        assert llm_attributes.pop(f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_ROLE}", None) is not None
        assert llm_attributes.pop(f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_CONTENT}", None) is not None
        if status_code == 200:
            assert llm_span.status.status_code == trace_api.StatusCode.OK
            assert not llm_span.status.description
            assert llm_attributes.pop(OUTPUT_VALUE, None) == answer
            if not is_stream:
                # FIXME: currently we can't capture messages when streaming
                assert (
                    llm_attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}", None)
                    == "assistant"
                )
                assert (
                    llm_attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}", None) == answer
                )
        else:
            assert llm_span.status.status_code == trace_api.StatusCode.ERROR
            assert llm_span.status.description and llm_span.status.description.startswith(
                openai.BadRequestError.__name__,
            )
        if use_context_attributes:
            _check_context_attributes(llm_attributes, session_id, user_id, metadata, tags)
        assert llm_attributes == {}  # all attributes should be accounted for

        # FIXME: maybe chunking spans should be discarded?
        assert (chunking_span := spans_by_name.pop("chunking", None)) is not None
        assert chunking_span.parent is not None
        assert chunking_span.parent.span_id == synthesize_span.context.span_id
        assert chunking_span.context.trace_id == synthesize_span.context.trace_id
        chunking_attributes = dict(chunking_span.attributes or {})
        assert chunking_attributes.pop(OPENINFERENCE_SPAN_KIND, None) is not None
        if use_context_attributes:
            _check_context_attributes(chunking_attributes, session_id, user_id, metadata, tags)
        assert chunking_attributes == {}  # all attributes should be accounted for

        assert spans_by_name == {}  # all spans should be accounted for
    assert len(questions) == 0  # all questions should be accounted for


def _check_context_attributes(
    attributes: Dict[str, Any],
    session_id: str,
    user_id: str,
    metadata: Dict[str, Any],
    tags: List[str],
) -> None:
    assert attributes.pop(SESSION_ID, None) == session_id
    assert attributes.pop(USER_ID, None) == user_id
    attr_metadata = attributes.pop(METADATA, None)
    assert attr_metadata is not None
    assert isinstance(attr_metadata, str)  # must be json string
    metadata_dict = json.loads(attr_metadata)
    assert metadata_dict == metadata
    attr_tags = attributes.pop(TAG_TAGS, None)
    assert attr_tags is not None
    assert len(attr_tags) == len(tags)
    assert list(attr_tags) == tags


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
def nodes() -> List[TextNode]:
    return [
        Document(text=randstr(), metadata={"category": randstr()}),  # type: ignore
        TextNode(text=randstr()),
    ]


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


@pytest.fixture(scope="module")
def in_memory_span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture(scope="module")
def tracer_provider(in_memory_span_exporter: InMemorySpanExporter) -> trace_api.TracerProvider:
    tracer_provider = trace_sdk.TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))
    return tracer_provider


@pytest.fixture(autouse=True)
def instrument(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> Generator[None, None, None]:
    LlamaIndexInstrumentor().instrument(
        tracer_provider=tracer_provider,
        use_legacy_callback_handler=True,
    )
    yield
    LlamaIndexInstrumentor().uninstrument()
    in_memory_span_exporter.clear()


@pytest.fixture(autouse=True)
def openai_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-")


@pytest.fixture(scope="module")
def seed() -> Iterator[int]:
    """
    Use rolling seeds to help debugging, because the rolling pseudo-random values
    allow conditional breakpoints to be hit precisely (and repeatably).
    """
    return count()


@pytest.fixture(autouse=True)
def set_seed(seed: Iterator[int]) -> Iterator[None]:
    random.seed(next(seed))
    yield


def randstr() -> str:
    return str(random.random())


class MockSyncByteStream(SyncByteStream):
    def __init__(self, byte_stream: Iterable[bytes]):
        self._byte_stream = byte_stream

    def __iter__(self) -> Iterator[bytes]:
        for byte_string in self._byte_stream:
            yield byte_string


class MockAsyncByteStream(AsyncByteStream):
    def __init__(self, byte_stream: Iterable[bytes]):
        self._byte_stream = byte_stream

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for byte_string in self._byte_stream:
            yield byte_string


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
LLM_TOKEN_COUNT_COMPLETION = SpanAttributes.LLM_TOKEN_COUNT_COMPLETION
LLM_TOKEN_COUNT_PROMPT = SpanAttributes.LLM_TOKEN_COUNT_PROMPT
LLM_TOKEN_COUNT_TOTAL = SpanAttributes.LLM_TOKEN_COUNT_TOTAL
MESSAGE_CONTENT = MessageAttributes.MESSAGE_CONTENT
MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON = MessageAttributes.MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON
MESSAGE_FUNCTION_CALL_NAME = MessageAttributes.MESSAGE_FUNCTION_CALL_NAME
MESSAGE_ROLE = MessageAttributes.MESSAGE_ROLE
MESSAGE_TOOL_CALLS = MessageAttributes.MESSAGE_TOOL_CALLS
OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
OUTPUT_MIME_TYPE = SpanAttributes.OUTPUT_MIME_TYPE
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
RETRIEVAL_DOCUMENTS = SpanAttributes.RETRIEVAL_DOCUMENTS
TOOL_CALL_FUNCTION_ARGUMENTS_JSON = ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON
TOOL_CALL_FUNCTION_NAME = ToolCallAttributes.TOOL_CALL_FUNCTION_NAME
LLM_PROMPT_TEMPLATE = SpanAttributes.LLM_PROMPT_TEMPLATE
LLM_PROMPT_TEMPLATE_VARIABLES = SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES

CHAIN = OpenInferenceSpanKindValues.CHAIN
LLM = OpenInferenceSpanKindValues.LLM
RETRIEVER = OpenInferenceSpanKindValues.RETRIEVER
SESSION_ID = SpanAttributes.SESSION_ID
USER_ID = SpanAttributes.USER_ID
METADATA = SpanAttributes.METADATA
TAG_TAGS = SpanAttributes.TAG_TAGS

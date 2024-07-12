import asyncio
import json
import logging
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from contextvars import copy_context
from functools import partial
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
    Mapping,
    Optional,
    Tuple,
    cast,
)

import openai
import pytest
from httpx import AsyncByteStream, Response, SyncByteStream
from llama_index.core import Document, ListIndex, Settings
from llama_index.core.callbacks import CallbackManager
from llama_index.core.schema import TextNode
from llama_index.llms.openai import OpenAI  # type: ignore
from openinference.instrumentation import using_attributes
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from openinference.instrumentation.openai import OpenAIInstrumentor
from openinference.semconv.trace import (
    DocumentAttributes,
    EmbeddingAttributes,
    MessageAttributes,
    OpenInferenceMimeTypeValues,
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
def test_handler_basic_retrieval(
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
    if status_code == 400 and is_stream and LLAMA_INDEX_VERSION < (0, 10, 44):
        pytest.xfail("streaming errors can't be detected")
    n = 10  # number of concurrent queries
    questions = {randstr() for _ in range(n)}
    answer = chat_completion_mock_stream[1][0]["content"] if is_stream else randstr()
    callback_manager = CallbackManager()
    Settings.callback_manager = callback_manager
    Settings.llm = OpenAI(max_retries=0, timeout=0.01)
    query_engine = ListIndex(nodes).as_query_engine(use_async=is_async, streaming=is_stream)
    respx_kwargs: Dict[str, Any] = (
        {
            "stream": (
                MockAsyncByteStream(chat_completion_mock_stream[0])
                if is_async
                else MockSyncByteStream(chat_completion_mock_stream[0])
            )
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

    async def aquery(question: str) -> None:
        await (await query_engine.aquery(question)).get_response()

    async def task() -> None:
        await asyncio.gather(*(aquery(question) for question in questions), return_exceptions=True)

    def query(question: str) -> None:
        response = query_engine.query(question)
        if is_stream:
            response.get_response()

    def main() -> None:
        if is_async:
            asyncio.run(task())
            return
        with ThreadPoolExecutor() as executor:
            for question in questions:
                executor.submit(copy_context().run, partial(query, question))

    with suppress(openai.BadRequestError):
        if use_context_attributes:
            with using_attributes(
                session_id=session_id,
                user_id=user_id,
                metadata=metadata,
                tags=tags,
            ):
                main()
        else:
            main()

    spans = in_memory_span_exporter.get_finished_spans()
    traces: DefaultDict[int, Dict[str, ReadableSpan]] = defaultdict(dict)
    for span in spans:
        traces[span.context.trace_id][span.name] = span

    if is_stream:
        # OpenAIInstrumentor is on a separate trace because no span
        # is open when the stream iteration starts.
        assert len(traces) == n * 2
    else:
        assert len(traces) == n
    for spans_by_name in traces.values():
        spans_by_id = _spans_by_id(spans_by_name.values())
        if is_stream and len(spans_by_name) == 1:
            # This is the span from the OpenAIInstrumentor. It's on a separate
            # trace because no span is open when the stream iteration starts.
            continue
        if is_async:
            assert (query_span := spans_by_name.pop("BaseQueryEngine.aquery")) is not None
        else:
            assert (query_span := spans_by_name.pop("BaseQueryEngine.query")) is not None
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
            if not is_stream:
                assert query_attributes.pop(OUTPUT_VALUE, None) == answer
            else:
                assert query_attributes.pop(OUTPUT_VALUE, None) is not None
                assert query_attributes.pop(OUTPUT_MIME_TYPE, None)
        elif is_stream:
            assert query_attributes.pop(OUTPUT_VALUE, None) is not None
            assert query_attributes.pop(OUTPUT_MIME_TYPE, None)

        if is_async:
            assert (
                _query_span := spans_by_name.pop("RetrieverQueryEngine._aquery", None)
            ) is not None
        else:
            assert (
                _query_span := spans_by_name.pop("RetrieverQueryEngine._query", None)
            ) is not None
        assert _is_descendant(_query_span, query_span, spans_by_id)

        if use_context_attributes:
            _check_context_attributes(query_attributes, session_id, user_id, metadata, tags)
        assert query_attributes == {}  # all attributes should be accounted for

        if is_async:
            assert (retrieve_span := spans_by_name.pop("BaseRetriever.aretrieve", None)) is not None
        else:
            assert (retrieve_span := spans_by_name.pop("BaseRetriever.retrieve", None)) is not None
        assert _is_descendant(retrieve_span, _query_span, spans_by_id)
        retrieve_attributes = dict(retrieve_span.attributes or {})
        assert retrieve_attributes.pop(OPENINFERENCE_SPAN_KIND, None) == RETRIEVER.value
        assert retrieve_attributes.pop(INPUT_VALUE, None) == question
        assert retrieve_attributes.pop(OUTPUT_VALUE, None) is not None
        assert retrieve_attributes.pop(OUTPUT_MIME_TYPE, None)
        retrieve_attributes.pop(f"{RETRIEVAL_DOCUMENTS}.0.{DOCUMENT_ID}", None)
        retrieve_attributes.pop(f"{RETRIEVAL_DOCUMENTS}.1.{DOCUMENT_ID}", None)
        retrieve_attributes.pop(f"{RETRIEVAL_DOCUMENTS}.0.{DOCUMENT_SCORE}", 0.0)
        retrieve_attributes.pop(f"{RETRIEVAL_DOCUMENTS}.1.{DOCUMENT_SCORE}", 0.0)
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

        if is_async:
            assert (
                synthesize_span := spans_by_name.pop("BaseSynthesizer.asynthesize", None)
            ) is not None
        else:
            assert (
                synthesize_span := spans_by_name.pop("BaseSynthesizer.synthesize", None)
            ) is not None
        assert _is_descendant(synthesize_span, _query_span, spans_by_id)
        synthesize_attributes = dict(synthesize_span.attributes or {})
        assert synthesize_attributes.pop(OPENINFERENCE_SPAN_KIND, None) == CHAIN.value
        assert synthesize_attributes.pop(INPUT_VALUE, None) == question
        if status_code == 200:
            assert synthesize_span.status.status_code == trace_api.StatusCode.OK
            assert not synthesize_span.status.description
            if not is_stream:
                assert synthesize_attributes.pop(OUTPUT_VALUE, None) == answer
            else:
                assert synthesize_attributes.pop(OUTPUT_VALUE, None) is not None
                assert synthesize_attributes.pop(OUTPUT_MIME_TYPE, None)
        elif is_stream:
            assert synthesize_attributes.pop(OUTPUT_VALUE, None) is not None
            assert synthesize_attributes.pop(OUTPUT_MIME_TYPE, None)

        if use_context_attributes:
            _check_context_attributes(synthesize_attributes, session_id, user_id, metadata, tags)
        assert synthesize_attributes == {}  # all attributes should be accounted for

        if is_async:
            assert (_ := spans_by_name.pop("CompactAndRefine.aget_response", None)) is not None
            assert (refine_span := spans_by_name.pop("Refine.aget_response", None)) is not None
        else:
            assert (_ := spans_by_name.pop("CompactAndRefine.get_response", None)) is not None
            assert (refine_span := spans_by_name.pop("Refine.get_response", None)) is not None
        assert _is_descendant(refine_span, synthesize_span, spans_by_id)

        if is_async:
            if is_stream:
                assert (llm_span := spans_by_name.pop("LLM.astream", None)) is not None
            else:
                assert (llm_span := spans_by_name.pop("LLM.apredict", None)) is not None
        else:
            if is_stream:
                assert (llm_span := spans_by_name.pop("LLM.stream", None)) is not None
            else:
                assert (llm_span := spans_by_name.pop("LLM.predict", None)) is not None
        assert _is_descendant(llm_span, refine_span, spans_by_id)
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
        assert llm_attributes.pop(INPUT_VALUE, None) is not None
        assert llm_attributes.pop(INPUT_MIME_TYPE, None)
        if status_code == 200:
            assert llm_span.status.status_code == trace_api.StatusCode.OK
            assert not llm_span.status.description
            if not is_stream:
                assert llm_attributes.pop(OUTPUT_VALUE, None) == answer
            else:
                # FIXME: output should be propagated
                ...

        if use_context_attributes:
            _check_context_attributes(llm_attributes, session_id, user_id, metadata, tags)
        assert llm_attributes == {}  # all attributes should be accounted for

        if is_async:
            if is_stream:
                assert (openai_span := spans_by_name.pop("OpenAI.astream_chat")) is not None
            else:
                assert (openai_span := spans_by_name.pop("OpenAI.achat")) is not None
        else:
            if is_stream:
                assert (openai_span := spans_by_name.pop("OpenAI.stream_chat")) is not None
            else:
                assert (openai_span := spans_by_name.pop("OpenAI.chat")) is not None
        assert _is_descendant(openai_span, llm_span, spans_by_id)
        openai_attributes = dict(openai_span.attributes or {})
        assert openai_attributes.pop(OPENINFERENCE_SPAN_KIND, None) == LLM.value
        assert openai_attributes.pop(LLM_MODEL_NAME, None) is not None
        assert openai_attributes.pop(LLM_INVOCATION_PARAMETERS, None) is not None
        assert openai_attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}", None) == "system"
        assert openai_attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}", None) is not None
        assert openai_attributes.pop(f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_ROLE}", None) == "user"
        assert openai_attributes.pop(f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_CONTENT}", None) is not None
        assert openai_attributes.pop(INPUT_VALUE, None) is not None
        assert openai_attributes.pop(INPUT_MIME_TYPE, None)
        if status_code == 200:
            assert openai_span.status.status_code == trace_api.StatusCode.OK
            assert not openai_span.status.description
            assert openai_attributes.pop(OUTPUT_VALUE, None) == f"assistant: {answer}"
            assert (
                openai_attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}", None)
                == "assistant"
            )
            assert (
                openai_attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}", None) == answer
            )
        else:
            assert openai_span.status.status_code == trace_api.StatusCode.ERROR
            assert openai_span.status.description and openai_span.status.description.startswith(
                openai.BadRequestError.__name__,
            )
        if use_context_attributes:
            _check_context_attributes(openai_attributes, session_id, user_id, metadata, tags)
        assert openai_attributes == {}  # all attributes should be accounted for

        for span in spans_by_name.values():
            assert _is_descendant(span, query_span, spans_by_id)
    assert len(questions) == 0  # all questions should be accounted for


def _spans_by_id(spans: Iterable[ReadableSpan]) -> Dict[int, ReadableSpan]:
    spans_by_id = {}
    for span in spans:
        spans_by_id[span.context.span_id] = span
    return spans_by_id


def _is_descendant(
    span: Optional[ReadableSpan],
    ancestor: ReadableSpan,
    spans_by_id: Mapping[int, ReadableSpan],
) -> bool:
    if not ancestor.context:
        return False
    while span and span.parent:
        if span.parent.span_id == ancestor.context.span_id:
            return True
        span = spans_by_id.get(span.parent.span_id)
    return False


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
def tracer_provider(
    in_memory_span_exporter: InMemorySpanExporter,
) -> trace_api.TracerProvider:
    tracer_provider = trace_sdk.TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))
    return tracer_provider


@pytest.fixture(autouse=True)
def instrument(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> Generator[None, None, None]:
    LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
    OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    OpenAIInstrumentor().uninstrument()
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

JSON = OpenInferenceMimeTypeValues.JSON.value

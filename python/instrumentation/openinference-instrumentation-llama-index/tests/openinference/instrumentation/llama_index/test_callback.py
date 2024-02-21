import asyncio
import json
import logging
import random
from contextlib import suppress
from itertools import count
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Generator,
    Iterable,
    Iterator,
    List,
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
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from respx import MockRouter

for name, logger in logging.root.manager.loggerDict.items():
    if name.startswith("openinference.") and isinstance(logger, logging.Logger):
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()
        logger.addHandler(logging.StreamHandler())


@pytest.mark.parametrize(
    "is_stream,is_async",
    [
        (False, False),
        (False, True),
        (True, False),
        # FIXME: stream + async is not supported by LlamaIndex as of v0.9.33
        # (True, True),
    ],
)
@pytest.mark.parametrize("status_code", [200, 400])
def test_callback_llm(
    is_async: bool,
    is_stream: bool,
    status_code: int,
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
    nodes: List[Document],
    chat_completion_mock_stream: Tuple[List[bytes], List[Dict[str, Any]]],
) -> None:
    question = randstr()
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

    with suppress(openai.BadRequestError):
        if is_async:

            async def task() -> None:
                await query_engine.aquery(question)
                # FIXME: stream + async is not supported by LlamaIndex as of v0.9.33
                # if is_stream:
                #     async for _ in response.response_gen:
                #         pass

            asyncio.run(task())
        else:
            response = query_engine.query(question)
            if is_stream:
                for _ in cast(StreamingResponse, response).response_gen:
                    pass

    spans = in_memory_span_exporter.get_finished_spans()
    spans_by_name = {span.name: span for span in spans}

    assert (query_span := spans_by_name.pop("query")) is not None
    assert query_span.parent is None
    query_attributes = dict(query_span.attributes or {})
    assert query_attributes.pop(OPENINFERENCE_SPAN_KIND, None) == CHAIN.value
    assert query_attributes.pop(INPUT_VALUE, None) == question
    if status_code == 200:
        assert query_span.status.status_code == trace_api.StatusCode.OK
        assert not query_span.status.description
        assert query_attributes.pop(OUTPUT_VALUE, None) == answer
    elif (
        # FIXME: currently the error is propagated when streaming because we don't rely on
        # `on_event_end` to set the status code.
        status_code == 400 and is_stream
    ):
        assert query_span.status.status_code == trace_api.StatusCode.ERROR
        assert query_span.status.description and query_span.status.description.startswith(
            openai.BadRequestError.__name__,
        )
    assert query_attributes == {}

    assert (synthesize_span := spans_by_name.pop("synthesize")) is not None
    assert synthesize_span.parent is not None
    assert synthesize_span.parent.span_id == query_span.context.span_id
    assert synthesize_span.context.trace_id == query_span.context.trace_id
    synthesize_attributes = dict(synthesize_span.attributes or {})
    assert synthesize_attributes.pop(OPENINFERENCE_SPAN_KIND, None) == CHAIN.value
    assert synthesize_attributes.pop(INPUT_VALUE, None) == question
    if status_code == 200:
        assert synthesize_span.status.status_code == trace_api.StatusCode.OK
        assert not synthesize_span.status.description
        assert synthesize_attributes.pop(OUTPUT_VALUE, None) == answer
    elif (
        # FIXME: currently the error is propagated when streaming because we don't rely on
        # `on_event_end` to set the status code.
        status_code == 400 and is_stream
    ):
        assert synthesize_span.status.status_code == trace_api.StatusCode.ERROR
        assert query_span.status.description and query_span.status.description.startswith(
            openai.BadRequestError.__name__,
        )
    assert synthesize_attributes == {}

    assert (retrieve_span := spans_by_name.pop("retrieve")) is not None
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
    assert retrieve_attributes == {}

    if status_code == 200:
        # FIXME: LlamaIndex doesn't currently capture the LLM span when status_code == 400
        # For example, if an exception is raised by the LLM at the following location,
        # `on_event_end` never gets called.
        # https://github.com/run-llama/llama_index/blob/dcef41ee67925cccf1ee7bb2dd386bcf0564ba29/llama_index/llms/base.py#L100 # noqa E501
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
        assert llm_span.status.status_code == trace_api.StatusCode.OK
        assert not synthesize_span.status.description
        assert llm_attributes.pop(OUTPUT_VALUE, None) == answer
        if not is_stream:
            # FIXME: currently we can't capture messages when streaming
            assert (
                llm_attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}", None) == "assistant"
            )
            assert llm_attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}", None) == answer
        assert llm_attributes == {}

    # FIXME: maybe chunking spans should be discarded?
    assert (chunking_span := spans_by_name.pop("chunking", None)) is not None
    assert chunking_span.parent is not None
    assert chunking_span.parent.span_id == synthesize_span.context.span_id
    assert chunking_span.context.trace_id == synthesize_span.context.trace_id
    chunking_attributes = dict(chunking_span.attributes or {})
    assert chunking_attributes.pop(OPENINFERENCE_SPAN_KIND, None) is not None
    assert chunking_attributes == {}

    assert spans_by_name == {}


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
    LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
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

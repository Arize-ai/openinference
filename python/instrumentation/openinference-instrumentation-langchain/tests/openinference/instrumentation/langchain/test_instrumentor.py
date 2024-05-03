import asyncio
import json
import logging
import random
from contextlib import suppress
from itertools import count
from typing import Any, AsyncIterator, Dict, Generator, Iterable, Iterator, List, Tuple

import numpy as np
import openai
import pytest
from httpx import AsyncByteStream, Response, SyncByteStream
from langchain.chains import LLMChain, RetrievalQA
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.retrievers import KNNRetriever
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from openinference.instrumentation.langchain import LangChainInstrumentor
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
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.semconv.trace import SpanAttributes as OTELSpanAttributes
from respx import MockRouter

for name, logger in logging.root.manager.loggerDict.items():
    if name.startswith("openinference.") and isinstance(logger, logging.Logger):
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()
        logger.addHandler(logging.StreamHandler())


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("is_stream", [False, True])
@pytest.mark.parametrize("status_code", [200, 400])
def test_callback_llm(
    is_async: bool,
    is_stream: bool,
    status_code: int,
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
    documents: List[str],
    chat_completion_mock_stream: Tuple[List[bytes], List[Dict[str, Any]]],
    model_name: str,
    completion_usage: Dict[str, Any],
) -> None:
    question = randstr()
    template = "{context}{question}"
    prompt = PromptTemplate(input_variables=["context", "question"], template=template)
    output_messages: List[Dict[str, Any]] = (
        chat_completion_mock_stream[1] if is_stream else [{"role": randstr(), "content": randstr()}]
    )
    url = "https://api.openai.com/v1/chat/completions"
    respx_kwargs: Dict[str, Any] = {
        **(
            {"stream": MockByteStream(chat_completion_mock_stream[0])}
            if is_stream
            else {
                "json": {
                    "choices": [
                        {"index": i, "message": message, "finish_reason": "stop"}
                        for i, message in enumerate(output_messages)
                    ],
                    "model": model_name,
                    "usage": completion_usage,
                }
            }
        ),
    }
    respx_mock.post(url).mock(return_value=Response(status_code=status_code, **respx_kwargs))
    chat_model = ChatOpenAI(model_name="gpt-3.5-turbo", streaming=is_stream)  # type: ignore
    retriever = KNNRetriever(
        index=np.ones((len(documents), 2)),
        texts=documents,
        embeddings=FakeEmbeddings(size=2),
    )
    rqa = RetrievalQA.from_chain_type(
        llm=chat_model,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
    )
    with suppress(openai.BadRequestError):
        if is_async:
            asyncio.run(rqa.ainvoke({"query": question}))
        else:
            rqa.invoke({"query": question})

    spans = in_memory_span_exporter.get_finished_spans()
    spans_by_name = {span.name: span for span in spans}

    assert (rqa_span := spans_by_name.pop("RetrievalQA")) is not None
    assert rqa_span.parent is None
    rqa_attributes = dict(rqa_span.attributes or {})
    assert rqa_attributes.pop(OPENINFERENCE_SPAN_KIND, None) == CHAIN.value
    assert rqa_attributes.pop(INPUT_VALUE, None) == question
    if status_code == 200:
        assert rqa_span.status.status_code == trace_api.StatusCode.OK
        assert rqa_attributes.pop(OUTPUT_VALUE, None) == output_messages[0]["content"]
    elif status_code == 400:
        assert rqa_span.status.status_code == trace_api.StatusCode.ERROR
        assert rqa_span.events[0].name == "exception"
        assert (rqa_span.events[0].attributes or {}).get(
            OTELSpanAttributes.EXCEPTION_TYPE
        ) == "BadRequestError"
    assert rqa_attributes == {}

    assert (sd_span := spans_by_name.pop("StuffDocumentsChain")) is not None
    assert sd_span.parent is not None
    assert sd_span.parent.span_id == rqa_span.context.span_id
    assert sd_span.context.trace_id == rqa_span.context.trace_id
    sd_attributes = dict(sd_span.attributes or {})
    assert sd_attributes.pop(OPENINFERENCE_SPAN_KIND, None) == CHAIN.value
    assert sd_attributes.pop(INPUT_VALUE, None) is not None
    assert sd_attributes.pop(INPUT_MIME_TYPE, None) == JSON.value
    if status_code == 200:
        assert sd_span.status.status_code == trace_api.StatusCode.OK
        assert sd_attributes.pop(OUTPUT_VALUE, None) == output_messages[0]["content"]
    elif status_code == 400:
        assert sd_span.status.status_code == trace_api.StatusCode.ERROR
        assert sd_span.events[0].name == "exception"
        assert (sd_span.events[0].attributes or {}).get(
            OTELSpanAttributes.EXCEPTION_TYPE
        ) == "BadRequestError"
    assert sd_attributes == {}

    assert (retriever_span := spans_by_name.pop("Retriever")) is not None
    assert retriever_span.parent is not None
    assert retriever_span.parent.span_id == rqa_span.context.span_id
    assert retriever_span.context.trace_id == rqa_span.context.trace_id
    retriever_attributes = dict(retriever_span.attributes or {})
    assert retriever_attributes.pop(OPENINFERENCE_SPAN_KIND, None) == RETRIEVER.value
    assert retriever_attributes.pop(INPUT_VALUE, None) == question
    assert retriever_attributes.pop(OUTPUT_VALUE, None) is not None
    assert retriever_attributes.pop(OUTPUT_MIME_TYPE, None) == JSON.value
    for i, text in enumerate(documents):
        assert (
            retriever_attributes.pop(f"{RETRIEVAL_DOCUMENTS}.{i}.{DOCUMENT_CONTENT}", None) == text
        )
    assert retriever_attributes == {}

    assert (llm_span := spans_by_name.pop("LLMChain", None)) is not None
    assert llm_span.parent is not None
    assert llm_span.parent.span_id == sd_span.context.span_id
    assert llm_span.context.trace_id == sd_span.context.trace_id
    llm_attributes = dict(llm_span.attributes or {})
    assert llm_attributes.pop(OPENINFERENCE_SPAN_KIND, None) == CHAIN.value
    assert llm_attributes.pop(INPUT_VALUE, None) is not None
    assert llm_attributes.pop(INPUT_MIME_TYPE, None) == JSON.value
    assert llm_attributes.pop(LLM_PROMPT_TEMPLATE, None) == template
    assert isinstance(
        template_variables_json_string := llm_attributes.pop(LLM_PROMPT_TEMPLATE_VARIABLES, None),
        str,
    )
    assert json.loads(template_variables_json_string) == {
        "context": "\n\n".join(documents),
        "question": question,
    }
    if status_code == 200:
        assert llm_attributes.pop(OUTPUT_VALUE, None) == output_messages[0]["content"]
    elif status_code == 400:
        assert llm_span.status.status_code == trace_api.StatusCode.ERROR
        assert llm_span.events[0].name == "exception"
        assert (llm_span.events[0].attributes or {}).get(
            OTELSpanAttributes.EXCEPTION_TYPE
        ) == "BadRequestError"
    assert llm_attributes == {}

    assert (oai_span := spans_by_name.pop("ChatOpenAI", None)) is not None
    assert oai_span.parent is not None
    assert oai_span.parent.span_id == llm_span.context.span_id
    assert oai_span.context.trace_id == llm_span.context.trace_id
    oai_attributes = dict(oai_span.attributes or {})
    assert oai_attributes.pop(OPENINFERENCE_SPAN_KIND, None) == LLM.value
    assert oai_attributes.pop(LLM_MODEL_NAME, None) is not None
    assert oai_attributes.pop(LLM_INVOCATION_PARAMETERS, None) is not None
    assert oai_attributes.pop(INPUT_VALUE, None) is not None
    assert oai_attributes.pop(INPUT_MIME_TYPE, None) == JSON.value
    assert oai_attributes.pop(LLM_PROMPTS, None) is not None
    if status_code == 200:
        assert oai_span.status.status_code == trace_api.StatusCode.OK
        assert oai_attributes.pop(OUTPUT_VALUE, None) is not None
        assert oai_attributes.pop(OUTPUT_MIME_TYPE, None) == JSON.value
        assert (
            oai_attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}", None)
            == output_messages[0]["role"]
        )
        assert (
            oai_attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}", None)
            == output_messages[0]["content"]
        )
        if not is_stream:
            assert (
                oai_attributes.pop(LLM_TOKEN_COUNT_TOTAL, None) == completion_usage["total_tokens"]
            )
            assert (
                oai_attributes.pop(LLM_TOKEN_COUNT_PROMPT, None)
                == completion_usage["prompt_tokens"]
            )
            assert (
                oai_attributes.pop(LLM_TOKEN_COUNT_COMPLETION, None)
                == completion_usage["completion_tokens"]
            )
    elif status_code == 400:
        assert oai_span.status.status_code == trace_api.StatusCode.ERROR
        assert oai_span.events[0].name == "exception"
        assert (oai_span.events[0].attributes or {}).get(
            OTELSpanAttributes.EXCEPTION_TYPE
        ) == "BadRequestError"
    assert oai_attributes == {}

    assert spans_by_name == {}


def test_chain_metadata(
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
    completion_usage: Dict[str, Any],
) -> None:
    url = "https://api.openai.com/v1/chat/completions"
    respx_kwargs: Dict[str, Any] = {
        "json": {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "nock nock"},
                    "finish_reason": "stop",
                }
            ],
            "model": "gpt-3.5-turbo",
            "usage": completion_usage,
        }
    }
    respx_mock.post(url).mock(return_value=Response(status_code=200, **respx_kwargs))
    prompt_template = "Tell me a {adjective} joke"
    prompt = PromptTemplate(input_variables=["adjective"], template=prompt_template)
    llm = LLMChain(llm=ChatOpenAI(), prompt=prompt, metadata={"category": "jokes"})
    llm.predict(adjective="funny")
    spans = in_memory_span_exporter.get_finished_spans()
    spans_by_name = {span.name: span for span in spans}

    assert (llm_chain_span := spans_by_name.pop("LLMChain")) is not None
    assert llm_chain_span.attributes
    assert llm_chain_span.attributes.get(METADATA) == '{"category": "jokes"}'


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
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    LangChainInstrumentor().uninstrument()
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
TOOL_CALL_FUNCTION_ARGUMENTS_JSON = ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON
TOOL_CALL_FUNCTION_NAME = ToolCallAttributes.TOOL_CALL_FUNCTION_NAME
LLM_PROMPT_TEMPLATE = SpanAttributes.LLM_PROMPT_TEMPLATE
LLM_PROMPT_TEMPLATE_VARIABLES = SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES

CHAIN = OpenInferenceSpanKindValues.CHAIN
LLM = OpenInferenceSpanKindValues.LLM
RETRIEVER = OpenInferenceSpanKindValues.RETRIEVER

JSON = OpenInferenceMimeTypeValues.JSON

import asyncio
import json
import logging
import random
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from contextvars import copy_context
from functools import partial
from importlib.metadata import version
from typing import (
    Any,
    AsyncIterator,
    DefaultDict,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
)

import numpy as np
import openai
import pytest
from httpx import AsyncByteStream, Response, SyncByteStream
from langchain.chains import LLMChain, RetrievalQA
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.retrievers import KNNRetriever
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from openinference.instrumentation import using_attributes
from openinference.instrumentation.langchain import get_current_span
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
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.semconv.trace import SpanAttributes as OTELSpanAttributes
from opentelemetry.trace import Span
from respx import MockRouter

for name, logger in logging.root.manager.loggerDict.items():
    if name.startswith("openinference.") and isinstance(logger, logging.Logger):
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()
        logger.addHandler(logging.StreamHandler())

LANGCHAIN_VERSION = tuple(map(int, version("langchain-core").split(".")[:3]))
LANGCHAIN_OPENAI_VERSION = tuple(map(int, version("langchain-openai").split(".")[:3]))


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


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("is_stream", [False, True])
@pytest.mark.parametrize("status_code", [200, 400])
@pytest.mark.parametrize("use_context_attributes", [False, True])
def test_callback_llm(
    is_async: bool,
    is_stream: bool,
    status_code: int,
    use_context_attributes: bool,
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
    documents: List[str],
    chat_completion_mock_stream: Tuple[List[bytes], List[Dict[str, Any]]],
    model_name: str,
    completion_usage: Dict[str, Any],
    session_id: str,
    user_id: str,
    metadata: Dict[str, Any],
    tags: List[str],
    prompt_template: str,
    prompt_template_version: str,
    prompt_template_variables: Dict[str, Any],
) -> None:
    n = 10  # number of concurrent queries
    questions = {randstr() for _ in range(n)}
    langchain_template = "{context}{question}"
    langchain_prompt = PromptTemplate(
        input_variables=["context", "question"], template=langchain_template
    )
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
        chain_type_kwargs={"prompt": langchain_prompt},
    )

    async def task() -> None:
        await asyncio.gather(
            *(rqa.ainvoke({"query": question}) for question in questions),
            return_exceptions=True,
        )

    def main() -> None:
        if is_async:
            asyncio.run(task())
            return
        with ThreadPoolExecutor() as executor:
            for question in questions:
                executor.submit(
                    copy_context().run,
                    partial(rqa.invoke, {"query": question}),
                )

    with suppress(openai.BadRequestError):
        if use_context_attributes:
            with using_attributes(
                session_id=session_id,
                user_id=user_id,
                metadata=metadata,
                tags=tags,
                prompt_template=prompt_template,
                prompt_template_version=prompt_template_version,
                prompt_template_variables=prompt_template_variables,
            ):
                main()
        else:
            main()

    spans = in_memory_span_exporter.get_finished_spans()
    traces: DefaultDict[int, Dict[str, ReadableSpan]] = defaultdict(dict)
    for span in spans:
        traces[span.context.trace_id][span.name] = span

    assert len(traces) == n
    for spans_by_name in traces.values():
        assert (rqa_span := spans_by_name.pop("RetrievalQA")) is not None
        assert rqa_span.parent is None
        rqa_attributes = dict(rqa_span.attributes or {})
        assert rqa_attributes.pop(OPENINFERENCE_SPAN_KIND, None) == CHAIN.value
        question = rqa_attributes.pop(INPUT_VALUE, None)
        assert isinstance(question, str)
        assert question in questions
        questions.remove(question)
        if status_code == 200:
            assert rqa_span.status.status_code == trace_api.StatusCode.OK
            assert rqa_attributes.pop(OUTPUT_VALUE, None) == output_messages[0]["content"]
        elif status_code == 400:
            assert rqa_span.status.status_code == trace_api.StatusCode.ERROR
            assert rqa_span.events[0].name == "exception"
            exception_type = (rqa_span.events[0].attributes or {}).get(
                OTELSpanAttributes.EXCEPTION_TYPE
            )
            assert isinstance(exception_type, str)
            assert exception_type.endswith("BadRequestError")
        if use_context_attributes:
            _check_context_attributes(
                rqa_attributes,
                session_id=session_id,
                user_id=user_id,
                metadata=metadata,
                tags=tags,
                prompt_template=prompt_template,
                prompt_template_version=prompt_template_version,
                prompt_template_variables=prompt_template_variables,
            )
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
            exception_type = (sd_span.events[0].attributes or {}).get(
                OTELSpanAttributes.EXCEPTION_TYPE
            )
            assert isinstance(exception_type, str)
            assert exception_type.endswith("BadRequestError")
        if use_context_attributes:
            _check_context_attributes(
                sd_attributes,
                session_id=session_id,
                user_id=user_id,
                metadata=metadata,
                tags=tags,
                prompt_template=prompt_template,
                prompt_template_version=prompt_template_version,
                prompt_template_variables=prompt_template_variables,
            )
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
                retriever_attributes.pop(f"{RETRIEVAL_DOCUMENTS}.{i}.{DOCUMENT_CONTENT}", None)
                == text
            )
        if use_context_attributes:
            _check_context_attributes(
                retriever_attributes,
                session_id=session_id,
                user_id=user_id,
                metadata=metadata,
                tags=tags,
                prompt_template=prompt_template,
                prompt_template_version=prompt_template_version,
                prompt_template_variables=prompt_template_variables,
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
        if status_code == 200:
            assert llm_attributes.pop(OUTPUT_VALUE, None) == output_messages[0]["content"]
        elif status_code == 400:
            assert llm_span.status.status_code == trace_api.StatusCode.ERROR
            assert llm_span.events[0].name == "exception"
            exception_type = (llm_span.events[0].attributes or {}).get(
                OTELSpanAttributes.EXCEPTION_TYPE
            )
            assert isinstance(exception_type, str)
            assert exception_type.endswith("BadRequestError")
        langchain_prompt_variables = {
            "context": "\n\n".join(documents),
            "question": question,
        }
        if use_context_attributes:
            _check_context_attributes(
                llm_attributes,
                session_id=session_id,
                user_id=user_id,
                metadata=metadata,
                tags=tags,
                prompt_template=langchain_template,
                prompt_template_version=prompt_template_version,
                prompt_template_variables=langchain_prompt_variables,
            )
        else:
            assert (
                llm_attributes.pop(SpanAttributes.LLM_PROMPT_TEMPLATE, None) == langchain_template
            )
            assert isinstance(
                llm_prompt_template_variables := llm_attributes.pop(
                    SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES, None
                ),
                str,
            )
            assert json.loads(llm_prompt_template_variables) == langchain_prompt_variables
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
        assert oai_attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}", None) == "user"
        assert (
            oai_attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}", None)
            == "\n\n".join(documents) + question
        )
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
                    oai_attributes.pop(LLM_TOKEN_COUNT_TOTAL, None)
                    == completion_usage["total_tokens"]
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
            exception_type = (oai_span.events[0].attributes or {}).get(
                OTELSpanAttributes.EXCEPTION_TYPE
            )
            assert isinstance(exception_type, str)
            assert exception_type.endswith("BadRequestError")
        if use_context_attributes:
            _check_context_attributes(
                oai_attributes,
                session_id=session_id,
                user_id=user_id,
                metadata={
                    "ls_provider": "openai",
                    "ls_model_name": "gpt-3.5-turbo",
                    "ls_model_type": "chat",
                    "ls_temperature": 0.7,
                }
                if LANGCHAIN_VERSION >= (0, 2)
                else metadata,
                tags=tags,
                prompt_template=prompt_template,
                prompt_template_version=prompt_template_version,
                prompt_template_variables=prompt_template_variables,
            )
        else:
            if LANGCHAIN_VERSION >= (0, 2):
                assert isinstance(_metadata := oai_attributes.pop(METADATA, None), str)
                assert json.loads(_metadata) == {
                    "ls_provider": "openai",
                    "ls_model_name": "gpt-3.5-turbo",
                    "ls_model_type": "chat",
                    "ls_temperature": 0.7,
                }
        assert oai_attributes == {}

        assert spans_by_name == {}
    assert len(questions) == 0


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
                "usage": {"input_tokens": 22, "output_tokens": 5},
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


@pytest.mark.parametrize("use_context_attributes", [False, True])
@pytest.mark.parametrize("use_langchain_metadata", [False, True])
def test_chain_metadata(
    use_context_attributes: bool,
    use_langchain_metadata: bool,
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
    completion_usage: Dict[str, Any],
    session_id: str,
    user_id: str,
    metadata: Dict[str, Any],
    tags: List[str],
    prompt_template: str,
    prompt_template_version: str,
    prompt_template_variables: Dict[str, Any],
) -> None:
    url = "https://api.openai.com/v1/chat/completions"
    output_val = "nock nock"
    respx_kwargs: Dict[str, Any] = {
        "json": {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": output_val},
                    "finish_reason": "stop",
                }
            ],
            "model": "gpt-3.5-turbo",
            "usage": completion_usage,
        }
    }
    respx_mock.post(url).mock(return_value=Response(status_code=200, **respx_kwargs))
    langchain_prompt_template = "Tell me a {adjective} joke"
    prompt = PromptTemplate(input_variables=["adjective"], template=langchain_prompt_template)
    if use_langchain_metadata:
        langchain_metadata = {"category": "jokes"}
        llm = LLMChain(llm=ChatOpenAI(), prompt=prompt, metadata=langchain_metadata)
    else:
        llm = LLMChain(llm=ChatOpenAI(), prompt=prompt)
    langchain_prompt_variables = {
        "adjective": "funny",
    }
    if use_context_attributes:
        with using_attributes(
            session_id=session_id,
            user_id=user_id,
            metadata=metadata,
            tags=tags,
            # We will test that this prompt template does not overwrite the passed prompt template
            prompt_template=prompt_template,
            prompt_template_version=prompt_template_version,
            # We will test that these variables do not overwrite the passed variables
            prompt_template_variables=prompt_template_variables,
        ):
            llm.predict(**langchain_prompt_variables)  # type: ignore[arg-type,unused-ignore]
    else:
        llm.predict(**langchain_prompt_variables)  # type: ignore[arg-type,unused-ignore]
    spans = in_memory_span_exporter.get_finished_spans()
    spans_by_name = {span.name: span for span in spans}

    assert (llm_chain_span := spans_by_name.pop("LLMChain")) is not None
    llm_attributes = dict(llm_chain_span.attributes or {})
    assert llm_attributes
    if use_langchain_metadata:
        check_metadata = langchain_metadata
    else:
        if use_context_attributes:
            check_metadata = metadata
        else:
            check_metadata = None

    _check_context_attributes(
        attributes=llm_attributes,
        session_id=session_id if use_context_attributes else None,
        user_id=user_id if use_context_attributes else None,
        metadata=check_metadata,
        tags=tags if use_context_attributes else None,
        prompt_template=langchain_prompt_template,
        prompt_template_version=prompt_template_version if use_context_attributes else None,
        prompt_template_variables=langchain_prompt_variables,
    )
    assert (
        llm_attributes.pop(OPENINFERENCE_SPAN_KIND, None) == OpenInferenceSpanKindValues.CHAIN.value
    )
    assert llm_attributes.pop(INPUT_VALUE, None) == langchain_prompt_variables["adjective"]
    assert llm_attributes.pop(OUTPUT_VALUE, None) == output_val
    assert llm_attributes == {}


@pytest.mark.parametrize(
    "session_metadata, expected_session_id",
    [
        (
            {
                "session_id": "test-langchain-session-id",
            },
            "test-langchain-session-id",
        ),
        (
            {
                "conversation_id": "test-langchain-conversation-id",
            },
            "test-langchain-conversation-id",
        ),
        (
            {
                "thread_id": "test-langchain-thread-id",
            },
            "test-langchain-thread-id",
        ),
        (
            {
                "session_id": "test-langchain-session-id",
                "conversation_id": "test-langchain-conversation-id",
                "thread_id": "test-langchain-thread-id",
            },
            "test-langchain-session-id",
        ),
        (
            {
                "conversation_id": "test-langchain-conversation-id",
                "thread_id": "test-langchain-thread-id",
            },
            "test-langchain-conversation-id",
        ),
    ],
)
@pytest.mark.parametrize("use_context_attributes", [False, True])
def test_read_session_from_metadata(
    use_context_attributes: bool,
    session_metadata: Dict[str, str],
    expected_session_id: str,
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
    completion_usage: Dict[str, Any],
    session_id: str,
    user_id: str,
    metadata: Dict[str, Any],
    tags: List[str],
    prompt_template: str,
    prompt_template_version: str,
    prompt_template_variables: Dict[str, Any],
) -> None:
    url = "https://api.openai.com/v1/chat/completions"
    output_val = "nock nock"
    respx_kwargs: Dict[str, Any] = {
        "json": {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": output_val},
                    "finish_reason": "stop",
                }
            ],
            "model": "gpt-3.5-turbo",
            "usage": completion_usage,
        }
    }
    respx_mock.post(url).mock(return_value=Response(status_code=200, **respx_kwargs))
    langchain_prompt_template = "Tell me a {adjective} joke"
    prompt = PromptTemplate(input_variables=["adjective"], template=langchain_prompt_template)
    langchain_metadata = session_metadata
    langchain_metadata["category"] = "jokes"
    llm = LLMChain(llm=ChatOpenAI(), prompt=prompt, metadata=langchain_metadata)
    langchain_prompt_variables = {
        "adjective": "funny",
    }
    if use_context_attributes:
        with using_attributes(
            session_id=session_id,
            user_id=user_id,
            metadata=metadata,
            tags=tags,
            # We will test that this prompt template does not overwrite the passed prompt template
            prompt_template=prompt_template,
            prompt_template_version=prompt_template_version,
            # We will test that these variables do not overwrite the passed variables
            prompt_template_variables=prompt_template_variables,
        ):
            llm.predict(**langchain_prompt_variables)  # type: ignore[arg-type,unused-ignore]
    else:
        llm.predict(**langchain_prompt_variables)  # type: ignore[arg-type,unused-ignore]
    spans = in_memory_span_exporter.get_finished_spans()
    spans_by_name = {span.name: span for span in spans}

    assert (llm_chain_span := spans_by_name.pop("LLMChain")) is not None
    llm_attributes = dict(llm_chain_span.attributes or {})
    print(f"{llm_attributes=}")
    assert llm_attributes

    _check_context_attributes(
        attributes=llm_attributes,
        session_id=expected_session_id,
        user_id=user_id if use_context_attributes else None,
        metadata=langchain_metadata,
        tags=tags if use_context_attributes else None,
        prompt_template=langchain_prompt_template,
        prompt_template_version=prompt_template_version if use_context_attributes else None,
        prompt_template_variables=langchain_prompt_variables,
    )
    assert (
        llm_attributes.pop(OPENINFERENCE_SPAN_KIND, None) == OpenInferenceSpanKindValues.CHAIN.value
    )
    assert llm_attributes.pop(INPUT_VALUE, None) == langchain_prompt_variables["adjective"]
    assert llm_attributes.pop(OUTPUT_VALUE, None) == output_val
    assert llm_attributes == {}


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
    openai_api_key: str,
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
        # print(prompt_template_variables)
        # x = attributes.pop(SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES, None)
        # print(x)
        # assert x == json.dumps(prompt_template_variables)
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
SESSION_ID = SpanAttributes.SESSION_ID
USER_ID = SpanAttributes.USER_ID
TAG_TAGS = SpanAttributes.TAG_TAGS

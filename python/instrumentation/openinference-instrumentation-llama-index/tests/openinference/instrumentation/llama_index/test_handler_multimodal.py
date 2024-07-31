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
    Optional,
    Tuple,
    cast,
)

import openai
import pytest
from httpx import AsyncByteStream, Response, SyncByteStream
from llama_index.core import Document
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.multi_modal_llms.generic_utils import load_image_urls
from llama_index.core.schema import TextNode
from llama_index.llms.openai import OpenAI  # type: ignore
from llama_index.multi_modal_llms.openai import OpenAIMultiModal  # type: ignore
from llama_index.multi_modal_llms.openai import utils as openai_utils
from openinference.instrumentation import using_attributes
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from openinference.semconv.trace import (
    ImageAttributes,
    MessageAttributes,
    MessageContentAttributes,
    OpenInferenceSpanKindValues,
    SpanAttributes,
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
def test_handler_multimodal(
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
    image_url = (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/"
        "Gfp-wisconsin-madison-the-nature-boardwalk.jpg/"
        "2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
    )
    image_documents = load_image_urls([image_url])
    chat_prompt = "Describe the images as an alternative text"
    chat_msg = openai_utils.generate_openai_multi_modal_chat_message(
        prompt=chat_prompt,
        role="user",
        image_documents=image_documents,
    )
    messages = [chat_msg for _ in range(n)]
    answer = (
        "The image shows a scenic landscape with a wooden boardwalk pathway "
        "leading through a grassy field. The sky is clear and blue with some "
        "scattered clouds. There is greenery on either side of the walkway, "
        "including tall grasses and bushes, and trees in the background. "
        "This peaceful and open setting is suggestive of a nature reserve "
        "or park."
    )
    llm_model_name = "gpt-4o"
    llm = OpenAIMultiModal(
        model=llm_model_name,
        max_retries=0,
        timeout=0.01,
    )
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

    def chat(message: ChatMessage) -> None:
        if is_stream:
            resp_gen = llm.stream_chat([message])
            _ = [r for r in resp_gen]
        else:
            _ = llm.chat([message])

    async def achat(message: ChatMessage) -> None:
        if is_stream:
            resp_gen = await llm.astream_chat([message])
            _ = [r async for r in resp_gen]
        else:
            _ = await llm.achat([message])

    async def task() -> None:
        await asyncio.gather(*(achat(message) for message in messages), return_exceptions=True)

    def main() -> None:
        if is_async:
            asyncio.run(task())
            return
        with ThreadPoolExecutor() as executor:
            for message in messages:
                executor.submit(copy_context().run, partial(chat, message))

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

    assert len(traces) == n
    for spans_by_name in traces.values():
        if is_stream:
            if is_async:
                name = "OpenAIMultiModal.astream_chat"
            else:
                name = "OpenAIMultiModal.stream_chat"
        else:
            if is_async:
                name = "OpenAIMultiModal.achat"
            else:
                name = "OpenAIMultiModal.chat"
        span = spans_by_name.pop(name)
        assert span is not None
        assert span.parent is None
        attributes = dict(span.attributes or {})
        assert attributes.pop(OPENINFERENCE_SPAN_KIND, None) == LLM.value
        if use_context_attributes:
            _check_context_attributes(attributes, session_id, user_id, metadata, tags)
        assert attributes.pop(LLM_MODEL_NAME, None) == llm_model_name
        assert attributes.pop(LLM_INVOCATION_PARAMETERS, None) is not None
        # Input value
        input_value = cast(Optional[str], attributes.pop(INPUT_VALUE, None))
        assert input_value is not None
        input_value = json.loads(input_value)
        input_messages = input_value.pop("messages")  # type:ignore
        assert len(input_messages) == 1
        input_message = input_messages[0]
        assert input_message.pop("role") == messages[0].role.value
        assert input_message.pop("content") == messages[0].content
        assert input_message.pop("additional_kwargs") == messages[0].additional_kwargs
        assert input_message == {}

        # Input messages
        assert attributes.pop(INPUT_MIME_TYPE, None) == "application/json"
        assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}", None) == "user"
        assert (
            attributes.pop(
                f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TYPE}",
                None,
            )
            == "text"
        )
        assert (
            attributes.pop(
                f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TEXT}",
                None,
            )
            == chat_prompt
        )
        assert (
            attributes.pop(
                f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.1.{MESSAGE_CONTENT_TYPE}",
                None,
            )
            == "image"
        )
        assert (
            attributes.pop(
                f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.1.{MESSAGE_CONTENT_IMAGE}.{IMAGE_URL}",
                None,
            )
            == image_url
        )
        # Output value
        if status_code == 200:
            assert span.status.status_code == trace_api.StatusCode.OK
            assert not span.status.description
            output_value = cast(Optional[str], attributes.pop(OUTPUT_VALUE, None))
            if is_stream:
                assert output_value == "assistant: ABC"
            else:
                assert output_value == f"assistant: {answer}"

        # Output messages
        if status_code == 200:
            assert (
                attributes.pop(
                    f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}",
                    None,
                )
                == "assistant"
            )
            if is_stream:
                assert (
                    attributes.pop(
                        f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}",
                        None,
                    )
                    == "ABC"
                )
            else:
                assert (
                    attributes.pop(
                        f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}",
                        None,
                    )
                    == answer
                )

        assert attributes == {}


def _spans_by_id(spans: Iterable[ReadableSpan]) -> Dict[int, ReadableSpan]:
    spans_by_id = {}
    for span in spans:
        spans_by_id[span.context.span_id] = span
    return spans_by_id


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


INPUT_MIME_TYPE = SpanAttributes.INPUT_MIME_TYPE
INPUT_VALUE = SpanAttributes.INPUT_VALUE
LLM_INPUT_MESSAGES = SpanAttributes.LLM_INPUT_MESSAGES
LLM_INVOCATION_PARAMETERS = SpanAttributes.LLM_INVOCATION_PARAMETERS
LLM_MODEL_NAME = SpanAttributes.LLM_MODEL_NAME
LLM_OUTPUT_MESSAGES = SpanAttributes.LLM_OUTPUT_MESSAGES
MESSAGE_CONTENT = MessageAttributes.MESSAGE_CONTENT
MESSAGE_CONTENTS = MessageAttributes.MESSAGE_CONTENTS
MESSAGE_CONTENT_TYPE = MessageContentAttributes.MESSAGE_CONTENT_TYPE
MESSAGE_CONTENT_TEXT = MessageContentAttributes.MESSAGE_CONTENT_TEXT
MESSAGE_CONTENT_IMAGE = MessageContentAttributes.MESSAGE_CONTENT_IMAGE
IMAGE_URL = ImageAttributes.IMAGE_URL
MESSAGE_ROLE = MessageAttributes.MESSAGE_ROLE
OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
LLM = OpenInferenceSpanKindValues.LLM
SESSION_ID = SpanAttributes.SESSION_ID
USER_ID = SpanAttributes.USER_ID
METADATA = SpanAttributes.METADATA
TAG_TAGS = SpanAttributes.TAG_TAGS

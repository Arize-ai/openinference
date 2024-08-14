import asyncio
from typing import (
    Any,
    Dict,
    Generator,
    List,
    Optional,
    Type,
    Union,
    cast,
)

import pytest
from anthropic import Anthropic, AsyncAnthropic
from anthropic._base_client import _StreamT
from anthropic._types import Body, RequestFiles, RequestOptions, ResponseT
from anthropic.resources.completions import AsyncCompletions, Completions
from anthropic.resources.messages import (  # type: ignore[attr-defined]
    AsyncMessages,
    Message,
    Messages,
)
from anthropic.types import TextBlock, Usage
from anthropic.types.completion import Completion
from openinference.instrumentation import OITracer, using_attributes
from openinference.instrumentation.anthropic import AnthropicInstrumentor
from openinference.semconv.trace import SpanAttributes
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from wrapt import BoundFunctionWrapper

mock_completion = Completion(
    id="chat_comp_0",
    completion="sorry, idk!",
    model="fake_model",
    type="completion",
)

mock_message = Message(
    id="msg_018gCsTGsXkYJVqYPxTgDHBU",
    type="message",
    model="fake_model",
    role="assistant",
    content=[TextBlock(type="text", text="Sure, I'd be happy to provide...")],
    stop_reason="end_turn",
    stop_sequence=None,
    usage=Usage(input_tokens=30, output_tokens=309),
)


def _mock_post_generation(
    self: Any,
    path: str = "fake/url",
    *,
    cast_to: Type[ResponseT],
    body: Optional[Body] = None,
    options: RequestOptions = {},
    files: Optional[RequestFiles] = None,
    stream: bool = False,
    stream_cls: Optional[Type[_StreamT]] = None,
) -> Union[ResponseT, _StreamT]:
    """
    opts = FinalRequestOptions.construct(
        method="post", url=path, json_data=body, files=to_httpx_files(files), **options
    )
    return cast(ResponseT, self.request(cast_to, opts, stream=stream, stream_cls=stream_cls))
    """
    return cast(ResponseT, mock_completion)


def _mock_post_messages(
    self: Any,
    path: str = "fake/url",
    *,
    cast_to: Type[ResponseT],
    body: Optional[Body] = None,
    options: RequestOptions = {},
    files: Optional[RequestFiles] = None,
    stream: bool = False,
    stream_cls: Optional[Type[_StreamT]] = None,
) -> Union[ResponseT, _StreamT]:
    """
    opts = FinalRequestOptions.construct(
        method="post", url=path, json_data=body, files=to_httpx_files(files), **options
    )
    return cast(ResponseT, self.request(cast_to, opts, stream=stream, stream_cls=stream_cls))
    """
    return cast(ResponseT, mock_message)


async def _async_mock_post_generation(
    self: Any,
    path: str = "fake/url",
    *,
    cast_to: Type[ResponseT],
    body: Optional[Body] = None,
    options: RequestOptions = {},
    files: Optional[RequestFiles] = None,
    stream: bool = False,
    stream_cls: Optional[Type[_StreamT]] = None,
) -> Union[ResponseT, _StreamT]:
    """
    opts = FinalRequestOptions.construct(
        method="post", url=path, json_data=body, files=to_httpx_files(files), **options
    )
    return cast(ResponseT, self.request(cast_to, opts, stream=stream, stream_cls=stream_cls))
    """
    return cast(ResponseT, mock_completion)


async def _async_mock_post_messages(
    self: Any,
    path: str = "fake/url",
    *,
    cast_to: Type[ResponseT],
    body: Optional[Body] = None,
    options: RequestOptions = {},
    files: Optional[RequestFiles] = None,
    stream: bool = False,
    stream_cls: Optional[Type[_StreamT]] = None,
) -> Union[ResponseT, _StreamT]:
    """
    opts = FinalRequestOptions.construct(
        method="post", url=path, json_data=body, files=to_httpx_files(files), **options
    )
    return cast(ResponseT, self.request(cast_to, opts, stream=stream, stream_cls=stream_cls))
    """
    return cast(ResponseT, mock_message)


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
def setup_anthropic_instrumentation(
    tracer_provider: TracerProvider,
) -> Generator[None, None, None]:
    AnthropicInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    AnthropicInstrumentor().uninstrument()


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


def test_anthropic_instrumentation_generation(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
    session_id: str,
    user_id: str,
    metadata: Dict[str, Any],
    tags: List[str],
    prompt_template: str,
    prompt_template_version: str,
    prompt_template_variables: Dict[str, Any],
) -> None:
    client = Anthropic(api_key="fake")
    client.completions._post = _mock_post_generation  # type: ignore[assignment]

    with using_attributes(
        session_id=session_id,
        user_id=user_id,
        metadata=metadata,
        tags=tags,
        prompt_template=prompt_template,
        prompt_template_version=prompt_template_version,
        prompt_template_variables=prompt_template_variables,
    ):
        client.completions.create(
            model="claude-2.1",
            prompt="How does a court case get to the Supreme Court?",
            max_tokens_to_sample=1000,
        )

    spans = in_memory_span_exporter.get_finished_spans()

    assert spans[0].name == "Completions"
    assert spans[0].attributes and spans[0].attributes.get("openinference.span.kind") == "LLM"

    for span in spans:
        att = span.attributes
        _check_context_attributes(
            att,
        )


def test_anthropic_instrumentation_chat_completion(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
    session_id: str,
    user_id: str,
    metadata: Dict[str, Any],
    tags: List[str],
    prompt_template: str,
    prompt_template_version: str,
    prompt_template_variables: Dict[str, Any],
) -> None:
    client = Anthropic(api_key="fake")
    client.messages._post = _mock_post_messages  # type: ignore[assignment]

    with using_attributes(
        session_id=session_id,
        user_id=user_id,
        metadata=metadata,
        tags=tags,
        prompt_template=prompt_template,
        prompt_template_version=prompt_template_version,
        prompt_template_variables=prompt_template_variables,
    ):
        client.messages.create(
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": "Hello!",
                }
            ],
            model="claude-3-opus-20240229",
        )

    spans = in_memory_span_exporter.get_finished_spans()

    assert spans[0].name == "Messages"
    assert spans[0].attributes and spans[0].attributes.get("openinference.span.kind") == "LLM"

    for span in spans:
        att = span.attributes
        _check_context_attributes(
            att,
        )


def test_anthropic_async_instrumentation_generation(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
    session_id: str,
    user_id: str,
    metadata: Dict[str, Any],
    tags: List[str],
    prompt_template: str,
    prompt_template_version: str,
    prompt_template_variables: Dict[str, Any],
) -> None:
    client = AsyncAnthropic(api_key="fake")
    client.completions._post = _async_mock_post_generation  # type: ignore[assignment]

    async def exec_comp() -> None:
        await client.completions.create(
            model="claude-2.1",
            prompt="How does a court case get to the Supreme Court?",
            max_tokens_to_sample=1000,
        )

    with using_attributes(
        session_id=session_id,
        user_id=user_id,
        metadata=metadata,
        tags=tags,
        prompt_template=prompt_template,
        prompt_template_version=prompt_template_version,
        prompt_template_variables=prompt_template_variables,
    ):
        asyncio.run(exec_comp())

    spans = in_memory_span_exporter.get_finished_spans()

    assert spans[0].name == "AsyncCompletions"
    assert spans[0].attributes and spans[0].attributes.get("openinference.span.kind") == "LLM"

    for span in spans:
        att = span.attributes
        _check_context_attributes(
            att,
        )


def test_anthropic_async_instrumentation_chat_completion(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
    session_id: str,
    user_id: str,
    metadata: Dict[str, Any],
    tags: List[str],
    prompt_template: str,
    prompt_template_version: str,
    prompt_template_variables: Dict[str, Any],
) -> None:
    client = AsyncAnthropic(api_key="fake")
    client.messages._post = _async_mock_post_messages  # type: ignore[assignment]

    async def exec_comp() -> None:
        await client.messages.create(
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": "Hello!",
                }
            ],
            model="claude-3-opus-20240229",
        )

    with using_attributes(
        session_id=session_id,
        user_id=user_id,
        metadata=metadata,
        tags=tags,
        prompt_template=prompt_template,
        prompt_template_version=prompt_template_version,
        prompt_template_variables=prompt_template_variables,
    ):
        asyncio.run(exec_comp())

    spans = in_memory_span_exporter.get_finished_spans()

    assert spans[0].name == "AsyncMessages"
    assert spans[0].attributes and spans[0].attributes.get("openinference.span.kind") == "LLM"

    for span in spans:
        att = span.attributes
        _check_context_attributes(
            att,
        )


def test_anthropic_uninstrumentation(
    tracer_provider: TracerProvider,
) -> None:
    AnthropicInstrumentor().instrument(tracer_provider=tracer_provider)

    assert isinstance(Completions.create, BoundFunctionWrapper)
    assert isinstance(Messages.create, BoundFunctionWrapper)
    assert isinstance(AsyncCompletions.create, BoundFunctionWrapper)
    assert isinstance(AsyncMessages.create, BoundFunctionWrapper)

    AnthropicInstrumentor().uninstrument()

    assert not isinstance(Completions.create, BoundFunctionWrapper)
    assert not isinstance(Messages.create, BoundFunctionWrapper)
    assert not isinstance(AsyncCompletions.create, BoundFunctionWrapper)
    assert not isinstance(AsyncMessages.create, BoundFunctionWrapper)


# Ensure we're using the common OITracer from common openinference-instrumentation pkg
def test_oitracer(
    setup_anthropic_instrumentation: Any,
) -> None:
    assert isinstance(AnthropicInstrumentor()._tracer, OITracer)


SESSION_ID = SpanAttributes.SESSION_ID
USER_ID = SpanAttributes.USER_ID
METADATA = SpanAttributes.METADATA
TAG_TAGS = SpanAttributes.TAG_TAGS
LLM_PROMPT_TEMPLATE = SpanAttributes.LLM_PROMPT_TEMPLATE
LLM_PROMPT_TEMPLATE_VERSION = SpanAttributes.LLM_PROMPT_TEMPLATE_VERSION
LLM_PROMPT_TEMPLATE_VARIABLES = SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES

import asyncio
from typing import (
    Any,
    Generator,
    Optional,
    Type,
    Union,
    cast,
)

import pytest
from groq import AsyncGroq, Groq
from groq._base_client import _StreamT
from groq._types import Body, RequestFiles, RequestOptions, ResponseT
from groq.resources.chat.completions import AsyncCompletions, Completions
from groq.types.chat.chat_completion import (  # type: ignore[attr-defined]
    ChatCompletion,
    ChatCompletionMessage,
    Choice,
)
from groq.types.completion_usage import CompletionUsage
from openinference.instrumentation.groq import GroqInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

mock_completion = ChatCompletion(
    id="chat_comp_0",
    choices=[
        Choice(
            finish_reason="stop",
            index=0,
            logprobs=None,
            message=ChatCompletionMessage(
                content="idk, sorry!", role="assistant", function_call=None, tool_calls=None
            ),
        )
    ],
    created=1722531851,
    model="fake_model",
    object="chat.completion",
    system_fingerprint="fp0",
    usage=CompletionUsage(
        completion_tokens=379,
        prompt_tokens=25,
        total_tokens=404,
        completion_time=0.616262398,
        prompt_time=0.002549632,
        queue_time=None,
        total_time=0.6188120300000001,
    ),
)


def _mock_post(
    self: Any,
    path: str = "fake/url",
    *,
    cast_to: Type[ResponseT],
    body: Optional[Body] = None,
    options: RequestOptions = {},
    files: Optional[RequestFiles] = None,
    stream: bool = False,
    stream_cls: Optional[type[_StreamT]] = None,
) -> Union[ResponseT, _StreamT]:
    """
    opts = FinalRequestOptions.construct(
        method="post", url=path, json_data=body, files=to_httpx_files(files), **options
    )
    return cast(ResponseT, self.request(cast_to, opts, stream=stream, stream_cls=stream_cls))
    """
    return cast(ResponseT, mock_completion)


async def _async_mock_post(
    self: Any,
    path: str = "fake/url",
    *,
    cast_to: Type[ResponseT],
    body: Optional[Body] = None,
    options: RequestOptions = {},
    files: Optional[RequestFiles] = None,
    stream: bool = False,
    stream_cls: Optional[type[_StreamT]] = None,  # type:ignore[type-arg]
) -> Union[ResponseT, _StreamT]:
    """
    opts = FinalRequestOptions.construct(
        method="post", url=path, json_data=body, files=to_httpx_files(files), **options
    )
    return cast(ResponseT, self.request(cast_to, opts, stream=stream, stream_cls=stream_cls))
    """
    return cast(ResponseT, mock_completion)


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
def setup_groq_instrumentation(
    tracer_provider: TracerProvider,
) -> Generator[None, None, None]:
    GroqInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    GroqInstrumentor().uninstrument()


def test_groq_instrumentation(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_groq_instrumentation: Any,
) -> None:
    client = Groq(api_key="fake")
    client.chat.completions._post = _mock_post  # type: ignore[assignment]

    client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Explain the importance of low latency LLMs",
            }
        ],
        model="fake_model",
    )
    spans = in_memory_span_exporter.get_finished_spans()

    assert spans[0].name == "Completions"
    assert spans[0].attributes and spans[0].attributes.get("openinference.span.kind") == "LLM"


def test_groq_async_instrumentation(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_groq_instrumentation: Any,
) -> None:
    client = AsyncGroq(api_key="fake")
    client.chat.completions._post = _async_mock_post  # type: ignore[assignment]

    async def exec_comp() -> None:
        await client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": "Explain the importance of low latency LLMs",
                }
            ],
            model="fake_model",
        )

    asyncio.run(exec_comp())

    spans = in_memory_span_exporter.get_finished_spans()

    assert spans[0].name == "AsyncCompletions"
    assert spans[0].attributes and spans[0].attributes.get("openinference.span.kind") == "LLM"


def test_groq_uninstrumentation(
    tracer_provider: TracerProvider,
) -> None:
    GroqInstrumentor().instrument(tracer_provider=tracer_provider)

    assert hasattr(Completions.create, "__wrapped__")
    assert hasattr(AsyncCompletions.create, "__wrapped__")

    GroqInstrumentor().uninstrument()

    assert not hasattr(Completions.create, "__wrapped__")
    assert not hasattr(AsyncCompletions.create, "__wrapped__")

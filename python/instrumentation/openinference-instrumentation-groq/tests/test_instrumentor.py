import asyncio
from typing import (
    Any,
    Dict,
    Generator,
    List,
    Mapping,
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
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.util.types import AttributeValue

from openinference.instrumentation import OITracer, using_attributes
from openinference.instrumentation.groq import GroqInstrumentor
from openinference.semconv.trace import (
    MessageAttributes,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)

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
    stream_cls: Optional[Type[_StreamT]] = None,
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
    stream_cls: Optional[Type[_StreamT]] = None,
) -> Union[ResponseT, _StreamT]:
    """
    opts = FinalRequestOptions.construct(
        method="post", url=path, json_data=body, files=to_httpx_files(files), **options
    )
    return cast(ResponseT, self.request(cast_to, opts, stream=stream, stream_cls=stream_cls))
    """
    return cast(ResponseT, mock_completion)


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
def setup_groq_instrumentation(
    tracer_provider: TracerProvider,
) -> Generator[None, None, None]:
    GroqInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    GroqInstrumentor().uninstrument()


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


def test_groq_instrumentation(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_groq_instrumentation: Any,
    session_id: str,
    user_id: str,
    metadata: Dict[str, Any],
    tags: List[str],
    prompt_template: str,
    prompt_template_version: str,
    prompt_template_variables: Dict[str, Any],
) -> None:
    client = Groq(api_key="fake")
    client.chat.completions._post = _mock_post  # type: ignore[assignment]

    with using_attributes(
        session_id=session_id,
        user_id=user_id,
        metadata=metadata,
        tags=tags,
        prompt_template=prompt_template,
        prompt_template_version=prompt_template_version,
        prompt_template_variables=prompt_template_variables,
    ):
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

    for span in spans:
        att = span.attributes
        _check_context_attributes(
            att,
        )

    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))
    assert attributes[f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}"] == "user"
    assert attributes[f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}"] == "Explain the importance of low latency LLMs"
    print(attributes)


def test_groq_async_instrumentation(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_groq_instrumentation: Any,
    session_id: str,
    user_id: str,
    metadata: Dict[str, Any],
    tags: List[str],
    prompt_template: str,
    prompt_template_version: str,
    prompt_template_variables: Dict[str, Any],
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


def test_groq_uninstrumentation(
    tracer_provider: TracerProvider,
) -> None:
    GroqInstrumentor().instrument(tracer_provider=tracer_provider)

    assert hasattr(Completions.create, "__wrapped__")
    assert hasattr(AsyncCompletions.create, "__wrapped__")

    GroqInstrumentor().uninstrument()

    assert not hasattr(Completions.create, "__wrapped__")
    assert not hasattr(AsyncCompletions.create, "__wrapped__")


# Ensure we're using the common OITracer from common opeinference-instrumentation pkg
def test_oitracer(
    setup_groq_instrumentation: Any,
) -> None:
    assert isinstance(GroqInstrumentor()._tracer, OITracer)


SESSION_ID = SpanAttributes.SESSION_ID
USER_ID = SpanAttributes.USER_ID
METADATA = SpanAttributes.METADATA
TAG_TAGS = SpanAttributes.TAG_TAGS
LLM_PROMPT_TEMPLATE = SpanAttributes.LLM_PROMPT_TEMPLATE
LLM_PROMPT_TEMPLATE_VERSION = SpanAttributes.LLM_PROMPT_TEMPLATE_VERSION
LLM_PROMPT_TEMPLATE_VARIABLES = SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES

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
import anthropic
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
from openinference.semconv.trace import (
    DocumentAttributes,
    EmbeddingAttributes,
    MessageAttributes,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolCallAttributes,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from wrapt import BoundFunctionWrapper


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


mock_completion = Completion(
    id="chat_comp_0",
    completion="idk",
    model="fake_model",
    type="completion",
)

mock_message = Message(
    id="msg_018gCsTGsXkYJVqYPxTgDHBU",
    type="message",
    model="fake_model",
    role="assistant",
    content=[TextBlock(type="text", text="idk")],
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

@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
def test_anthropic_instrumentation_completions(
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
    #client = Anthropic(api_key="fake")
    #client.completions._post = _mock_post_generation  # type: ignore[assignment]
    client = Anthropic(api_key="fake")

    prompt = f"{anthropic.HUMAN_PROMPT} how does a court case get to the Supreme Court? {anthropic.AI_PROMPT}"

    # with using_attributes(
    #     session_id=session_id,
    #     user_id=user_id,
    #     metadata=metadata,
    #     tags=tags,
    #     prompt_template=prompt_template,
    #     prompt_template_version=prompt_template_version,
    #     prompt_template_variables=prompt_template_variables,
    # ):
    #     client.completions.create(
    #         model="claude-2.1",
    #         prompt=prompt,
    #         max_tokens_to_sample=1000,
    #     )

    client.completions.create(
        model="claude-2.1",
        prompt=prompt,
        max_tokens_to_sample=1000,
    )

    spans = in_memory_span_exporter.get_finished_spans()

    assert spans[0].name == "Completions"
    attributes = dict(spans[0].attributes or {})

    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == "LLM"
    assert isinstance(attributes.pop(INPUT_VALUE), str)
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    assert isinstance(attributes.pop(OUTPUT_VALUE), str)
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON

    assert attributes.pop(LLM_PROMPTS) == (prompt,)
    assert attributes.pop(LLM_MODEL_NAME) == "claude-2.1"
    assert not attributes


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
def test_anthropic_instrumentation_messages(
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
    attributes = dict(spans[0].attributes or {})

    assert attributes.get(OPENINFERENCE_SPAN_KIND) == "LLM"
    assert attributes.get(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}") == "idk"
    assert attributes.get(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
    assert isinstance(attributes.get(LLM_TOKEN_COUNT_PROMPT), int)
    assert isinstance(attributes.get(LLM_TOKEN_COUNT_COMPLETION), int)

@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
@pytest.mark.asyncio
async def test_anthropic_instrumentation_async_completions(
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

    await client.completions.create(
        model="claude-2.1",
        prompt="How does a court case get to the Supreme Court?",
        max_tokens_to_sample=1000,
    )

    spans = in_memory_span_exporter.get_finished_spans()

    assert spans[0].name == "AsyncCompletions"
    attributes = dict(spans[0].attributes or {})

    assert attributes.get(OPENINFERENCE_SPAN_KIND) == "LLM"
    assert attributes.get(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}") == "idk"
    assert attributes.get(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
@pytest.mark.asyncio
async def test_anthropic_instrumentation_async_messages(
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

    spans = in_memory_span_exporter.get_finished_spans()

    assert spans[0].name == "AsyncMessages"
    attributes = dict(spans[0].attributes or {})

    assert attributes.get(OPENINFERENCE_SPAN_KIND) == "LLM"
    assert attributes.get(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}") == "idk"
    assert attributes.get(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
    assert isinstance(attributes.get(LLM_TOKEN_COUNT_PROMPT), int)
    assert isinstance(attributes.get(LLM_TOKEN_COUNT_COMPLETION), int)


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


CHAIN = OpenInferenceSpanKindValues.CHAIN
LLM = OpenInferenceSpanKindValues.LLM
RETRIEVER = OpenInferenceSpanKindValues.RETRIEVER

JSON = OpenInferenceMimeTypeValues.JSON.value
TEXT = OpenInferenceMimeTypeValues.TEXT.value

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
LLM_PROMPT_TEMPLATE_VERSION = SpanAttributes.LLM_PROMPT_TEMPLATE_VERSION
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
SESSION_ID = SpanAttributes.SESSION_ID
TAG_TAGS = SpanAttributes.TAG_TAGS
TOOL_CALL_FUNCTION_ARGUMENTS_JSON = ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON
TOOL_CALL_FUNCTION_NAME = ToolCallAttributes.TOOL_CALL_FUNCTION_NAME
LLM_PROMPT_TEMPLATE = SpanAttributes.LLM_PROMPT_TEMPLATE
LLM_PROMPT_TEMPLATE_VARIABLES = SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES
USER_ID = SpanAttributes.USER_ID

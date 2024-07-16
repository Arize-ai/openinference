import asyncio
import base64
import contextlib
import json
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Generator,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
    TypeVar,
    cast,
)

import grpc
import pytest
from google.auth.api_key import Credentials
from google.cloud.aiplatform_v1 import (
    Candidate,
    Content,
    FunctionCall,
    FunctionDeclaration,
    FunctionResponse,
    GenerateContentRequest,
    GenerateContentResponse,
    PredictionServiceAsyncClient,
    PredictionServiceClient,
    Schema,
    Tool,
    Type,
)
from google.cloud.aiplatform_v1.services.prediction_service.transports import (
    PredictionServiceGrpcAsyncIOTransport,
    PredictionServiceGrpcTransport,
)
from openinference.instrumentation import using_attributes
from openinference.semconv.trace import (
    EmbeddingAttributes,
    ImageAttributes,
    MessageAttributes,
    MessageContentAttributes,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolCallAttributes,
)
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Tracer, TracerProvider
from opentelemetry.util.types import AttributeValue


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("is_stream", [False, True])
@pytest.mark.parametrize("has_error", [False, True])
async def test_instrumentor(
    is_async: bool,
    is_stream: bool,
    has_error: bool,
    mock_img: Tuple[bytes, str, str],
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    questions: List[str],
    system_instruction_text: str,
    model: str,
    tool: Tool,
    max_output_tokens: int,
    candidates: List[Candidate],
    usage_metadata: Dict[str, Any],
    mock_generate_content_response: GenerateContentResponse,
    mock_generate_content_response_gen: Iterator[GenerateContentResponse],
    mock_generate_content_response_gen_with_error: Iterator[GenerateContentResponse],
    mock_async_generate_content_response_gen: AsyncIterator[GenerateContentResponse],
    mock_async_generate_content_response_gen_with_error: AsyncIterator[GenerateContentResponse],
) -> None:
    img_bytes, img_mime_type, img_base64 = mock_img
    inline_data = dict(mime_type=img_mime_type, data=img_bytes)
    contents = [
        Content(
            dict(
                role="model" if i % 2 else "user",
                parts=[dict(text=text), dict(inline_data=inline_data)],
            )
        )
        for i, text in enumerate(questions)
    ] + [
        Content(
            dict(
                role="user",
                parts=[
                    dict(
                        function_response=FunctionResponse.from_json(
                            json.dumps(dict(name="xyz", response=dict(a=1, b=2, c=3)))
                        )
                    )
                ],
            )
        )
    ]
    system_instruction = dict(role="user", parts=[dict(text=system_instruction_text)])
    generation_config = dict(max_output_tokens=max_output_tokens)
    request = GenerateContentRequest(
        dict(
            contents=contents,
            system_instruction=system_instruction,
            model=model,
            generation_config=generation_config,
            tools=[tool],
        )
    )
    credentials = Credentials("123")  # type: ignore[no-untyped-call]
    tracer = tracer_provider.get_tracer(__name__)
    with contextlib.ExitStack() as stack:
        stack.enter_context(contextlib.suppress(Error))
        stack.enter_context(using_attributes(metadata={"a": [{"b": "c"}]}))
        if is_async:
            if is_stream:
                stack.enter_context(
                    patch_grpc_asyncio_transport_stream_generate_content(
                        MockAsyncStreamGenerateContent(
                            mock_async_generate_content_response_gen_with_error
                            if has_error
                            else mock_async_generate_content_response_gen,
                            tracer,
                        )
                    )
                )
                async_client = PredictionServiceAsyncClient(credentials=credentials)
                stream_async_func = async_client.stream_generate_content
                _ = [_ async for _ in await stream_async_func(request)]
            else:
                stack.enter_context(
                    patch_grpc_asyncio_transport_generate_content(
                        MockAsyncGenerateContentWithError(tracer)
                        if has_error
                        else MockAsyncGenerateContent(mock_generate_content_response, tracer)
                    )
                )
                async_client = PredictionServiceAsyncClient(credentials=credentials)
                await async_client.generate_content(request)
        else:
            if is_stream:
                stack.enter_context(
                    patch_grpc_transport_stream_generate_content(
                        MockStreamGenerateContent(
                            mock_generate_content_response_gen_with_error
                            if has_error
                            else mock_generate_content_response_gen,
                            tracer,
                        )
                    )
                )
                client = PredictionServiceClient(credentials=credentials)
                _ = [_ for _ in client.stream_generate_content(request)]
            else:
                stack.enter_context(
                    patch_grpc_transport_generate_content(
                        MockGenerateContentWithError(tracer)
                        if has_error
                        else MockGenerateContent(mock_generate_content_response, tracer)
                    )
                )
                client = PredictionServiceClient(credentials=credentials)
                client.generate_content(request)

    spans = sorted(
        in_memory_span_exporter.get_finished_spans(),
        key=lambda _: cast(int, _.start_time),
    )
    assert spans
    span = spans[0]
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))
    assert attributes.pop(METADATA, None) == '{"a": [{"b": "c"}]}'
    assert attributes.pop(OPENINFERENCE_SPAN_KIND, None) == OpenInferenceSpanKindValues.LLM.value
    assert isinstance(attributes.pop(INPUT_VALUE, None), str)
    assert attributes.pop(INPUT_MIME_TYPE, None) == JSON
    assert (invocation_parameters := cast(str, attributes.pop(LLM_INVOCATION_PARAMETERS, None)))
    assert json.loads(invocation_parameters).get("max_output_tokens") == max_output_tokens
    prefix = LLM_INPUT_MESSAGES
    assert attributes.pop(message_role(prefix, 0), None) == "system"
    assert attributes.pop(message_contents_text(prefix, 0, 0), None) == system_instruction_text
    for i, question in enumerate(questions, 1):
        assert attributes.pop(message_role(prefix, i), None) == ("user" if i % 2 else "assistant")
        assert attributes.pop(message_contents_text(prefix, i, 0), None) == question
        assert attributes.pop(message_contents_image_url(prefix, i, 1), None) == img_base64
    function_response = contents[-1].parts[0].function_response
    assert attributes.pop(message_role(prefix, len(contents)), None) == "tool"
    assert attributes.pop(message_name(prefix, len(contents)), None) == function_response.name
    response_json_str = attributes.pop(message_content(prefix, len(contents)), None)
    assert isinstance(response_json_str, str)
    assert json.loads(response_json_str) == FunctionResponse.to_dict(function_response)["response"]
    assert attributes.pop(LLM_MODEL_NAME, None) == model
    status = span.status
    if has_error:
        assert not status.is_ok
        assert not status.is_unset
        assert status.description
        assert status.description.endswith(ERR_MSG)
        assert span.events
        event = span.events[0]
        assert event.attributes
        assert event.attributes.get("exception.message") == ERR_MSG
        assert attributes == {}
        return
    assert status.is_ok
    assert not status.is_unset
    assert not status.description
    assert isinstance((output_value := attributes.pop(OUTPUT_VALUE, None)), str)
    assert attributes.pop(OUTPUT_MIME_TYPE, None) == JSON
    assert json.loads(output_value)
    prefix = LLM_OUTPUT_MESSAGES
    for i, candidate in enumerate(candidates):
        assert attributes.pop(message_role(prefix, i), None) == "assistant"
        for j, part in enumerate(candidate.content.parts):
            if part.text:
                assert attributes.pop(message_contents_text(prefix, i, j), None) == part.text
            elif part.function_call.name:
                assert (
                    attributes.pop(tool_call_function_name(prefix, i, j), None)
                    == part.function_call.name
                )
                args_json_str = attributes.pop(tool_call_function_arguments(prefix, i, j), None)
                assert isinstance(args_json_str, str)
                assert json.loads(args_json_str) == FunctionCall.to_dict(part.function_call)["args"]
    assert attributes.pop(LLM_TOKEN_COUNT_TOTAL, None) == usage_metadata["total_token_count"]
    assert attributes.pop(LLM_TOKEN_COUNT_PROMPT, None) == usage_metadata["prompt_token_count"]
    assert (
        attributes.pop(LLM_TOKEN_COUNT_COMPLETION, None) == usage_metadata["candidates_token_count"]
    )
    assert attributes == {}

    assert len(spans) > 1
    spans_by_id = {}
    for span in spans:
        spans_by_id[span.context.span_id] = span
    for span in spans[1:]:
        assert is_descendant(span, spans[0], spans_by_id)


@pytest.fixture
def tool() -> Tool:
    location = Schema(
        dict(type_=Type.STRING, description="The city and state, e.g. San Francisco, CA")
    )
    unit = Schema(dict(type_=Type.STRING, enum=["celsius", "fahrenheit"]))
    properties = dict(location=location, unit=unit)
    parameters = Schema(dict(type_=Type.OBJECT, properties=properties, required=["location"]))
    name = "get_current_weather"
    description = "Get the current weather in a given location"
    function_declaration = FunctionDeclaration(
        dict(name=name, description=description, parameters=parameters)
    )
    return Tool(dict(function_declarations=[function_declaration]))


@pytest.fixture
def questions() -> List[str]:
    return list("abc")


@pytest.fixture
def system_instruction_text() -> str:
    return "12345"


@pytest.fixture
def model() -> str:
    return "xyz"


@pytest.fixture
def max_output_tokens() -> int:
    return 123


@pytest.fixture
def usage_metadata() -> Dict[str, int]:
    return dict(
        prompt_token_count=11,
        candidates_token_count=22,
        total_token_count=33,
    )


@pytest.fixture
def candidates() -> List[Candidate]:
    return [
        Candidate(dict(index=0, content=dict(role="model", parts=[dict(text="1 2 3")]))),
        Candidate(dict(index=1, content=dict(role="model", parts=[dict(text="a b c")]))),
        Candidate(
            dict(
                index=2,
                content=dict(
                    role="model",
                    parts=[
                        dict(
                            function_call=FunctionCall.from_json(
                                json.dumps(dict(name="xyz", args=dict(a=1, b=2, c=3)))
                            )
                        )
                    ],
                ),
            )
        ),
    ]


@pytest.fixture
def mock_generate_content_response(
    candidates: List[Candidate],
    usage_metadata: Dict[str, int],
) -> GenerateContentResponse:
    return GenerateContentResponse(
        dict(
            candidates=candidates,
            usage_metadata=usage_metadata,
        )
    )


@pytest.fixture
def mock_generate_content_response_gen(
    mock_generate_content_response: GenerateContentResponse,
    tracer_provider: TracerProvider,
) -> Iterator[GenerateContentResponse]:
    def _() -> Iterator[GenerateContentResponse]:
        for index, candidate in reversed(
            list(enumerate(mock_generate_content_response.candidates))
        ):
            for part in candidate.content.parts:
                if part.text:
                    for t in part.text:
                        content = dict(role="model", parts=[dict(text=t)])
                        with tracer_provider.get_tracer(__name__).start_as_current_span("TEST"):
                            yield GenerateContentResponse(
                                dict(candidates=[dict(index=index, content=content)])
                            )
                else:
                    content = dict(role="model", parts=[part])
                    with tracer_provider.get_tracer(__name__).start_as_current_span("TEST"):
                        yield GenerateContentResponse(
                            dict(candidates=[dict(index=index, content=content)])
                        )
        yield GenerateContentResponse(
            dict(usage_metadata=mock_generate_content_response.usage_metadata)
        )

    return _()


@pytest.fixture
def mock_generate_content_response_gen_with_error(
    mock_generate_content_response_gen: Iterator[GenerateContentResponse],
    tracer_provider: TracerProvider,
) -> Iterator[GenerateContentResponse]:
    def _() -> Iterator[GenerateContentResponse]:
        yield from mock_generate_content_response_gen
        with tracer_provider.get_tracer(__name__).start_as_current_span("TEST"):
            raise Error(ERR_MSG)

    return _()


@pytest.fixture
def mock_async_generate_content_response_gen(
    mock_generate_content_response_gen: Iterator[GenerateContentResponse],
    tracer_provider: TracerProvider,
) -> AsyncIterator[GenerateContentResponse]:
    async def _() -> AsyncIterator[GenerateContentResponse]:
        for _ in mock_generate_content_response_gen:
            with tracer_provider.get_tracer(__name__).start_as_current_span("TEST"):
                yield _

    return _()


@pytest.fixture
def mock_async_generate_content_response_gen_with_error(
    mock_async_generate_content_response_gen: AsyncIterator[GenerateContentResponse],
    tracer_provider: TracerProvider,
) -> AsyncIterator[GenerateContentResponse]:
    async def _() -> AsyncIterator[GenerateContentResponse]:
        async for _ in mock_async_generate_content_response_gen:
            with tracer_provider.get_tracer(__name__).start_as_current_span("TEST"):
                yield _
        with tracer_provider.get_tracer(__name__).start_as_current_span("TEST"):
            raise Error(ERR_MSG)

    return _()


RequestType = TypeVar("RequestType")
ResponseType = TypeVar("ResponseType")


class HasTracer:
    def __init__(self, tracer: Tracer) -> None:
        self._tracer = tracer


class MockGenerateContentWithError(HasTracer):
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        with self._tracer.start_as_current_span("TEST"):
            raise Error(ERR_MSG)


class MockGenerateContent(HasTracer):
    def __init__(self, response: GenerateContentResponse, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._response = response

    def __call__(self, *args: Any, **kwargs: Any) -> GenerateContentResponse:
        with self._tracer.start_as_current_span("TEST"):
            return self._response


class MockStreamGenerateContent(
    HasTracer,
    grpc.UnaryStreamMultiCallable,  # type: ignore[misc]
):
    def __init__(
        self, response_gen: Iterator[GenerateContentResponse], *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self._response_gen = response_gen

    def __call__(self, *args: Any, **kwargs: Any) -> Iterator[GenerateContentResponse]:
        with self._tracer.start_as_current_span("TEST"):
            return self._response_gen


class MockAsyncGenerateContentWithError(
    HasTracer,
    grpc.aio.UnaryUnaryMultiCallable[RequestType, ResponseType],  # type: ignore[misc]
):
    def __call__(self, request: RequestType, **kwargs: Any) -> Any:
        with self._tracer.start_as_current_span("TEST"):
            raise Error(ERR_MSG)


class MockAsyncGenerateContent(
    HasTracer,
    grpc.aio.UnaryUnaryMultiCallable[RequestType, ResponseType],  # type: ignore[misc]
):
    def __init__(self, response: GenerateContentResponse, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._response = response

    def __call__(
        self,
        request: RequestType,
        **kwargs: Any,
    ) -> grpc.aio.UnaryUnaryCall[RequestType, ResponseType]:
        with self._tracer.start_as_current_span("TEST"):
            return MockUnaryUnaryCall(self._response, self._tracer)


class MockAsyncStreamGenerateContent(
    HasTracer,
    grpc.aio.UnaryStreamMultiCallable[RequestType, ResponseType],  # type: ignore[misc]
):
    def __init__(
        self, response_gen: AsyncIterator[ResponseType], *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self._response_gen = response_gen

    def __call__(
        self, request: RequestType, **kwargs: Any
    ) -> grpc.aio.UnaryStreamCall[RequestType, ResponseType]:
        with self._tracer.start_as_current_span("TEST"):
            return MockUnaryStreamCall(self._response_gen, self._tracer)


class MockAsyncRpcContext(grpc.aio.RpcContext):  # type: ignore[misc]
    def cancelled(self) -> Any: ...

    def done(self) -> Any: ...

    def time_remaining(self) -> Any: ...

    def cancel(self) -> Any: ...

    def add_done_callback(self, callback: Callable[[Any], None]) -> Any: ...


class MockAsyncCall(MockAsyncRpcContext, grpc.aio.Call):  # type: ignore[misc]
    async def initial_metadata(self) -> Any: ...

    async def trailing_metadata(self) -> Any: ...

    async def code(self) -> Any: ...

    async def details(self) -> Any: ...

    async def wait_for_connection(self) -> Any: ...

    async def cancelled(self) -> Any: ...

    async def done(self) -> Any: ...


class MockUnaryUnaryCall(
    HasTracer,
    MockAsyncCall,
    grpc.aio.UnaryUnaryCall[RequestType, ResponseType],  # type: ignore[misc]
):
    def __init__(self, response: ResponseType, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._response = response

    def __await__(self) -> Generator[Any, None, ResponseType]:
        with self._tracer.start_as_current_span("TEST"):
            yield from asyncio.sleep(0).__await__()
        with self._tracer.start_as_current_span("TEST"):
            return self._response


class MockUnaryStreamCall(
    HasTracer,
    MockAsyncCall,
    grpc.aio.UnaryStreamCall[RequestType, ResponseType],  # type: ignore[misc]
):
    def __init__(
        self, response_gen: AsyncIterator[ResponseType], *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self._response_gen = response_gen

    def __aiter__(self) -> AsyncIterator[ResponseType]:
        with self._tracer.start_as_current_span("TEST"):
            return self._response_gen

    async def read(self) -> Any: ...


@contextlib.contextmanager
def patch_grpc_transport_generate_content(replacement: Any) -> Iterator[None]:
    cls = PredictionServiceGrpcTransport
    original = cls.generate_content
    setattr(cls, "generate_content", property(lambda _: replacement))
    yield
    setattr(cls, "generate_content", original)


@contextlib.contextmanager
def patch_grpc_transport_stream_generate_content(replacement: Any) -> Iterator[None]:
    cls = PredictionServiceGrpcTransport
    original = cls.stream_generate_content
    setattr(cls, "stream_generate_content", property(lambda _: replacement))
    yield
    setattr(cls, "stream_generate_content", original)


@contextlib.contextmanager
def patch_grpc_asyncio_transport_generate_content(replacement: Any) -> Iterator[None]:
    cls = PredictionServiceGrpcAsyncIOTransport
    original = cls.generate_content
    setattr(cls, "generate_content", property(lambda _: replacement))
    yield
    setattr(cls, "generate_content", original)


@contextlib.contextmanager
def patch_grpc_asyncio_transport_stream_generate_content(replacement: Any) -> Iterator[None]:
    cls = PredictionServiceGrpcAsyncIOTransport
    original = cls.stream_generate_content
    setattr(cls, "stream_generate_content", property(lambda _: replacement))
    yield
    setattr(cls, "stream_generate_content", original)


@pytest.fixture
def mock_img() -> Tuple[bytes, str, str]:
    b64 = "R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw=="
    mime_type = "image/gif"
    return base64.b64decode(b64.encode()), mime_type, f"data:{mime_type};base64,{b64}"


def is_descendant(
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


def message_name(prefix: str, i: int) -> str:
    return f"{prefix}.{i}.{MESSAGE_NAME}"


def message_role(prefix: str, i: int) -> str:
    return f"{prefix}.{i}.{MESSAGE_ROLE}"


def message_content(prefix: str, i: int) -> str:
    return f"{prefix}.{i}.{MESSAGE_CONTENT}"


def message_contents_type(prefix: str, i: int, j: int) -> str:
    return f"{prefix}.{i}.{MESSAGE_CONTENTS}.{j}.{MESSAGE_CONTENT_TYPE}"


def message_contents_text(prefix: str, i: int, j: int) -> str:
    return f"{prefix}.{i}.{MESSAGE_CONTENTS}.{j}.{MESSAGE_CONTENT_TEXT}"


def message_contents_image_url(prefix: str, i: int, j: int) -> str:
    return f"{prefix}.{i}.{MESSAGE_CONTENTS}.{j}.{MESSAGE_CONTENT_IMAGE}.{IMAGE_URL}"


def message_function_call_name(prefix: str, i: int) -> str:
    return f"{prefix}.{i}.{MESSAGE_FUNCTION_CALL_NAME}"


def message_function_call_arguments(prefix: str, i: int) -> str:
    return f"{prefix}.{i}.{MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON}"


def tool_call_function_name(prefix: str, i: int, j: int) -> str:
    return f"{prefix}.{i}.{MESSAGE_TOOL_CALLS}.{j}.{TOOL_CALL_FUNCTION_NAME}"


def tool_call_function_arguments(prefix: str, i: int, j: int) -> str:
    return f"{prefix}.{i}.{MESSAGE_TOOL_CALLS}.{j}.{TOOL_CALL_FUNCTION_ARGUMENTS_JSON}"


ERR_MSG = "ERR_MSG"


class Error(RuntimeError): ...


EMBEDDING_EMBEDDINGS = SpanAttributes.EMBEDDING_EMBEDDINGS
EMBEDDING_MODEL_NAME = SpanAttributes.EMBEDDING_MODEL_NAME
EMBEDDING_TEXT = EmbeddingAttributes.EMBEDDING_TEXT
EMBEDDING_VECTOR = EmbeddingAttributes.EMBEDDING_VECTOR
IMAGE_URL = ImageAttributes.IMAGE_URL
INPUT_MIME_TYPE = SpanAttributes.INPUT_MIME_TYPE
INPUT_VALUE = SpanAttributes.INPUT_VALUE
JSON = OpenInferenceMimeTypeValues.JSON.value
LLM_INPUT_MESSAGES = SpanAttributes.LLM_INPUT_MESSAGES
LLM_INVOCATION_PARAMETERS = SpanAttributes.LLM_INVOCATION_PARAMETERS
LLM_MODEL_NAME = SpanAttributes.LLM_MODEL_NAME
LLM_OUTPUT_MESSAGES = SpanAttributes.LLM_OUTPUT_MESSAGES
LLM_PROMPTS = SpanAttributes.LLM_PROMPTS
LLM_TOKEN_COUNT_COMPLETION = SpanAttributes.LLM_TOKEN_COUNT_COMPLETION
LLM_TOKEN_COUNT_PROMPT = SpanAttributes.LLM_TOKEN_COUNT_PROMPT
LLM_TOKEN_COUNT_TOTAL = SpanAttributes.LLM_TOKEN_COUNT_TOTAL
MESSAGE_CONTENT = MessageAttributes.MESSAGE_CONTENT
MESSAGE_CONTENTS = MessageAttributes.MESSAGE_CONTENTS
MESSAGE_CONTENT_IMAGE = MessageContentAttributes.MESSAGE_CONTENT_IMAGE
MESSAGE_CONTENT_TEXT = MessageContentAttributes.MESSAGE_CONTENT_TEXT
MESSAGE_CONTENT_TYPE = MessageContentAttributes.MESSAGE_CONTENT_TYPE
MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON = MessageAttributes.MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON
MESSAGE_FUNCTION_CALL_NAME = MessageAttributes.MESSAGE_FUNCTION_CALL_NAME
MESSAGE_NAME = MessageAttributes.MESSAGE_NAME
MESSAGE_ROLE = MessageAttributes.MESSAGE_ROLE
MESSAGE_TOOL_CALLS = MessageAttributes.MESSAGE_TOOL_CALLS
METADATA = SpanAttributes.METADATA
OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
OUTPUT_MIME_TYPE = SpanAttributes.OUTPUT_MIME_TYPE
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
SESSION_ID = SpanAttributes.SESSION_ID
TAG_TAGS = SpanAttributes.TAG_TAGS
TOOL_CALL_FUNCTION_ARGUMENTS_JSON = ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON
TOOL_CALL_FUNCTION_NAME = ToolCallAttributes.TOOL_CALL_FUNCTION_NAME
USER_ID = SpanAttributes.USER_ID

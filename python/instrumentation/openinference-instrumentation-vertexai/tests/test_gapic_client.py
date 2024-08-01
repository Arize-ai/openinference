import asyncio
import base64
import json
from contextlib import ExitStack, contextmanager, suppress
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
from openinference.instrumentation import REDACTED_VALUE, TraceConfig, using_attributes
from openinference.instrumentation.vertexai import VertexAIInstrumentor
from openinference.instrumentation.vertexai._wrapper import _role
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
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import ReadableSpan, Tracer
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.util.types import AttributeValue


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("is_stream", [False, True])
@pytest.mark.parametrize("has_error", [False, True])
async def test_instrumentor(
    is_async: bool,
    is_stream: bool,
    has_error: bool,
    in_memory_span_exporter: InMemorySpanExporter,
    mock_generate_content_request: GenerateContentRequest,
    mock_generate_content_response: GenerateContentResponse,
    mock_img: Tuple[bytes, str, str],
    metadata: Dict[str, Any],
    tracer: Tracer,
) -> None:
    request = mock_generate_content_request
    response = mock_generate_content_response
    with ExitStack() as stack:
        stack.enter_context(suppress(Err))
        stack.enter_context(using_attributes(metadata=metadata))
        args = (request, response, has_error, stack, tracer)
        if is_async:
            if is_stream:
                await mock_async_stream_generate_content(*args)
            else:
                await mock_async_generate_content(*args)
        else:
            if is_stream:
                mock_stream_generate_content(*args)
            else:
                mock_generate_content(*args)
    spans = sorted(
        in_memory_span_exporter.get_finished_spans(),
        key=lambda _: cast(int, _.start_time),
    )
    assert spans
    span = spans[0]
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))
    assert isinstance(metadata_json_str := attributes.pop(METADATA, None), str)
    assert json.loads(metadata_json_str) == metadata
    assert attributes.pop(OPENINFERENCE_SPAN_KIND, None) == OpenInferenceSpanKindValues.LLM.value
    assert isinstance(attributes.pop(INPUT_VALUE, None), str)
    assert attributes.pop(INPUT_MIME_TYPE, None) == JSON
    assert (invocation_parameters := cast(str, attributes.pop(LLM_INVOCATION_PARAMETERS, None)))
    assert (
        json.loads(invocation_parameters).get("max_output_tokens")
        == request.generation_config.max_output_tokens
    )
    prefix = LLM_INPUT_MESSAGES
    _, _, img_base64 = mock_img
    assert attributes.pop(message_role(prefix, 0), None) == "system"
    assert (
        attributes.pop(message_contents_text(prefix, 0, 0), None)
        == request.system_instruction.parts[0].text
    )
    contents = request.contents
    for i, content in enumerate(contents[:-1], 1):
        assert attributes.pop(message_role(prefix, i), None) == _role(content.role)
        assert attributes.pop(message_contents_text(prefix, i, 0), None) == content.parts[0].text
        assert attributes.pop(message_contents_image_url(prefix, i, 1), None) == img_base64
    function_response = contents[-1].parts[0].function_response
    assert attributes.pop(message_role(prefix, len(contents)), None) == "tool"
    assert attributes.pop(message_name(prefix, len(contents)), None) == function_response.name
    response_json_str = attributes.pop(message_content(prefix, len(contents)), None)
    assert isinstance(response_json_str, str)
    assert json.loads(response_json_str) == FunctionResponse.to_dict(function_response)["response"]
    assert attributes.pop(LLM_MODEL_NAME, None) == request.model
    status = span.status
    if has_error:
        assert not status.is_ok
        assert not status.is_unset
        assert status.description
        assert status.description.endswith(MSG)
        assert span.events
        event = span.events[0]
        assert event.attributes
        assert event.attributes.get("exception.message") == MSG
        assert attributes == {}
        return
    assert status.is_ok
    assert not status.is_unset
    assert not status.description
    assert isinstance((output_value := attributes.pop(OUTPUT_VALUE, None)), str)
    assert attributes.pop(OUTPUT_MIME_TYPE, None) == JSON
    assert json.loads(output_value)
    prefix = LLM_OUTPUT_MESSAGES
    candidates = response.candidates
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
    usage_metadata = response.usage_metadata
    assert attributes.pop(LLM_TOKEN_COUNT_TOTAL, None) == usage_metadata.total_token_count
    assert attributes.pop(LLM_TOKEN_COUNT_PROMPT, None) == usage_metadata.prompt_token_count
    assert attributes.pop(LLM_TOKEN_COUNT_COMPLETION, None) == usage_metadata.candidates_token_count
    assert attributes == {}

    assert len(spans) > 1
    spans_by_id = {}
    for span in spans:
        spans_by_id[span.context.span_id] = span
    for span in spans[1:]:
        assert is_descendant(span, spans[0], spans_by_id)


@pytest.mark.parametrize("hide_inputs", [False, True])
@pytest.mark.parametrize("hide_input_messages", [False, True])
@pytest.mark.parametrize("hide_input_images", [False, True])
@pytest.mark.parametrize("hide_input_text", [False, True])
@pytest.mark.parametrize("base64_image_max_length", [0, 100_000])
async def test_instrumentor_config_hiding_inputs(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    mock_generate_content_request: GenerateContentRequest,
    mock_generate_content_response: GenerateContentResponse,
    mock_img: Tuple[bytes, str, str],
    metadata: Dict[str, Any],
    tracer: Tracer,
    hide_inputs: bool,
    hide_input_messages: bool,
    hide_input_images: bool,
    hide_input_text: bool,
    base64_image_max_length: int,
) -> None:
    VertexAIInstrumentor().uninstrument()
    config = TraceConfig(
        hide_inputs=hide_inputs,
        hide_input_messages=hide_input_messages,
        hide_input_images=hide_input_images,
        hide_input_text=hide_input_text,
        base64_image_max_length=base64_image_max_length,
    )
    assert config.hide_inputs is hide_inputs
    assert config.hide_input_messages is hide_input_messages
    assert config.hide_input_images is hide_input_images
    assert config.hide_input_text is hide_input_text
    assert config.base64_image_max_length is base64_image_max_length
    VertexAIInstrumentor().instrument(tracer_provider=tracer_provider, config=config)
    request = mock_generate_content_request
    response = mock_generate_content_response
    with ExitStack() as stack:
        stack.enter_context(suppress(Err))
        stack.enter_context(using_attributes(metadata=metadata))
        args = (request, response, False, stack, tracer)
        mock_generate_content(*args)
    spans = sorted(
        in_memory_span_exporter.get_finished_spans(),
        key=lambda _: cast(int, _.start_time),
    )
    assert spans
    span = spans[0]
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))
    assert isinstance(metadata_json_str := attributes.pop(METADATA, None), str)
    assert json.loads(metadata_json_str) == metadata
    assert attributes.pop(OPENINFERENCE_SPAN_KIND, None) == OpenInferenceSpanKindValues.LLM.value
    assert attributes.pop(LLM_MODEL_NAME, None) == request.model
    assert cast(str, attributes.pop(LLM_INVOCATION_PARAMETERS, None))
    usage_metadata = response.usage_metadata
    assert attributes.pop(LLM_TOKEN_COUNT_TOTAL, None) == usage_metadata.total_token_count
    assert attributes.pop(LLM_TOKEN_COUNT_PROMPT, None) == usage_metadata.prompt_token_count
    assert attributes.pop(LLM_TOKEN_COUNT_COMPLETION, None) == usage_metadata.candidates_token_count
    # Input value
    input_value = attributes.pop(INPUT_VALUE, None)
    assert input_value is not None
    if hide_inputs:
        assert input_value == REDACTED_VALUE
    else:
        assert isinstance(input_value, str)
        assert attributes.pop(INPUT_MIME_TYPE, None) == JSON
    # Input messages
    if not hide_inputs and not hide_input_messages:
        prefix = LLM_INPUT_MESSAGES
        _, _, img_base64 = mock_img
        assert attributes.pop(message_role(prefix, 0), None) == "system"
        msg_contents_text = attributes.pop(message_contents_text(prefix, 0, 0), None)
        expected_contents_text = (
            REDACTED_VALUE if hide_input_text else request.system_instruction.parts[0].text
        )
        assert msg_contents_text == expected_contents_text

        contents = request.contents
        for i, content in enumerate(contents[:-1], 1):
            assert attributes.pop(message_role(prefix, i), None) == _role(content.role)
            msg_contents_text = attributes.pop(message_contents_text(prefix, i, 0), None)
            expected_contents_text = REDACTED_VALUE if hide_input_text else content.parts[0].text
            assert msg_contents_text == expected_contents_text
            if not hide_input_images:
                expected_img_url = (
                    REDACTED_VALUE if len(img_base64) > base64_image_max_length else img_base64
                )
                img_url = attributes.pop(message_contents_image_url(prefix, i, 1), None)
                assert img_url == expected_img_url

        function_response = contents[-1].parts[0].function_response
        assert attributes.pop(message_role(prefix, len(contents)), None) == "tool"
        assert attributes.pop(message_name(prefix, len(contents)), None) == function_response.name
        response_json_str = attributes.pop(message_content(prefix, len(contents)), None)
        assert isinstance(response_json_str, str)
        if hide_input_text:
            assert response_json_str == REDACTED_VALUE
        else:
            assert (
                json.loads(response_json_str)
                == FunctionResponse.to_dict(function_response)["response"]
            )

    # Output value
    assert isinstance((output_value := attributes.pop(OUTPUT_VALUE, None)), str)
    assert attributes.pop(OUTPUT_MIME_TYPE, None) == JSON
    assert json.loads(output_value)
    # Output messages
    prefix = LLM_OUTPUT_MESSAGES
    candidates = response.candidates
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
    assert attributes == {}

    assert len(spans) > 1
    spans_by_id = {}
    for span in spans:
        spans_by_id[span.context.span_id] = span
    for span in spans[1:]:
        assert is_descendant(span, spans[0], spans_by_id)


@pytest.mark.parametrize("hide_outputs", [False, True])
@pytest.mark.parametrize("hide_output_messages", [False, True])
@pytest.mark.parametrize("hide_output_text", [False, True])
async def test_instrumentor_config_hiding_outputs(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    mock_generate_content_request: GenerateContentRequest,
    mock_generate_content_response: GenerateContentResponse,
    mock_img: Tuple[bytes, str, str],
    metadata: Dict[str, Any],
    tracer: Tracer,
    hide_outputs: bool,
    hide_output_messages: bool,
    hide_output_text: bool,
) -> None:
    VertexAIInstrumentor().uninstrument()
    config = TraceConfig(
        hide_outputs=hide_outputs,
        hide_output_messages=hide_output_messages,
        hide_output_text=hide_output_text,
    )
    assert config.hide_outputs is hide_outputs
    assert config.hide_output_messages is hide_output_messages
    assert config.hide_output_text is hide_output_text
    VertexAIInstrumentor().instrument(tracer_provider=tracer_provider, config=config)
    request = mock_generate_content_request
    response = mock_generate_content_response
    with ExitStack() as stack:
        stack.enter_context(suppress(Err))
        stack.enter_context(using_attributes(metadata=metadata))
        args = (request, response, False, stack, tracer)
        mock_generate_content(*args)
    spans = sorted(
        in_memory_span_exporter.get_finished_spans(),
        key=lambda _: cast(int, _.start_time),
    )
    assert spans
    span = spans[0]
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))
    assert isinstance(metadata_json_str := attributes.pop(METADATA, None), str)
    assert json.loads(metadata_json_str) == metadata
    assert attributes.pop(OPENINFERENCE_SPAN_KIND, None) == OpenInferenceSpanKindValues.LLM.value
    assert attributes.pop(LLM_MODEL_NAME, None) == request.model
    assert cast(str, attributes.pop(LLM_INVOCATION_PARAMETERS, None))
    usage_metadata = response.usage_metadata
    assert attributes.pop(LLM_TOKEN_COUNT_TOTAL, None) == usage_metadata.total_token_count
    assert attributes.pop(LLM_TOKEN_COUNT_PROMPT, None) == usage_metadata.prompt_token_count
    assert attributes.pop(LLM_TOKEN_COUNT_COMPLETION, None) == usage_metadata.candidates_token_count
    # Input value
    input_value = attributes.pop(INPUT_VALUE, None)
    assert input_value is not None
    assert isinstance(input_value, str)
    assert attributes.pop(INPUT_MIME_TYPE, None) == JSON
    # Input messages
    prefix = LLM_INPUT_MESSAGES
    _, _, img_base64 = mock_img
    assert attributes.pop(message_role(prefix, 0), None) == "system"
    msg_contents_text = attributes.pop(message_contents_text(prefix, 0, 0), None)
    expected_contents_text = request.system_instruction.parts[0].text
    assert msg_contents_text == expected_contents_text

    contents = request.contents
    for i, content in enumerate(contents[:-1], 1):
        assert attributes.pop(message_role(prefix, i), None) == _role(content.role)
        msg_contents_text = attributes.pop(message_contents_text(prefix, i, 0), None)
        expected_contents_text = content.parts[0].text
        assert msg_contents_text == expected_contents_text
        img_url = attributes.pop(message_contents_image_url(prefix, i, 1), None)
        assert img_url == img_base64

    function_response = contents[-1].parts[0].function_response
    assert attributes.pop(message_role(prefix, len(contents)), None) == "tool"
    assert attributes.pop(message_name(prefix, len(contents)), None) == function_response.name
    response_json_str = attributes.pop(message_content(prefix, len(contents)), None)
    assert isinstance(response_json_str, str)
    assert json.loads(response_json_str) == FunctionResponse.to_dict(function_response)["response"]

    # Output value
    output_value = attributes.pop(OUTPUT_VALUE, None)
    assert output_value is not None
    assert isinstance(output_value, str)
    if hide_outputs:
        assert output_value == REDACTED_VALUE
    else:
        assert attributes.pop(OUTPUT_MIME_TYPE, None) == JSON
        assert json.loads(output_value)
    # Output messages
    if not hide_outputs and not hide_output_messages:
        prefix = LLM_OUTPUT_MESSAGES
        candidates = response.candidates
        for i, candidate in enumerate(candidates):
            assert attributes.pop(message_role(prefix, i), None) == "assistant"
            for j, part in enumerate(candidate.content.parts):
                if part.text:
                    expected = REDACTED_VALUE if hide_output_text else part.text
                    assert attributes.pop(message_contents_text(prefix, i, j), None) == expected
                elif part.function_call.name:
                    expected_name = part.function_call.name
                    assert (
                        attributes.pop(tool_call_function_name(prefix, i, j), None) == expected_name
                    )
                    args_json_str = attributes.pop(tool_call_function_arguments(prefix, i, j), None)
                    assert isinstance(args_json_str, str)
                    assert (
                        json.loads(args_json_str)
                        == FunctionCall.to_dict(part.function_call)["args"]
                    )
    assert attributes == {}

    assert len(spans) > 1
    spans_by_id = {}
    for span in spans:
        spans_by_id[span.context.span_id] = span
    for span in spans[1:]:
        assert is_descendant(span, spans[0], spans_by_id)


async def mock_async_stream_generate_content(
    request: GenerateContentRequest,
    response: GenerateContentResponse,
    has_error: bool,
    stack: ExitStack,
    tracer: Tracer,
) -> None:
    gen = _may_err(has_error, tracer, mock_generate_content_response_gen(response))
    response_gen = mock_async_generate_content_response_gen(gen)
    patch = MockAsyncStreamGenerateContent(response_gen, tracer)
    stack.enter_context(patch_grpc_asyncio_transport_stream_generate_content(patch))
    client = PredictionServiceAsyncClient(credentials=CREDENTIALS)
    _ = [_ async for _ in await client.stream_generate_content(request)]


def mock_stream_generate_content(
    request: GenerateContentRequest,
    response: GenerateContentResponse,
    has_error: bool,
    stack: ExitStack,
    tracer: Tracer,
) -> None:
    response_gen = _may_err(has_error, tracer, mock_generate_content_response_gen(response))
    patch = MockStreamGenerateContent(response_gen)
    stack.enter_context(patch_grpc_transport_stream_generate_content(patch))
    client = PredictionServiceClient(credentials=CREDENTIALS)
    _ = [_ for _ in client.stream_generate_content(request)]


async def mock_async_generate_content(
    request: GenerateContentRequest,
    response: GenerateContentResponse,
    has_error: bool,
    stack: ExitStack,
    tracer: Tracer,
) -> None:
    patch = _may_err(has_error, tracer, MockAsyncGenerateContent(response, tracer))
    stack.enter_context(patch_grpc_asyncio_transport_generate_content(patch))
    client = PredictionServiceAsyncClient(credentials=CREDENTIALS)
    await client.generate_content(request)


def mock_generate_content(
    request: GenerateContentRequest,
    response: GenerateContentResponse,
    has_error: bool,
    stack: ExitStack,
    tracer: Tracer,
) -> None:
    patch = _may_err(has_error, tracer, lambda *_, **__: response)
    stack.enter_context(patch_grpc_transport_generate_content(patch))
    client = PredictionServiceClient(credentials=CREDENTIALS)
    client.generate_content(request)


@pytest.fixture
def metadata() -> Dict[str, Any]:
    return {"1": [{"2": 3}]}


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
def mock_generate_content_request(
    mock_img: Tuple[bytes, str, str],
    questions: List[str],
    system_instruction_text: str,
    model: str,
    tool: Tool,
    max_output_tokens: int,
    candidates: List[Candidate],
    usage_metadata: Dict[str, Any],
) -> GenerateContentRequest:
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
    return GenerateContentRequest(
        dict(
            contents=contents,
            system_instruction=system_instruction,
            model=model,
            generation_config=generation_config,
            tools=[tool],
        )
    )


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


def mock_generate_content_response_gen(
    response: GenerateContentResponse,
) -> Iterator[GenerateContentResponse]:
    for index, candidate in reversed(list(enumerate(response.candidates))):
        for part in candidate.content.parts:
            if part.text:
                for t in part.text:
                    content = dict(role="model", parts=[dict(text=t)])
                    yield GenerateContentResponse(
                        dict(candidates=[dict(index=index, content=content)])
                    )
            else:
                content = dict(role="model", parts=[part])
                yield GenerateContentResponse(dict(candidates=[dict(index=index, content=content)]))
    yield GenerateContentResponse(dict(usage_metadata=response.usage_metadata))


async def mock_async_generate_content_response_gen(
    response_gen: Iterator[GenerateContentResponse],
) -> AsyncIterator[GenerateContentResponse]:
    for _ in response_gen:
        yield _


class HasTracer:
    def __init__(self, tracer: Tracer) -> None:
        self._tracer = tracer


class MockStreamGenerateContent(
    grpc.UnaryStreamMultiCallable,  # type: ignore[misc]
):
    def __init__(self, response_gen: Iterator[GenerateContentResponse]) -> None:
        self._response_gen = response_gen

    def __call__(self, *args: Any, **kwargs: Any) -> Iterator[GenerateContentResponse]:
        return self._response_gen


class MockAsyncGenerateContent(
    HasTracer,
    grpc.aio.UnaryUnaryMultiCallable[GenerateContentRequest, GenerateContentResponse],  # type: ignore[misc]
):
    def __init__(self, response: GenerateContentResponse, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._response = response

    def __call__(
        self, request: GenerateContentRequest, **kwargs: Any
    ) -> grpc.aio.UnaryUnaryCall[GenerateContentRequest, GenerateContentResponse]:
        return MockUnaryUnaryCall(self._response, self._tracer)


class MockAsyncStreamGenerateContent(
    HasTracer,
    grpc.aio.UnaryStreamMultiCallable[GenerateContentRequest, GenerateContentResponse],  # type: ignore[misc]
):
    def __init__(
        self, response_gen: AsyncIterator[GenerateContentResponse], *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self._response_gen = response_gen

    def __call__(
        self, request: GenerateContentRequest, **kwargs: Any
    ) -> grpc.aio.UnaryStreamCall[GenerateContentRequest, GenerateContentResponse]:
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
    grpc.aio.UnaryUnaryCall[GenerateContentRequest, GenerateContentResponse],  # type: ignore[misc]
):
    def __init__(self, response: GenerateContentResponse, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._response = response

    def __await__(self) -> Generator[Any, None, GenerateContentResponse]:
        with self._tracer.start_as_current_span(TEST):
            yield from asyncio.sleep(0).__await__()
        with self._tracer.start_as_current_span(TEST):
            return self._response


class MockUnaryStreamCall(
    HasTracer,
    MockAsyncCall,
    grpc.aio.UnaryStreamCall[GenerateContentRequest, GenerateContentResponse],  # type: ignore[misc]
):
    def __init__(
        self, response_gen: AsyncIterator[GenerateContentResponse], *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self._response_gen = response_gen

    def __aiter__(self) -> AsyncIterator[GenerateContentResponse]:
        with self._tracer.start_as_current_span(TEST):
            return self._response_gen

    async def read(self) -> Any: ...


@contextmanager
def patch_grpc_transport_generate_content(patch: Any) -> Iterator[None]:
    replacement = property(lambda *_, **__: patch)
    cls = PredictionServiceGrpcTransport
    original = cls.generate_content
    setattr(cls, "generate_content", replacement)
    yield
    setattr(cls, "generate_content", original)


@contextmanager
def patch_grpc_transport_stream_generate_content(patch: Any) -> Iterator[None]:
    replacement = property(lambda *_, **__: patch)
    cls = PredictionServiceGrpcTransport
    original = cls.stream_generate_content
    setattr(cls, "stream_generate_content", replacement)
    yield
    setattr(cls, "stream_generate_content", original)


@contextmanager
def patch_grpc_asyncio_transport_generate_content(patch: Any) -> Iterator[None]:
    replacement = property(lambda *_, **__: patch)
    cls = PredictionServiceGrpcAsyncIOTransport
    original = cls.generate_content
    setattr(cls, "generate_content", replacement)
    yield
    setattr(cls, "generate_content", original)


@contextmanager
def patch_grpc_asyncio_transport_stream_generate_content(patch: Any) -> Iterator[None]:
    replacement = property(lambda *_, **__: patch)
    cls = PredictionServiceGrpcAsyncIOTransport
    original = cls.stream_generate_content
    setattr(cls, "stream_generate_content", replacement)
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


def _check_llm_message(
    prefix: str,
    i: int,
    attributes: Dict[str, Any],
    message: Dict[str, Any],
    hide_text: bool = False,
    hide_images: bool = False,
    image_limit: Optional[int] = None,
) -> None:
    assert attributes.pop(message_role(prefix, i), None) == message.get("role")
    expected_content = message.get("content")
    if isinstance(expected_content, list):
        for j, expected_content_item in enumerate(expected_content):
            content_item_type = attributes.pop(message_contents_type(prefix, i, j), None)
            expected_content_item_type = expected_content_item.get("type")
            if expected_content_item_type == "image_url":
                expected_content_item_type = "image"
            assert content_item_type == expected_content_item_type
            if content_item_type == "text":
                content_item_text = attributes.pop(message_contents_text(prefix, i, j), None)
                if hide_text:
                    assert content_item_text == REDACTED_VALUE
                else:
                    assert content_item_text == expected_content_item.get("text")
            elif content_item_type == "image":
                content_item_image_url = attributes.pop(
                    message_contents_image_url(prefix, i, j), None
                )
                if hide_images:
                    assert content_item_image_url is None
                else:
                    expected_url = expected_content_item.get("image_url").get("url")
                    if image_limit is not None and len(expected_url) > image_limit:
                        assert content_item_image_url == REDACTED_VALUE
                    else:
                        assert content_item_image_url == expected_url
    else:
        content = attributes.pop(message_content(prefix, i), None)
        if expected_content is not None and hide_text:
            assert content == REDACTED_VALUE
        else:
            assert content == expected_content


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


CREDENTIALS = Credentials("123")  # type: ignore[no-untyped-call]
MSG = "MSG"
TEST = "TEST"


class Err(BaseException): ...


F = TypeVar("F", Callable[..., Any], Iterator[GenerateContentResponse])


def _may_err(has_error: bool, tracer: Tracer, obj: F) -> F:
    if isinstance(obj, Iterator):

        def gen() -> Iterator[GenerateContentResponse]:
            for item in obj:
                with tracer.start_as_current_span(TEST):
                    yield item
            if has_error:
                with tracer.start_as_current_span(TEST):
                    raise Err(MSG)

        return gen()

    def fn(*args: Any, **kwargs: Any) -> Any:
        with tracer.start_as_current_span(TEST):
            ans = obj(*args, **kwargs)
        if has_error:
            with tracer.start_as_current_span(TEST):
                raise Err(MSG)
        return ans

    return fn


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

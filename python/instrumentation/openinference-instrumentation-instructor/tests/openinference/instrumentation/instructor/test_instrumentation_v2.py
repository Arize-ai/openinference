import gc
from importlib import import_module
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Tuple

import instructor
import pytest
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import BaseModel, model_serializer
from wrapt import FunctionWrapper

from openinference.instrumentation.instructor import InstructorInstrumentor
from openinference.instrumentation.instructor._wrappers import _V2CreateFactoryWrapper
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes

patch_module = pytest.importorskip("instructor.v2.core.patch")
pytest.importorskip("instructor.v2.providers.openai.handlers")


class UserInfo(BaseModel):
    name: str
    age: int


class UnserializableResponse(BaseModel):
    value: str

    @model_serializer
    def serialize_model(self) -> Any:
        raise RuntimeError("output serialization failed")


class UnserializableInput:
    def __repr__(self) -> str:
        raise RuntimeError("input serialization failed")


class CloseAwareIterator:
    def __init__(self) -> None:
        self.closed = False

    def __iter__(self) -> "CloseAwareIterator":
        return self

    def __next__(self) -> UserInfo:
        return UserInfo(name="Jane Doe", age=31)

    def close(self) -> None:
        self.closed = True


class CloseAwareAsyncIterator:
    def __init__(self) -> None:
        self.closed = False

    def __aiter__(self) -> "CloseAwareAsyncIterator":
        return self

    async def __anext__(self) -> UserInfo:
        return UserInfo(name="Jane Doe", age=31)

    async def aclose(self) -> None:
        self.closed = True


def _make_tracer_provider() -> Tuple[TracerProvider, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    return tracer_provider, exporter


def _make_client(create: Callable[..., Any]) -> Any:
    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    )


def test_wraps_create_factories_and_restores_them_between_cycles() -> None:
    original_sync_factory = patch_module._create_sync_wrapper
    original_async_factory = patch_module._create_async_wrapper
    first_provider, first_exporter = _make_tracer_provider()
    second_provider, second_exporter = _make_tracer_provider()
    instrumentor = InstructorInstrumentor()

    try:
        instrumentor.instrument(tracer_provider=first_provider)
        assert isinstance(patch_module._create_sync_wrapper, FunctionWrapper)
        assert isinstance(patch_module._create_async_wrapper, FunctionWrapper)

        first_client = instructor.patch(_make_client(lambda **_: UserInfo(name="Jane Doe", age=31)))
        assert first_client.chat.completions.create(
            response_model=None,
            messages=[{"role": "user", "content": "Create a user"}],
        ) == UserInfo(name="Jane Doe", age=31)
        assert len(first_exporter.get_finished_spans()) == 1

        instrumentor.uninstrument()
        assert patch_module._create_sync_wrapper is original_sync_factory
        assert patch_module._create_async_wrapper is original_async_factory

        # A client patched while instrumented must stop tracing once uninstrumented.
        first_client.chat.completions.create(
            response_model=None,
            messages=[{"role": "user", "content": "Create a user"}],
        )
        assert len(first_exporter.get_finished_spans()) == 1

        instrumentor.instrument(tracer_provider=second_provider)
        second_client = instructor.patch(
            _make_client(lambda **_: UserInfo(name="John Doe", age=30))
        )
        second_client.chat.completions.create(
            response_model=None,
            messages=[{"role": "user", "content": "Create a user"}],
        )
        assert len(first_exporter.get_finished_spans()) == 1
        assert len(second_exporter.get_finished_spans()) == 1
    finally:
        instrumentor.uninstrument()

    assert patch_module._create_sync_wrapper is original_sync_factory
    assert patch_module._create_async_wrapper is original_async_factory


def test_public_positional_patch_returns_client_and_emits_span(monkeypatch: Any) -> None:
    tracer_provider, exporter = _make_tracer_provider()
    instrumentor = InstructorInstrumentor()
    client = _make_client(lambda **_: None)

    def retry_sync_v2(**kwargs: Any) -> UserInfo:
        return UserInfo(name="Jane Doe", age=31)

    monkeypatch.setattr(patch_module, "retry_sync_v2", retry_sync_v2)
    try:
        instrumentor.instrument(tracer_provider=tracer_provider)
        patched_client = instructor.patch(client)
        response = patched_client.chat.completions.create(
            response_model=UserInfo,
            messages=[{"role": "user", "content": "Create a user"}],
        )
    finally:
        instrumentor.uninstrument()

    assert patched_client is client
    assert response == UserInfo(name="Jane Doe", age=31)
    (span,) = exporter.get_finished_spans()
    attributes = dict(span.attributes or {})
    assert span.name == "instructor.create"
    assert attributes.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND) == (
        OpenInferenceSpanKindValues.CHAIN.value
    )
    assert attributes.pop(SpanAttributes.INPUT_VALUE) == (
        '[{"role": "user", "content": "Create a user"}]'
    )
    assert attributes.pop(SpanAttributes.INPUT_MIME_TYPE) == "application/json"
    assert attributes.pop(SpanAttributes.OUTPUT_VALUE) == '{"name": "Jane Doe", "age": 31}'
    assert attributes.pop(SpanAttributes.OUTPUT_MIME_TYPE) == "application/json"
    assert attributes.pop("instructor.response_model") == "UserInfo"
    assert not attributes
    assert span.status.status_code == trace_api.StatusCode.OK


def test_public_patch_create_emits_only_one_chain_span(monkeypatch: Any) -> None:
    tracer_provider, exporter = _make_tracer_provider()
    instrumentor = InstructorInstrumentor()

    def retry_sync_v2(**kwargs: Any) -> UserInfo:
        return UserInfo(name="Jane Doe", age=31)

    monkeypatch.setattr(patch_module, "retry_sync_v2", retry_sync_v2)
    try:
        instrumentor.instrument(tracer_provider=tracer_provider)
        create = instructor.patch(create=lambda **_: None)
        response = create(
            response_model=UserInfo,
            messages=[{"role": "user", "content": "Create a user"}],
        )
    finally:
        instrumentor.uninstrument()

    assert response == UserInfo(name="Jane Doe", age=31)
    (span,) = exporter.get_finished_spans()
    assert span.name == "instructor.create"
    attributes = dict(span.attributes or {})
    assert attributes.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND) == (
        OpenInferenceSpanKindValues.CHAIN.value
    )
    assert attributes.pop(SpanAttributes.INPUT_VALUE)
    assert attributes.pop(SpanAttributes.INPUT_MIME_TYPE) == "application/json"
    assert attributes.pop(SpanAttributes.OUTPUT_VALUE)
    assert attributes.pop(SpanAttributes.OUTPUT_MIME_TYPE) == "application/json"
    assert attributes.pop("instructor.response_model") == "UserInfo"
    assert not attributes


def test_from_provider_emits_span(monkeypatch: Any) -> None:
    tracer_provider, exporter = _make_tracer_provider()
    instrumentor = InstructorInstrumentor()

    def retry_sync_v2(**kwargs: Any) -> UserInfo:
        return UserInfo(name="Jane Doe", age=31)

    monkeypatch.setattr(patch_module, "retry_sync_v2", retry_sync_v2)
    try:
        instrumentor.instrument(tracer_provider=tracer_provider)
        client = getattr(instructor, "from_provider")("openai/gpt-4o-mini", api_key="test-api-key")
        response = client.create(
            response_model=UserInfo,
            messages=[{"role": "user", "content": "Create a user"}],
        )
    finally:
        instrumentor.uninstrument()

    assert response == UserInfo(name="Jane Doe", age=31)
    (span,) = exporter.get_finished_spans()
    attributes = dict(span.attributes or {})
    assert span.name == "instructor.create"
    assert attributes.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND) == (
        OpenInferenceSpanKindValues.CHAIN.value
    )
    assert attributes.pop(SpanAttributes.INPUT_VALUE) == (
        '[{"role": "user", "content": "Create a user"}]'
    )
    assert attributes.pop(SpanAttributes.INPUT_MIME_TYPE) == "application/json"
    assert attributes.pop(SpanAttributes.OUTPUT_VALUE) == '{"name": "Jane Doe", "age": 31}'
    assert attributes.pop(SpanAttributes.OUTPUT_MIME_TYPE) == "application/json"
    assert attributes.pop("instructor.response_model") == "UserInfo"
    assert not attributes
    assert span.status.status_code == trace_api.StatusCode.OK


def test_cache_hit_emits_a_span(monkeypatch: Any) -> None:
    AutoCache = import_module("instructor.cache").AutoCache

    tracer_provider, exporter = _make_tracer_provider()
    instrumentor = InstructorInstrumentor()
    retry_calls = 0

    def retry_sync_v2(**kwargs: Any) -> UserInfo:
        nonlocal retry_calls
        retry_calls += 1
        return UserInfo(name="Jane Doe", age=31)

    monkeypatch.setattr(patch_module, "retry_sync_v2", retry_sync_v2)
    try:
        instrumentor.instrument(tracer_provider=tracer_provider)
        client = instructor.patch(_make_client(lambda **_: None))
        cache = AutoCache()
        create_kwargs = {
            "response_model": UserInfo,
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Create a user"}],
            "cache": cache,
        }
        first_response = client.chat.completions.create(**create_kwargs)
        cached_response = client.chat.completions.create(**create_kwargs)
    finally:
        instrumentor.uninstrument()

    assert first_response.name == cached_response.name == "Jane Doe"
    assert first_response.age == cached_response.age == 31
    assert retry_calls == 1
    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    assert all(span.name == "instructor.create" for span in spans)
    assert all(span.status.status_code == trace_api.StatusCode.OK for span in spans)


def test_attribute_serialization_failures_do_not_change_application_result() -> None:
    tracer_provider, exporter = _make_tracer_provider()
    wrapper = _V2CreateFactoryWrapper(tracer_provider.get_tracer(__name__), is_async=False)
    expected = UnserializableResponse(value="valid response")

    def factory() -> Callable[..., UnserializableResponse]:
        return lambda **_: expected

    create = wrapper(factory, None, (), {})
    response = create(
        response_model=UserInfo,
        messages=[UnserializableInput()],
    )

    assert response is expected
    (span,) = exporter.get_finished_spans()
    attributes = dict(span.attributes or {})
    assert attributes.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND) == (
        OpenInferenceSpanKindValues.CHAIN.value
    )
    assert attributes.pop("instructor.response_model") == "UserInfo"
    assert SpanAttributes.INPUT_VALUE not in attributes
    assert SpanAttributes.INPUT_MIME_TYPE not in attributes
    assert SpanAttributes.OUTPUT_VALUE not in attributes
    assert SpanAttributes.OUTPUT_MIME_TYPE not in attributes
    assert not attributes
    assert span.status.status_code == trace_api.StatusCode.OK


def test_streaming_span_finishes_with_consumed_output() -> None:
    tracer_provider, exporter = _make_tracer_provider()
    wrapper = _V2CreateFactoryWrapper(tracer_provider.get_tracer(__name__), is_async=False)

    def factory() -> Callable[..., Any]:
        return lambda **_: iter(
            [
                UserInfo(name="Jane Doe", age=31),
                UserInfo(name="John Doe", age=30),
            ]
        )

    create = wrapper(factory, None, (), {})
    stream = create(
        response_model=Iterable[UserInfo],
        messages=[{"role": "user", "content": "Create users"}],
    )
    assert not exporter.get_finished_spans()
    assert next(stream) == UserInfo(name="Jane Doe", age=31)
    assert not exporter.get_finished_spans()
    assert list(stream) == [UserInfo(name="John Doe", age=30)]

    (span,) = exporter.get_finished_spans()
    attributes = dict(span.attributes or {})
    assert attributes.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND) == (
        OpenInferenceSpanKindValues.CHAIN.value
    )
    assert attributes.pop(SpanAttributes.INPUT_VALUE) == (
        '[{"role": "user", "content": "Create users"}]'
    )
    assert attributes.pop(SpanAttributes.INPUT_MIME_TYPE) == "application/json"
    assert attributes.pop(SpanAttributes.OUTPUT_VALUE) == (
        '[{"name": "Jane Doe", "age": 31}, {"name": "John Doe", "age": 30}]'
    )
    assert attributes.pop(SpanAttributes.OUTPUT_MIME_TYPE) == "application/json"
    assert attributes.pop("instructor.response_model") == "Iterable"
    assert not attributes
    assert span.status.status_code == trace_api.StatusCode.OK


def test_unstarted_stream_close_finishes_span_and_closes_underlying_iterator() -> None:
    tracer_provider, exporter = _make_tracer_provider()
    wrapper = _V2CreateFactoryWrapper(tracer_provider.get_tracer(__name__), is_async=False)
    iterator = CloseAwareIterator()

    def factory() -> Callable[..., Any]:
        return lambda **_: iterator

    create = wrapper(factory, None, (), {})
    stream = create(messages=[{"role": "user", "content": "Create users"}])
    assert not exporter.get_finished_spans()
    stream.close()

    assert iterator.closed
    (span,) = exporter.get_finished_spans()
    attributes = dict(span.attributes or {})
    assert span.name == "instructor.create"
    assert attributes.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND) == (
        OpenInferenceSpanKindValues.CHAIN.value
    )
    assert attributes.pop(SpanAttributes.INPUT_VALUE) == (
        '[{"role": "user", "content": "Create users"}]'
    )
    assert attributes.pop(SpanAttributes.INPUT_MIME_TYPE) == "application/json"
    assert not attributes
    assert span.status.status_code == trace_api.StatusCode.OK


def test_partial_stream_reports_only_the_latest_snapshot() -> None:
    tracer_provider, exporter = _make_tracer_provider()
    wrapper = _V2CreateFactoryWrapper(tracer_provider.get_tracer(__name__), is_async=False)
    snapshots = [
        UserInfo(name="J", age=0),
        UserInfo(name="Jane", age=0),
        UserInfo(name="Jane Doe", age=31),
    ]

    def factory() -> Callable[..., Any]:
        return lambda **_: iter(snapshots)

    create = wrapper(factory, None, (), {})
    stream = create(response_model=UserInfo, messages=[{"role": "user", "content": "Create"}])
    assert list(stream) == snapshots
    assert stream._output._items == []

    (span,) = exporter.get_finished_spans()
    attributes = dict(span.attributes or {})
    assert attributes[SpanAttributes.OUTPUT_VALUE] == '{"name": "Jane Doe", "age": 31}'
    assert span.status.status_code == trace_api.StatusCode.OK


def test_abandoned_stream_finishes_span_on_garbage_collection() -> None:
    tracer_provider, exporter = _make_tracer_provider()
    wrapper = _V2CreateFactoryWrapper(tracer_provider.get_tracer(__name__), is_async=False)

    def factory() -> Callable[..., Any]:
        return lambda **_: iter(
            [
                UserInfo(name="Jane Doe", age=31),
                UserInfo(name="John Doe", age=30),
            ]
        )

    create = wrapper(factory, None, (), {})
    stream = create(
        response_model=Iterable[UserInfo],
        messages=[{"role": "user", "content": "Create users"}],
    )
    assert next(stream) == UserInfo(name="Jane Doe", age=31)
    assert not exporter.get_finished_spans()

    del stream
    gc.collect()

    (span,) = exporter.get_finished_spans()
    attributes = dict(span.attributes or {})
    assert attributes[SpanAttributes.OUTPUT_VALUE] == '[{"name": "Jane Doe", "age": 31}]'
    assert span.status.status_code == trace_api.StatusCode.OK


def test_streaming_span_records_delayed_iteration_failure() -> None:
    tracer_provider, exporter = _make_tracer_provider()
    wrapper = _V2CreateFactoryWrapper(tracer_provider.get_tracer(__name__), is_async=False)

    def stream() -> Any:
        yield UserInfo(name="Jane Doe", age=31)
        raise RuntimeError("stream failure")

    def factory() -> Callable[..., Any]:
        return lambda **_: stream()

    create = wrapper(factory, None, (), {})
    result = create(messages=[{"role": "user", "content": "Create users"}])
    assert next(result) == UserInfo(name="Jane Doe", age=31)
    assert not exporter.get_finished_spans()

    with pytest.raises(RuntimeError, match="stream failure"):
        next(result)

    (span,) = exporter.get_finished_spans()
    attributes = dict(span.attributes or {})
    assert attributes.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND) == (
        OpenInferenceSpanKindValues.CHAIN.value
    )
    assert attributes.pop(SpanAttributes.INPUT_VALUE)
    assert attributes.pop(SpanAttributes.INPUT_MIME_TYPE) == "application/json"
    assert SpanAttributes.OUTPUT_VALUE not in attributes
    assert SpanAttributes.OUTPUT_MIME_TYPE not in attributes
    assert not attributes
    assert span.status.status_code == trace_api.StatusCode.ERROR


@pytest.mark.asyncio
async def test_async_create_awaits_execution_and_records_failure() -> None:
    tracer_provider, exporter = _make_tracer_provider()
    wrapper = _V2CreateFactoryWrapper(tracer_provider.get_tracer(__name__), is_async=True)

    async def fail(**kwargs: Any) -> None:
        raise RuntimeError("async failure")

    def factory() -> Callable[..., Any]:
        return fail

    create = wrapper(factory, None, (), {})
    coroutine = create(messages=[{"role": "user", "content": "Create a user"}])
    assert not exporter.get_finished_spans()

    with pytest.raises(RuntimeError, match="async failure"):
        await coroutine

    (span,) = exporter.get_finished_spans()
    attributes = dict(span.attributes or {})
    assert span.name == "instructor.async_create"
    assert attributes.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND) == (
        OpenInferenceSpanKindValues.CHAIN.value
    )
    assert attributes.pop(SpanAttributes.INPUT_VALUE) == (
        '[{"role": "user", "content": "Create a user"}]'
    )
    assert attributes.pop(SpanAttributes.INPUT_MIME_TYPE) == "application/json"
    assert SpanAttributes.OUTPUT_VALUE not in attributes
    assert SpanAttributes.OUTPUT_MIME_TYPE not in attributes
    assert not attributes
    assert span.status.status_code == trace_api.StatusCode.ERROR


@pytest.mark.asyncio
async def test_async_streaming_span_records_delayed_iteration_failure() -> None:
    tracer_provider, exporter = _make_tracer_provider()
    wrapper = _V2CreateFactoryWrapper(tracer_provider.get_tracer(__name__), is_async=True)

    async def create_stream(**kwargs: Any) -> Any:
        async def stream() -> Any:
            yield UserInfo(name="Jane Doe", age=31)
            raise RuntimeError("async stream failure")

        return stream()

    def factory() -> Callable[..., Any]:
        return create_stream

    create = wrapper(factory, None, (), {})
    stream = await create(messages=[{"role": "user", "content": "Create users"}])
    assert not exporter.get_finished_spans()
    assert await stream.__anext__() == UserInfo(name="Jane Doe", age=31)
    assert not exporter.get_finished_spans()

    with pytest.raises(RuntimeError, match="async stream failure"):
        await stream.__anext__()

    (span,) = exporter.get_finished_spans()
    attributes = dict(span.attributes or {})
    assert span.name == "instructor.async_create"
    assert attributes.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND) == (
        OpenInferenceSpanKindValues.CHAIN.value
    )
    assert attributes.pop(SpanAttributes.INPUT_VALUE)
    assert attributes.pop(SpanAttributes.INPUT_MIME_TYPE) == "application/json"
    assert SpanAttributes.OUTPUT_VALUE not in attributes
    assert SpanAttributes.OUTPUT_MIME_TYPE not in attributes
    assert not attributes
    assert span.status.status_code == trace_api.StatusCode.ERROR


@pytest.mark.asyncio
async def test_unstarted_async_stream_aclose_finishes_span_and_closes_underlying_iterator() -> None:
    tracer_provider, exporter = _make_tracer_provider()
    wrapper = _V2CreateFactoryWrapper(tracer_provider.get_tracer(__name__), is_async=True)
    iterator = CloseAwareAsyncIterator()

    async def create_stream(**kwargs: Any) -> CloseAwareAsyncIterator:
        return iterator

    def factory() -> Callable[..., Any]:
        return create_stream

    create = wrapper(factory, None, (), {})
    stream = await create(messages=[{"role": "user", "content": "Create users"}])
    assert not exporter.get_finished_spans()
    await stream.aclose()

    assert iterator.closed
    (span,) = exporter.get_finished_spans()
    attributes = dict(span.attributes or {})
    assert span.name == "instructor.async_create"
    assert attributes.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND) == (
        OpenInferenceSpanKindValues.CHAIN.value
    )
    assert attributes.pop(SpanAttributes.INPUT_VALUE) == (
        '[{"role": "user", "content": "Create users"}]'
    )
    assert attributes.pop(SpanAttributes.INPUT_MIME_TYPE) == "application/json"
    assert not attributes
    assert span.status.status_code == trace_api.StatusCode.OK


@pytest.mark.asyncio
async def test_abandoned_async_stream_finishes_span_on_garbage_collection() -> None:
    tracer_provider, exporter = _make_tracer_provider()
    wrapper = _V2CreateFactoryWrapper(tracer_provider.get_tracer(__name__), is_async=True)

    async def create_stream(**kwargs: Any) -> CloseAwareAsyncIterator:
        return CloseAwareAsyncIterator()

    def factory() -> Callable[..., Any]:
        return create_stream

    create = wrapper(factory, None, (), {})
    stream = await create(messages=[{"role": "user", "content": "Create users"}])
    assert await stream.__anext__() == UserInfo(name="Jane Doe", age=31)
    assert not exporter.get_finished_spans()

    del stream
    gc.collect()

    (span,) = exporter.get_finished_spans()
    attributes = dict(span.attributes or {})
    assert span.name == "instructor.async_create"
    assert attributes[SpanAttributes.OUTPUT_VALUE] == '{"name": "Jane Doe", "age": 31}'
    assert span.status.status_code == trace_api.StatusCode.OK

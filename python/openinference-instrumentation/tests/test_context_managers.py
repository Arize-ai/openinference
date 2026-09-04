import asyncio
import contextvars
import gc
import json
from typing import Any, AsyncGenerator, AsyncIterator, Dict, Generator, Iterator, List, Tuple, cast

import pytest
from opentelemetry import trace as trace_api
from opentelemetry.context import (
    _SUPPRESS_INSTRUMENTATION_KEY,
    get_current,
    get_value,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import INVALID_SPAN_CONTEXT, SpanContext

from openinference.instrumentation import (
    OITracer,
    TracerProvider,
    capture_span_context,
    get_attributes_from_context,
    safe_json_dumps,
    suppress_tracing,
    using_attributes,
    using_metadata,
    using_prompt_template,
    using_session,
    using_tags,
    using_user,
)
from openinference.instrumentation.context_attributes import _UsingAttributesContextManager
from openinference.semconv.trace import SpanAttributes

_live: contextvars.ContextVar[int] = contextvars.ContextVar("live")
_body_owned: contextvars.ContextVar[str] = contextvars.ContextVar("body_owned", default="unset")


def test_suppress_tracing() -> None:
    with suppress_tracing():
        assert get_value(_SUPPRESS_INSTRUMENTATION_KEY) is True
    assert get_value(_SUPPRESS_INSTRUMENTATION_KEY) is None


def test_using_session(session_id: str) -> None:
    with using_session(session_id):
        assert get_value(SpanAttributes.SESSION_ID) == session_id
    assert get_value(SpanAttributes.SESSION_ID) is None


def test_using_user(user_id: str) -> None:
    with using_user(user_id):
        assert get_value(SpanAttributes.USER_ID) == user_id
    assert get_value(SpanAttributes.USER_ID) is None


def test_using_metadata(metadata: Dict[str, Any]) -> None:
    with using_metadata(metadata):
        assert get_value(SpanAttributes.METADATA) == json.dumps(metadata)
    assert get_value(SpanAttributes.METADATA) is None


def test_using_tags(tags: List[str]) -> None:
    with using_tags(tags):
        assert get_value(SpanAttributes.TAG_TAGS) == tags
    assert get_value(SpanAttributes.TAG_TAGS) is None


def test_using_prompt_template(
    prompt_template: str, prompt_template_version: str, prompt_template_variables: Dict[str, Any]
) -> None:
    with using_prompt_template(
        template=prompt_template,
        version=prompt_template_version,
        variables=prompt_template_variables,
    ):
        assert get_value(SpanAttributes.LLM_PROMPT_TEMPLATE) == prompt_template
        assert get_value(SpanAttributes.LLM_PROMPT_TEMPLATE_VERSION) == prompt_template_version
        assert get_value(SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES) == json.dumps(
            prompt_template_variables
        )
    assert get_value(SpanAttributes.LLM_PROMPT_TEMPLATE) is None
    assert get_value(SpanAttributes.LLM_PROMPT_TEMPLATE_VERSION) is None
    assert get_value(SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES) is None


def test_using_attributes(
    session_id: str,
    user_id: str,
    metadata: Dict[str, Any],
    tags: List[str],
    prompt_template: str,
    prompt_template_version: str,
    prompt_template_variables: Dict[str, Any],
) -> None:
    with using_attributes(
        session_id=session_id,
        user_id=user_id,
        metadata=metadata,
        tags=tags,
        prompt_template=prompt_template,
        prompt_template_version=prompt_template_version,
        prompt_template_variables=prompt_template_variables,
    ):
        assert get_value(SpanAttributes.SESSION_ID) == session_id
        assert get_value(SpanAttributes.USER_ID) == user_id
        assert get_value(SpanAttributes.METADATA) == json.dumps(metadata)
        assert get_value(SpanAttributes.TAG_TAGS) == tags
        assert get_value(SpanAttributes.LLM_PROMPT_TEMPLATE) == prompt_template
        assert get_value(SpanAttributes.LLM_PROMPT_TEMPLATE_VERSION) == prompt_template_version
        assert get_value(SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES) == json.dumps(
            prompt_template_variables
        )
    assert get_value(SpanAttributes.SESSION_ID) is None
    assert get_value(SpanAttributes.USER_ID) is None
    assert get_value(SpanAttributes.METADATA) is None
    assert get_value(SpanAttributes.TAG_TAGS) is None
    assert get_value(SpanAttributes.LLM_PROMPT_TEMPLATE) is None
    assert get_value(SpanAttributes.LLM_PROMPT_TEMPLATE_VERSION) is None
    assert get_value(SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES) is None


def test_using_session_decorator(session_id: str) -> None:
    @using_session(session_id)
    def f() -> None:
        assert get_value(SpanAttributes.SESSION_ID) == session_id

    f()
    assert get_value(SpanAttributes.SESSION_ID) is None


def test_using_user_decorator(user_id: str) -> None:
    @using_user(user_id)
    def f() -> None:
        assert get_value(SpanAttributes.USER_ID) == user_id

    f()
    assert get_value(SpanAttributes.USER_ID) is None


def test_using_metadata_decorator(metadata: Dict[str, Any]) -> None:
    @using_metadata(metadata)
    def f() -> None:
        assert get_value(SpanAttributes.METADATA) == json.dumps(metadata)

    f()
    assert get_value(SpanAttributes.METADATA) is None


def test_using_tags_decorator(tags: List[str]) -> None:
    @using_tags(tags)
    def f() -> None:
        assert get_value(SpanAttributes.TAG_TAGS) == tags

    f()
    assert get_value(SpanAttributes.TAG_TAGS) is None


def test_using_prompt_template_decorator(
    prompt_template: str, prompt_template_version: str, prompt_template_variables: Dict[str, Any]
) -> None:
    @using_prompt_template(
        template=prompt_template,
        version=prompt_template_version,
        variables=prompt_template_variables,
    )
    def f() -> None:
        assert get_value(SpanAttributes.LLM_PROMPT_TEMPLATE) == prompt_template
        assert get_value(SpanAttributes.LLM_PROMPT_TEMPLATE_VERSION) == prompt_template_version
        assert get_value(SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES) == json.dumps(
            prompt_template_variables
        )

    f()
    assert get_value(SpanAttributes.LLM_PROMPT_TEMPLATE) is None
    assert get_value(SpanAttributes.LLM_PROMPT_TEMPLATE_VERSION) is None
    assert get_value(SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES) is None


def test_using_attributes_decorator(
    session_id: str,
    user_id: str,
    metadata: Dict[str, Any],
    tags: List[str],
    prompt_template: str,
    prompt_template_version: str,
    prompt_template_variables: Dict[str, Any],
) -> None:
    @using_attributes(
        session_id=session_id,
        user_id=user_id,
        metadata=metadata,
        tags=tags,
        prompt_template=prompt_template,
        prompt_template_version=prompt_template_version,
        prompt_template_variables=prompt_template_variables,
    )
    def f() -> None:
        assert get_value(SpanAttributes.SESSION_ID) == session_id
        assert get_value(SpanAttributes.USER_ID) == user_id
        assert get_value(SpanAttributes.METADATA) == json.dumps(metadata)
        assert get_value(SpanAttributes.TAG_TAGS) == tags
        assert get_value(SpanAttributes.LLM_PROMPT_TEMPLATE) == prompt_template
        assert get_value(SpanAttributes.LLM_PROMPT_TEMPLATE_VERSION) == prompt_template_version
        assert get_value(SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES) == json.dumps(
            prompt_template_variables
        )

    f()
    assert get_value(SpanAttributes.SESSION_ID) is None
    assert get_value(SpanAttributes.USER_ID) is None
    assert get_value(SpanAttributes.METADATA) is None
    assert get_value(SpanAttributes.TAG_TAGS) is None
    assert get_value(SpanAttributes.LLM_PROMPT_TEMPLATE) is None
    assert get_value(SpanAttributes.LLM_PROMPT_TEMPLATE_VERSION) is None
    assert get_value(SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES) is None


def test_using_session_decorator_is_reentrant() -> None:
    @using_session("recursive-session")
    def descend(depth: int) -> None:
        assert get_value(SpanAttributes.SESSION_ID) == "recursive-session"
        if depth > 0:
            descend(depth - 1)
        assert get_value(SpanAttributes.SESSION_ID) == "recursive-session"

    descend(2)
    assert get_value(SpanAttributes.SESSION_ID) is None


@pytest.mark.parametrize(
    "decorator,expected_context_values",
    [
        pytest.param(
            using_session("test-session"),
            {SpanAttributes.SESSION_ID: "test-session"},
            id="using_session",
        ),
        pytest.param(
            using_user("test-user"),
            {SpanAttributes.USER_ID: "test-user"},
            id="using_user",
        ),
        pytest.param(
            using_metadata({"key": "value"}),
            {SpanAttributes.METADATA: json.dumps({"key": "value"})},
            id="using_metadata",
        ),
        pytest.param(
            using_tags(["tag-1", "tag-2"]),
            {SpanAttributes.TAG_TAGS: ["tag-1", "tag-2"]},
            id="using_tags",
        ),
        pytest.param(
            using_prompt_template(template="Hello {name}", version="v1", variables={"name": "x"}),
            {
                SpanAttributes.LLM_PROMPT_TEMPLATE: "Hello {name}",
                SpanAttributes.LLM_PROMPT_TEMPLATE_VERSION: "v1",
                SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES: json.dumps({"name": "x"}),
            },
            id="using_prompt_template",
        ),
    ],
)
async def test_async_decorator_attaches_each_attribute_across_awaits(
    decorator: _UsingAttributesContextManager, expected_context_values: Dict[str, Any]
) -> None:
    @decorator
    async def read_context_values() -> Dict[str, Any]:
        values_before_await = {key: get_value(key) for key in expected_context_values}
        await asyncio.sleep(0.001)
        values_after_await = {key: get_value(key) for key in expected_context_values}
        assert values_before_await == values_after_await
        return values_after_await

    assert await read_context_values() == expected_context_values
    for attribute_key in expected_context_values:
        assert get_value(attribute_key) is None


async def test_using_attributes_async_decorator(
    session_id: str,
    user_id: str,
    metadata: Dict[str, Any],
    tags: List[str],
    prompt_template: str,
    prompt_template_version: str,
    prompt_template_variables: Dict[str, Any],
) -> None:
    @using_attributes(
        session_id=session_id,
        user_id=user_id,
        metadata=metadata,
        tags=tags,
        prompt_template=prompt_template,
        prompt_template_version=prompt_template_version,
        prompt_template_variables=prompt_template_variables,
    )
    async def read_all_attributes() -> None:
        for _ in range(2):
            assert get_value(SpanAttributes.SESSION_ID) == session_id
            assert get_value(SpanAttributes.USER_ID) == user_id
            assert get_value(SpanAttributes.METADATA) == json.dumps(metadata)
            assert get_value(SpanAttributes.TAG_TAGS) == tags
            assert get_value(SpanAttributes.LLM_PROMPT_TEMPLATE) == prompt_template
            assert get_value(SpanAttributes.LLM_PROMPT_TEMPLATE_VERSION) == prompt_template_version
            assert get_value(SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES) == json.dumps(
                prompt_template_variables
            )
            await asyncio.sleep(0.001)

    await read_all_attributes()
    assert get_value(SpanAttributes.SESSION_ID) is None
    assert get_value(SpanAttributes.USER_ID) is None
    assert get_value(SpanAttributes.METADATA) is None
    assert get_value(SpanAttributes.TAG_TAGS) is None
    assert get_value(SpanAttributes.LLM_PROMPT_TEMPLATE) is None
    assert get_value(SpanAttributes.LLM_PROMPT_TEMPLATE_VERSION) is None
    assert get_value(SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES) is None


async def test_async_decorator_isolates_concurrent_calls_of_the_same_function() -> None:
    @using_session("shared-session")
    async def read_session_after(delay_seconds: float) -> str:
        await asyncio.sleep(delay_seconds)
        return str(get_value(SpanAttributes.SESSION_ID))

    observed_session_ids = await asyncio.gather(
        read_session_after(0.004), read_session_after(0.001), read_session_after(0.002)
    )
    assert list(observed_session_ids) == ["shared-session"] * 3
    assert get_value(SpanAttributes.SESSION_ID) is None


async def test_async_decorator_detaches_context_on_exception() -> None:
    @using_session("session-err")
    async def fail_inside_session() -> None:
        assert get_value(SpanAttributes.SESSION_ID) == "session-err"
        raise ValueError("simulated error")

    with pytest.raises(ValueError, match="simulated error"):
        await fail_inside_session()

    assert get_value(SpanAttributes.SESSION_ID) is None


async def test_async_decorators_can_be_stacked() -> None:
    @using_session("session-stacked")
    @using_user("user-stacked")
    async def read_session_and_user() -> Tuple[Any, Any]:
        await asyncio.sleep(0.001)
        return get_value(SpanAttributes.SESSION_ID), get_value(SpanAttributes.USER_ID)

    observed_session_id, observed_user_id = await read_session_and_user()
    assert observed_session_id == "session-stacked"
    assert observed_user_id == "user-stacked"
    assert get_value(SpanAttributes.SESSION_ID) is None
    assert get_value(SpanAttributes.USER_ID) is None


async def test_async_generator_decorator_attaches_context_while_body_runs() -> None:
    @using_session("stream-session")
    async def stream_session_ids(item_count: int) -> AsyncIterator[Any]:
        for _ in range(item_count):
            await asyncio.sleep(0.001)
            yield get_value(SpanAttributes.SESSION_ID)

    observed_session_ids = []
    async for session_id_inside_generator in stream_session_ids(3):
        observed_session_ids.append(session_id_inside_generator)
        assert get_value(SpanAttributes.SESSION_ID) is None
    assert observed_session_ids == ["stream-session"] * 3
    assert get_value(SpanAttributes.SESSION_ID) is None


async def test_async_generator_decorator_survives_early_break() -> None:
    session_ids_seen_in_cleanup: List[Any] = []

    @using_session("abandoned-stream-session")
    async def stream_forever() -> AsyncGenerator[str, None]:
        try:
            while True:
                yield str(get_value(SpanAttributes.SESSION_ID))
        finally:
            session_ids_seen_in_cleanup.append(get_value(SpanAttributes.SESSION_ID))

    stream = stream_forever()
    async for session_id_inside_generator in stream:
        assert session_id_inside_generator == "abandoned-stream-session"
        break
    assert get_value(SpanAttributes.SESSION_ID) is None
    await stream.aclose()
    assert session_ids_seen_in_cleanup == ["abandoned-stream-session"]
    assert get_value(SpanAttributes.SESSION_ID) is None


async def test_async_generator_decorator_keeps_generator_context_across_yields(
    tracer: OITracer,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    @using_session("stream-session")
    async def stream_with_parent_span(item_count: int) -> AsyncIterator[int]:
        with tracer.start_as_current_span("parent"):
            for item in range(item_count):
                with tracer.start_as_current_span(f"child-{item}"):
                    pass
                yield item

    with tracer.start_as_current_span("consumer"):
        async for _ in stream_with_parent_span(3):
            assert get_value(SpanAttributes.SESSION_ID) is None
            assert getattr(trace_api.get_current_span(), "name", None) == "consumer"

    spans = {span.name: span for span in in_memory_span_exporter.get_finished_spans()}
    parent_span_id = spans["parent"].context.span_id
    for item in range(3):
        child_span = spans[f"child-{item}"]
        assert child_span.parent is not None
        assert child_span.parent.span_id == parent_span_id
        assert child_span.attributes is not None
        assert child_span.attributes[SpanAttributes.SESSION_ID] == "stream-session"
    assert spans["consumer"].attributes is not None
    assert SpanAttributes.SESSION_ID not in spans["consumer"].attributes


async def test_async_generator_decorator_forwards_asend_and_athrow() -> None:
    @using_session("bidirectional-session")
    async def echo() -> AsyncGenerator[str, Any]:
        received = yield "ready"
        while True:
            try:
                received = yield f"got {received!r} in {get_value(SpanAttributes.SESSION_ID)}"
            except KeyError as error:
                received = yield f"handled {error.args[0]!r}"

    stream = echo()
    assert await stream.__anext__() == "ready"
    assert await stream.asend(42) == "got 42 in bidirectional-session"
    assert await stream.athrow(KeyError("missing")) == "handled 'missing'"
    assert await stream.asend(None) == "got None in bidirectional-session"
    await stream.aclose()
    assert get_value(SpanAttributes.SESSION_ID) is None


def test_generator_decorator_attaches_context_while_body_runs() -> None:
    @using_session("sync-stream-session")
    def stream_session_ids(item_count: int) -> Iterator[Any]:
        for _ in range(item_count):
            yield get_value(SpanAttributes.SESSION_ID)

    observed_session_ids = []
    for session_id_inside_generator in stream_session_ids(3):
        observed_session_ids.append(session_id_inside_generator)
        assert get_value(SpanAttributes.SESSION_ID) is None
    assert observed_session_ids == ["sync-stream-session"] * 3
    assert get_value(SpanAttributes.SESSION_ID) is None


def test_generator_decorator_forwards_send_throw_and_return_value() -> None:
    @using_session("sync-bidirectional-session")
    def echo() -> Generator[str, Any, str]:
        received = yield "ready"
        try:
            yield f"got {received!r} in {get_value(SpanAttributes.SESSION_ID)}"
        except KeyError as error:
            yield f"handled {error.args[0]!r}"
        return "done"

    stream = echo()
    assert next(stream) == "ready"
    assert stream.send(42) == "got 42 in sync-bidirectional-session"
    assert stream.throw(KeyError("missing")) == "handled 'missing'"
    with pytest.raises(StopIteration) as stop:
        next(stream)
    assert stop.value.value == "done"
    assert get_value(SpanAttributes.SESSION_ID) is None


def test_generator_decorator_sees_live_contextvar_between_yields() -> None:
    _live.set(0)

    @using_session("s")
    def stream() -> Generator[int, None, None]:
        assert _live.get() == 0
        yield 1
        assert _live.get() == 2

    gen = stream()
    assert next(gen) == 1
    _live.set(2)
    with pytest.raises(StopIteration):
        next(gen)


async def test_async_generator_decorator_sees_live_contextvar_between_yields() -> None:
    _live.set(0)

    @using_session("s")
    async def stream() -> AsyncGenerator[int, None]:
        assert _live.get() == 0
        yield 1
        assert _live.get() == 2

    gen = stream()
    assert await gen.__anext__() == 1
    _live.set(2)
    with pytest.raises(StopAsyncIteration):
        await gen.__anext__()
    await gen.aclose()


def test_generator_decorator_keeps_body_owned_context_var_across_yields() -> None:
    @using_session("s")
    def stream() -> Generator[int, None, None]:
        _body_owned.set("body-value")
        yield 1
        assert _body_owned.get() == "body-value"
        yield 2

    gen = stream()
    assert next(gen) == 1
    _live.set(99)
    assert next(gen) == 2


def test_generator_decorator_ignores_consumer_span_between_yields(
    tracer: OITracer,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    @using_session("stream-session")
    def stream_with_spans() -> Generator[int, None, None]:
        with tracer.start_as_current_span("parent"):
            yield 0
            with tracer.start_as_current_span("child-after-resume"):
                pass
            yield 1

    gen = stream_with_spans()
    assert next(gen) == 0
    with tracer.start_as_current_span("consumer-between"):
        pass
    assert next(gen) == 1
    with pytest.raises(StopIteration):
        next(gen)

    spans = {span.name: span for span in in_memory_span_exporter.get_finished_spans()}
    parent_span_id = spans["parent"].context.span_id
    child_span = spans["child-after-resume"]
    assert child_span.parent is not None
    assert child_span.parent.span_id == parent_span_id
    assert child_span.attributes is not None
    assert child_span.attributes[SpanAttributes.SESSION_ID] == "stream-session"


def test_abandoned_generator_cleanup_runs_in_its_own_scope() -> None:
    session_ids_seen_in_cleanup: List[Any] = []

    @using_session("abandoned-sync-session")
    def stream_forever() -> Generator[str, None, None]:
        try:
            while True:
                yield str(get_value(SpanAttributes.SESSION_ID))
        finally:
            session_ids_seen_in_cleanup.append(get_value(SpanAttributes.SESSION_ID))

    stream = stream_forever()
    next(stream)
    del stream
    gc.collect()
    assert session_ids_seen_in_cleanup == ["abandoned-sync-session"]


async def test_decorator_stacked_on_tracer_decorator_puts_attributes_on_span(
    tracer: OITracer,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    @using_session("agent-session")
    @tracer.agent
    async def run_agent() -> None:
        await asyncio.sleep(0.001)

    await run_agent()
    (span,) = in_memory_span_exporter.get_finished_spans()
    assert span.attributes is not None
    assert span.attributes[SpanAttributes.SESSION_ID] == "agent-session"


def test_get_attributes_from_context(
    session_id: str,
    user_id: str,
    metadata: Dict[str, Any],
    tags: List[str],
    prompt_template: str,
    prompt_template_version: str,
    prompt_template_variables: Dict[str, Any],
) -> None:
    with using_attributes(
        session_id=session_id,
        user_id=user_id,
        metadata=metadata,
        tags=tags,
        prompt_template=prompt_template,
        prompt_template_version=prompt_template_version,
        prompt_template_variables=prompt_template_variables,
    ):
        ctx = get_current()
        context_vars = {attr[0]: attr[1] for attr in get_attributes_from_context()}
        assert len(ctx) == len(context_vars)
        for key, value in ctx.items():
            assert context_vars.pop(key, None) == value, f"Missing context variable {key}"

    context_vars = {attr[0]: attr[1] for attr in get_attributes_from_context()}
    assert context_vars == {}


def test_safe_json_dumps_encodes_non_serializable_objects() -> None:
    non_serializable_object = object()
    assert safe_json_dumps(non_serializable_object) == safe_json_dumps(str(non_serializable_object))


def test_safe_json_dumps_encodes_non_ascii_characters_without_escaping() -> None:
    assert (
        safe_json_dumps({"naïve façade café": "안녕하세요"})
        == '{"naïve façade café": "안녕하세요"}'
    )


def test_capture_span_context() -> None:
    tracer = TracerProvider().get_tracer("test_capture_span_context")
    with capture_span_context() as capture:
        assert capture.get_last_span_id() is None
        assert capture.get_first_span_id() is None
        assert capture.get_span_contexts() == []
        span1 = tracer.start_span("span1")
        assert capture.get_last_span_id() == f"{span1.get_span_context().span_id:016x}"
        assert capture.get_first_span_id() == f"{span1.get_span_context().span_id:016x}"
        assert capture.get_span_contexts() == [span1.get_span_context()]
        span2 = tracer.start_span("span2")
        assert span1.get_span_context() != span2.get_span_context()
        assert capture.get_last_span_id() == f"{span2.get_span_context().span_id:016x}"
        assert capture.get_first_span_id() == f"{span1.get_span_context().span_id:016x}"
        assert capture.get_span_contexts() == [span1.get_span_context(), span2.get_span_context()]
        cast(list[SpanContext], capture.get_span_contexts()).append(INVALID_SPAN_CONTEXT)
        assert capture.get_span_contexts() == [span1.get_span_context(), span2.get_span_context()]

    assert capture.get_last_span_id() is None
    assert capture.get_span_contexts() == []


@pytest.fixture
def session_id() -> str:
    return "test-session"


@pytest.fixture
def user_id() -> str:
    return "test-user"


@pytest.fixture
def metadata() -> Dict[str, Any]:
    return {
        "key-int": 1,
        "key-str": "2",
        "key-list": [1, 2, 3],
    }


@pytest.fixture
def tags() -> List[str]:
    return ["tag_1", "tag_2"]


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

import asyncio
import json
from typing import Any, Dict, List, Tuple, cast

import pytest
from opentelemetry.context import (
    _SUPPRESS_INSTRUMENTATION_KEY,
    get_current,
    get_value,
)
from opentelemetry.trace import INVALID_SPAN_CONTEXT, SpanContext

from openinference.instrumentation import (
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


@pytest.mark.parametrize(
    "decorator,expected_attributes",
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
        pytest.param(
            using_attributes(
                session_id="test-session",
                user_id="test-user",
                metadata={"key": "value"},
                tags=["tag-1", "tag-2"],
            ),
            {
                SpanAttributes.SESSION_ID: "test-session",
                SpanAttributes.USER_ID: "test-user",
                SpanAttributes.METADATA: json.dumps({"key": "value"}),
                SpanAttributes.TAG_TAGS: ["tag-1", "tag-2"],
            },
            id="using_attributes",
        ),
    ],
)
async def test_async_decorator_attaches_attributes_across_awaits(
    decorator: _UsingAttributesContextManager, expected_attributes: Dict[str, Any]
) -> None:
    @decorator
    async def read_attributes() -> Dict[str, Any]:
        assert {key: get_value(key) for key in expected_attributes} == expected_attributes
        await asyncio.sleep(0)
        return {key: get_value(key) for key in expected_attributes}

    assert await read_attributes() == expected_attributes
    assert await read_attributes() == expected_attributes
    for key in expected_attributes:
        assert get_value(key) is None


async def test_async_decorator_isolates_concurrent_calls() -> None:
    first_entered = asyncio.Event()
    second_entered = asyncio.Event()

    @using_session("shared-session")
    async def read_session(call: int) -> Any:
        if call == 1:
            first_entered.set()
            await second_entered.wait()
        else:
            await first_entered.wait()
            second_entered.set()
        return get_value(SpanAttributes.SESSION_ID)

    assert list(await asyncio.gather(read_session(1), read_session(2))) == [
        "shared-session",
        "shared-session",
    ]
    assert get_value(SpanAttributes.SESSION_ID) is None


async def test_async_decorator_detaches_context_on_exception() -> None:
    @using_session("test-session")
    async def fail() -> None:
        raise ValueError("test error")

    with pytest.raises(ValueError, match="test error"):
        await fail()

    assert get_value(SpanAttributes.SESSION_ID) is None


async def test_async_decorators_can_be_stacked() -> None:
    @using_session("test-session")
    @using_user("test-user")
    async def read_attributes() -> Tuple[Any, Any]:
        await asyncio.sleep(0)
        return (
            get_value(SpanAttributes.SESSION_ID),
            get_value(SpanAttributes.USER_ID),
        )

    assert await read_attributes() == ("test-session", "test-user")
    assert get_value(SpanAttributes.SESSION_ID) is None
    assert get_value(SpanAttributes.USER_ID) is None


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

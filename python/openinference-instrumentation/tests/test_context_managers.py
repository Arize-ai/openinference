import json
from typing import Any, Dict, List

import pytest
from openinference.instrumentation import (
    suppress_tracing,
    using_attributes,
    using_metadata,
    using_session,
    using_tags,
    using_user,
)
from openinference.semconv.trace import SpanAttributes
from opentelemetry.context import (
    _SUPPRESS_INSTRUMENTATION_KEY,
    get_value,
)


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
        assert get_value(SpanAttributes.METADATA) == json.dumps(metadata, default=str)
    assert get_value(SpanAttributes.METADATA) is None


def test_using_tags(tags: List[str]) -> None:
    with using_tags(tags):
        assert get_value(SpanAttributes.TAG_TAGS) == tags
    assert get_value(SpanAttributes.TAG_TAGS) is None


def test_using_attributes(
    session_id: str, user_id: str, metadata: Dict[str, Any], tags: List[str]
) -> None:
    with using_attributes(
        session_id=session_id,
        user_id=user_id,
        metadata=metadata,
        tags=tags,
    ):
        assert get_value(SpanAttributes.SESSION_ID) == session_id
        assert get_value(SpanAttributes.USER_ID) == user_id
        assert get_value(SpanAttributes.METADATA) == json.dumps(metadata, default=str)
        assert get_value(SpanAttributes.TAG_TAGS) == tags
    assert get_value(SpanAttributes.SESSION_ID) is None
    assert get_value(SpanAttributes.USER_ID) is None
    assert get_value(SpanAttributes.METADATA) is None
    assert get_value(SpanAttributes.TAG_TAGS) is None


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
        assert get_value(SpanAttributes.METADATA) == json.dumps(metadata, default=str)

    f()
    assert get_value(SpanAttributes.METADATA) is None


def test_using_tags_decorator(tags: List[str]) -> None:
    @using_tags(tags)
    def f() -> None:
        assert get_value(SpanAttributes.TAG_TAGS) == tags

    f()
    assert get_value(SpanAttributes.TAG_TAGS) is None


def test_using_attributes_decorator(
    session_id: str, user_id: str, metadata: Dict[str, Any], tags: List[str]
) -> None:
    @using_attributes(session_id=session_id, user_id=user_id, metadata=metadata, tags=tags)
    def f() -> None:
        assert get_value(SpanAttributes.SESSION_ID) == session_id
        assert get_value(SpanAttributes.USER_ID) == user_id
        assert get_value(SpanAttributes.METADATA) == json.dumps(metadata, default=str)
        assert get_value(SpanAttributes.TAG_TAGS) == tags

    f()
    assert get_value(SpanAttributes.SESSION_ID) is None
    assert get_value(SpanAttributes.USER_ID) is None
    assert get_value(SpanAttributes.METADATA) is None
    assert get_value(SpanAttributes.TAG_TAGS) is None


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

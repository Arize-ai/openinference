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


def test_using_session() -> None:
    with using_session("test-session"):
        assert get_value(SpanAttributes.SESSION_ID) == "test-session"
    assert get_value(SpanAttributes.SESSION_ID) is None


def test_using_user() -> None:
    with using_user("test-user"):
        assert get_value(SpanAttributes.USER_ID) == "test-user"
    assert get_value(SpanAttributes.USER_ID) is None


def test_using_metadata(metadata_fixture: Dict[str, Any]) -> None:
    with using_metadata(metadata_fixture):
        assert get_value(SpanAttributes.METADATA) == json.dumps(metadata_fixture, default=str)
    assert get_value(SpanAttributes.METADATA) is None


def test_using_tags(tags_fixture: List[str]) -> None:
    with using_tags(tags_fixture):
        assert get_value(SpanAttributes.TAG_TAGS) == tags_fixture
    assert get_value(SpanAttributes.TAG_TAGS) is None


def test_using_attributes(metadata_fixture: Dict[str, Any], tags_fixture: List[str]) -> None:
    with using_attributes(
        session_id="test-session",
        user_id="test-user",
        metadata=metadata_fixture,
        tags=tags_fixture,
    ):
        assert get_value(SpanAttributes.SESSION_ID) == "test-session"
        assert get_value(SpanAttributes.USER_ID) == "test-user"
        assert get_value(SpanAttributes.METADATA) == json.dumps(metadata_fixture, default=str)
        assert get_value(SpanAttributes.TAG_TAGS) == tags_fixture
    assert get_value(SpanAttributes.SESSION_ID) is None
    assert get_value(SpanAttributes.USER_ID) is None
    assert get_value(SpanAttributes.METADATA) is None
    assert get_value(SpanAttributes.TAG_TAGS) is None


# def test_combined_context_managers(
#     metadata_fixture: Dict[str, Any], tags_fixture: List[str]
# ) -> None:
#     with (
#         using_session("test-session"),
#         using_user("test-user"),
#         using_metadata(metadata_fixture),
#         using_tags(tags_fixture),
#     ):
#         assert get_value(SpanAttributes.SESSION_ID) == "test-session"
#         assert get_value(SpanAttributes.USER_ID) == "test-user"
#         assert get_value(SpanAttributes.METADATA) == json.dumps(metadata_fixture, default=str)
#         assert get_value(SpanAttributes.TAG_TAGS) == tags_fixture
#     assert get_value(SpanAttributes.SESSION_ID) is None
#     assert get_value(SpanAttributes.USER_ID) is None
#     assert get_value(SpanAttributes.METADATA) is None
#     assert get_value(SpanAttributes.TAG_TAGS) is None


@pytest.fixture
def tags_fixture() -> List[str]:
    return ["tag_1", "tag_2"]


@pytest.fixture
def metadata_fixture() -> Dict[str, Any]:
    return {
        "key-int": 1,
        "key-str": "2",
        "key-list": [1, 2, 3],
    }

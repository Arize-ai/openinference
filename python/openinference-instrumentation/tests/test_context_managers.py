import json
from typing import Any, Dict, List

import pytest
from opentelemetry.context import (
    _SUPPRESS_INSTRUMENTATION_KEY,
    get_current,
    get_value,
)

from openinference.instrumentation import (
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

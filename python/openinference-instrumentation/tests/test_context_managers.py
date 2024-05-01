from typing import Iterator

import pytest
from openinference.instrumentation import (
    suppress_tracing,
    using_session,
    using_user,
    using_metadata,
    using_tags,
    using_attributes,
)
from opentelemetry.context import (
    _SUPPRESS_INSTRUMENTATION_KEY,
    attach,
    detach,
    get_value,
    set_value,
)


def test_suppress_tracing(obj: object) -> None:
    with suppress_tracing():
        assert get_value(_SUPPRESS_INSTRUMENTATION_KEY) is True
    assert get_value(_SUPPRESS_INSTRUMENTATION_KEY) is obj


@pytest.fixture(autouse=True)
def instrument(obj: object) -> Iterator[None]:
    token = attach(set_value(_SUPPRESS_INSTRUMENTATION_KEY, obj))
    yield
    detach(token)


@pytest.fixture
def obj() -> object:
    return object()

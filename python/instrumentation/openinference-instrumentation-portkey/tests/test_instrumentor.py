import asyncio
from typing import Any, Dict, List, Mapping, Optional, Type, Union, cast

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.util._importlib_metadata import entry_points
from opentelemetry.util.types import AttributeValue

from openinference.instrumentation import OITracer, using_attributes
from openinference.instrumentation.portkey import PortkeyInstrumentor
from openinference.semconv.trace import MessageAttributes, SpanAttributes

# Mock response for testing
class MockResponse:
    def __init__(self):
        self.choices = [
            {
                "message": {
                    "role": "assistant",
                    "content": "This is a test response",
                }
            }
        ]
        self.usage = {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        }


# Mock function for testing
def _mock_function(*args, **kwargs):
    return MockResponse()


async def _async_mock_function(*args, **kwargs):
    return MockResponse()


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


class TestInstrumentor:
    def test_entrypoint_for_opentelemetry_instrument(self) -> None:
        (instrumentor_entrypoint,) = entry_points(group="opentelemetry_instrumentor", name="portkey")
        instrumentor = instrumentor_entrypoint.load()()
        assert isinstance(instrumentor, PortkeyInstrumentor)

    # Ensure we're using the common OITracer from common openinference-instrumentation pkg
    def test_oitracer(self) -> None:
        assert isinstance(PortkeyInstrumentor()._tracer, OITracer)


# Constants for attribute checking
SESSION_ID = SpanAttributes.SESSION_ID
USER_ID = SpanAttributes.USER_ID
METADATA = SpanAttributes.METADATA
TAG_TAGS = SpanAttributes.TAG_TAGS
LLM_PROMPT_TEMPLATE = SpanAttributes.LLM_PROMPT_TEMPLATE
LLM_PROMPT_TEMPLATE_VERSION = SpanAttributes.LLM_PROMPT_TEMPLATE_VERSION
LLM_PROMPT_TEMPLATE_VARIABLES = SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES 
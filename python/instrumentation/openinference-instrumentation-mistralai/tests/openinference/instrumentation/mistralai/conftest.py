from importlib.util import find_spec
from typing import Any, Dict, Generator, List

import pytest
from mistralai.client import Mistral
from opentelemetry import trace as trace_api
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation.mistralai import MistralAIInstrumentor

if find_spec("httpcore2") is not None:
    # mistralai>=3.0 performs HTTP through httpx2/httpcore2, forks of httpx/httpcore
    # with the same transport interface. respx only patches httpcore, so without
    # this the respx-mocked tests would send real requests under the newer SDK.
    # Teach respx's default mocker about the httpcore2 transports as well; the
    # request/response objects have the same shape, so its conversions work for
    # both. (vcrpy already patches httpx2, so the cassette tests need nothing.)
    from respx.mocks import HTTPCoreMocker

    HTTPCoreMocker.add_targets(
        *[
            target.replace("httpcore.", "httpcore2.", 1)
            for target in HTTPCoreMocker.targets
            if target.startswith("httpcore.")
        ]
    )


def _strip_request_headers(request: Any) -> Any:
    request.headers.clear()
    return request


def _strip_response_headers(response: Any) -> Any:
    # Preserve Content-Type so streaming SDKs can identify SSE responses
    content_type = response.get("headers", {}).get("Content-Type")
    headers = {"Content-Type": content_type} if content_type else {}
    return {**response, "headers": headers}


@pytest.fixture(scope="session")
def vcr_config() -> dict[str, Any]:
    return {
        "before_record_request": _strip_request_headers,
        "before_record_response": _strip_response_headers,
        "decode_compressed_response": True,
        "record_mode": "once",
    }


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


@pytest.fixture(scope="module")
def mistral_sync_client() -> Mistral:
    return Mistral(api_key="123")


@pytest.fixture(scope="module")
def in_memory_span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture(scope="module")
def tracer_provider(in_memory_span_exporter: InMemorySpanExporter) -> trace_api.TracerProvider:
    resource = Resource(attributes={})
    tracer_provider = trace_sdk.TracerProvider(resource=resource)
    span_processor = SimpleSpanProcessor(span_exporter=in_memory_span_exporter)
    tracer_provider.add_span_processor(span_processor=span_processor)
    return tracer_provider


@pytest.fixture(autouse=True)
def instrument(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> Generator[None, None, None]:
    MistralAIInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    MistralAIInstrumentor().uninstrument()
    in_memory_span_exporter.clear()

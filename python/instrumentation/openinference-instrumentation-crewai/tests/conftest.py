import os
from pathlib import Path
from typing import Any, Generator

import pytest
from opentelemetry import trace as trace_api
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation.crewai import CrewAIInstrumentor

_TEST_HOME = Path("/tmp/openinference-crewai-test-home")
_TEST_HOME.mkdir(parents=True, exist_ok=True)
# Keep CrewAI's SQLite/task-output state inside a writable temp home for tests.
os.environ["HOME"] = str(_TEST_HOME)
os.environ["CREWAI_STORAGE_DIR"] = "openinference-crewai-tests"
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["CREWAI_TESTING"] = "true"


def _strip_request_headers(request: Any) -> Any:
    request.headers.clear()
    return request


def _strip_response_headers(response: Any) -> Any:
    return {**response, "headers": {}}


@pytest.fixture(scope="session")
def vcr_config() -> dict[str, Any]:
    return {
        "before_record_request": _strip_request_headers,
        "before_record_response": _strip_response_headers,
        "decode_compressed_response": True,
        "record_mode": "once",
    }


@pytest.fixture(scope="session")
def in_memory_span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture(scope="session")
def tracer_provider(
    in_memory_span_exporter: InMemorySpanExporter,
) -> trace_api.TracerProvider:
    tracer_provider = trace_sdk.TracerProvider()
    span_processor = SimpleSpanProcessor(span_exporter=in_memory_span_exporter)
    tracer_provider.add_span_processor(span_processor=span_processor)
    return tracer_provider


@pytest.fixture(autouse=True)
def instrument(
    request: pytest.FixtureRequest,
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> Generator[None, None, None]:
    # Event-listener and assembler tests manage instrumentation explicitly.
    if request.node.get_closest_marker("no_autoinstrument"):
        in_memory_span_exporter.clear()
        yield
        in_memory_span_exporter.clear()
        return

    CrewAIInstrumentor().instrument(tracer_provider=tracer_provider)
    in_memory_span_exporter.clear()
    yield
    CrewAIInstrumentor().uninstrument()
    in_memory_span_exporter.clear()

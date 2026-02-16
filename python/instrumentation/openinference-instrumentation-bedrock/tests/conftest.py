import importlib.util
import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest
import yaml  # type: ignore[import-untyped]
from aioresponses import aioresponses
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation.bedrock import BedrockInstrumentor


@pytest.fixture(scope="function")
def tracer_provider(
    in_memory_span_exporter: InMemorySpanExporter,
) -> TracerProvider:
    tracer_provider = TracerProvider()
    span_processor = SimpleSpanProcessor(span_exporter=in_memory_span_exporter)
    tracer_provider.add_span_processor(span_processor=span_processor)
    # from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    #
    # tracer_provider.add_span_processor(
    #     SimpleSpanProcessor(OTLPSpanExporter("http://127.0.0.1:4317"))
    # )
    return tracer_provider


@pytest.fixture(scope="function")
def in_memory_span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture(autouse=True)
def instrument(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> Iterator[None]:
    BedrockInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    BedrockInstrumentor().uninstrument()


@pytest.fixture(scope="function")
def read_aio_cassette() -> Callable[..., Any]:
    def mock_from_cassette(cassette_path: str, aioresponses_mock: aioresponses) -> None:
        """
        Load a VCR cassette YAML file and configure aioresponses.
        Supports JSON responses and EventStream binary data.
        """
        cassette_file = Path(cassette_path)
        if not cassette_file.exists():
            raise FileNotFoundError(f"Cassette file not found: {cassette_path}")

        with open(cassette_file, "r") as f:
            cassette_data = yaml.safe_load(f)

        for interaction in cassette_data.get("interactions", []):
            request = interaction["request"]
            response = interaction["response"]

            # Extract response body
            body = response.get("body", {})
            response_body = body.get("string", "") if isinstance(body, dict) else body

            # Convert headers (list format to single value)
            headers = {
                k: v[0] if isinstance(v, list) else v
                for k, v in response.get("headers", {}).items()
            }

            # Determine response type
            is_binary = isinstance(response_body, bytes)
            is_event_stream = "event-stream" in headers.get("Content-Type", "").lower()

            # Parse JSON for non-binary responses
            payload = None
            if not is_binary and not is_event_stream and response_body:
                try:
                    payload = (
                        json.loads(response_body)
                        if isinstance(response_body, str)
                        else response_body
                    )
                except (json.JSONDecodeError, TypeError):
                    pass

            # Handle regex in URI
            uri = request["uri"]
            url = re.compile(uri) if any(c in uri for c in r"()[]{}*+?") else uri

            # Register mock
            method = request.get("method", "GET").upper()
            method_map = {
                "GET": aioresponses_mock.get,
                "POST": aioresponses_mock.post,
                "PUT": aioresponses_mock.put,
                "DELETE": aioresponses_mock.delete,
                "PATCH": aioresponses_mock.patch,
            }

            method_map[method](
                url=url,
                status=response["status"]["code"],
                headers=headers,
                payload=payload if payload and not is_binary and not is_event_stream else None,
                body=response_body if is_binary or is_event_stream or not payload else None,
            )

    return mock_from_cassette


def pytest_configure(config: Any) -> Any:
    config.addinivalue_line("markers", "aio: aioboto3-only tests")


def pytest_runtest_setup(item: Any) -> Any:
    if "aio" in item.keywords:
        if importlib.util.find_spec("aioboto3") is None:
            pytest.skip("aioboto3 is not installed")


def pytest_collection_modifyitems(config: Any, items: Any) -> None:
    aio_enabled = os.getenv("OPENINFERENCE_TEST_AIO") == "1"

    skip_aio = pytest.mark.skip(reason="aioboto3 tests only run in bedrock-aio env")

    for item in items:
        if "aio" in item.keywords and not aio_enabled:
            item.add_marker(skip_aio)

import re
from pathlib import Path
from typing import Any, Callable, Iterator, Tuple

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


@pytest.fixture()
def image_bytes_and_format() -> Tuple[bytes, str]:
    """Minimal image (bytes, format) for tests; no network."""
    return (
        b"GIF89a\x01\x00\x01\x00\x80\x00\x00\xff\xff\xff\x00\x00\x00!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;",  # noqa: E501
        "webp",
    )


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
        Supports JSON (replayed as raw body to preserve Content-Length), binary, and EventStream.
        """
        cassette_file = Path(cassette_path)
        if not cassette_file.exists():
            raise FileNotFoundError(f"Cassette file not found: {cassette_path}")

        with open(cassette_file, "r") as f:
            cassette_data = yaml.safe_load(f)

        for interaction in cassette_data.get("interactions", []):
            request = interaction["request"]
            response = interaction["response"]

            body_spec = response.get("body", {})
            response_body = (
                body_spec.get("string", "") if isinstance(body_spec, dict) else body_spec
            )

            headers = {
                k: v[0] if isinstance(v, list) else v
                for k, v in response.get("headers", {}).items()
            }

            is_binary = isinstance(response_body, bytes)
            is_event_stream = "event-stream" in headers.get("Content-Type", "").lower()
            request_uri = request.get("uri", "")
            if is_binary and not is_event_stream:
                uri_lower = request_uri.lower()
                if (
                    "converse-stream" in uri_lower
                    or "retrieveandgeneratestream" in uri_lower
                    or ("/agents/" in uri_lower and "/text" in uri_lower)
                ):
                    is_event_stream = True
                    headers = {**headers, "Content-Type": "application/vnd.amazon.eventstream"}

            # Replay body as-is (no JSON parse/reserialize) so Content-Length matches.
            if any(c in request_uri for c in r"()[]{}*+?"):
                url = re.compile(request_uri)
            elif "%3A" in request_uri:
                url = re.compile(re.escape(request_uri).replace(re.escape("%3A"), "(?:%3A|:)"))
            else:
                url = request_uri

            method = request.get("method", "GET").upper()
            method_map = {
                "GET": aioresponses_mock.get,
                "POST": aioresponses_mock.post,
                "PUT": aioresponses_mock.put,
                "DELETE": aioresponses_mock.delete,
                "PATCH": aioresponses_mock.patch,
            }
            call = method_map[method]
            call(
                url=url,
                status=response["status"]["code"],
                headers=headers,
                body=response_body,
            )

    return mock_from_cassette

from typing import Any, Generator

import pytest
from autogen import ConversableAgent  # type: ignore
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.util._importlib_metadata import entry_points

from openinference.instrumentation.autogen import AutogenInstrumentor
from openinference.semconv.trace import SpanAttributes


@pytest.fixture()
def in_memory_span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture()
def tracer_provider(
    in_memory_span_exporter: InMemorySpanExporter,
) -> TracerProvider:
    resource = Resource(attributes={})
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))
    return tracer_provider


@pytest.fixture()
def setup_autogen_instrumentation(
    tracer_provider: TracerProvider,
) -> Generator[None, None, None]:
    trace.set_tracer_provider(tracer_provider)
    AutogenInstrumentor().instrument()
    yield
    AutogenInstrumentor().uninstrument()


@pytest.fixture
def openai_api_key(monkeypatch: pytest.MonkeyPatch) -> str:
    api_key = "sk-0123456789"
    monkeypatch.setenv("OPENAI_API_KEY", api_key)
    return api_key


class TestInstrumentor:
    def test_entrypoint_for_opentelemetry_instrument(self) -> None:
        (instrumentor_entrypoint,) = entry_points(  # type: ignore[no-untyped-call]
            group="opentelemetry_instrumentor", name="autogen"
        )
        instrumentor = instrumentor_entrypoint.load()()
        assert isinstance(instrumentor, AutogenInstrumentor)


@pytest.mark.asyncio
async def test_autogen_generate_reply_instrumentation(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_autogen_instrumentation: Any,
    openai_api_key: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_create(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return {"fake": True}

    def fake_extract(self, response):  # type: ignore[no-untyped-def]
        return [{"role": "assistant", "content": "Hello from Autogen"}]

    monkeypatch.setattr(
        "autogen.oai.client.OpenAIWrapper.create",
        fake_create,
    )
    monkeypatch.setattr(
        "autogen.oai.client.OpenAIWrapper.extract_text_or_completion_object",
        fake_extract,
    )

    agent = ConversableAgent(
        name="TestAgent",
        llm_config={"model": "gpt-4o"},
    )

    agent.generate_reply(messages=[{"role": "user", "content": "Hello"}])

    spans = in_memory_span_exporter.get_finished_spans()
    assert spans, "Expected at least one span"

    span = spans[0]
    assert span.status.is_ok

    attributes = dict(span.attributes or {})

    assert span.name == "ConversableAgent"
    assert attributes.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND) == "AGENT"
    assert attributes.pop("agent.type") == "ConversableAgent"
    assert isinstance(attributes.pop(SpanAttributes.INPUT_VALUE), str)
    assert attributes.pop(SpanAttributes.INPUT_MIME_TYPE) == "application/json"
    assert isinstance(attributes.pop(SpanAttributes.OUTPUT_VALUE), str)
    assert attributes.pop(SpanAttributes.OUTPUT_MIME_TYPE) == "application/json"
    assert attributes.pop(SpanAttributes.LLM_MODEL_NAME, None) == "gpt-4o"
    assert attributes.pop(SpanAttributes.LLM_PROVIDER, None) == "openai"
    assert attributes.pop(SpanAttributes.LLM_SYSTEM, None) == "openai"
    assert not attributes

import random
from itertools import count
from typing import Any, Iterator

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation.langchain import LangChainInstrumentor


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


# Enable these fixtures to log langsmith data to ~/langsmith_data

# @pytest.fixture(scope="session")
# def _port() -> int:
#     return int(pick_unused_port())


# @pytest.fixture(autouse=True, scope="session")
# def _ls(_port: int) -> Iterator[None]:
#     values = (
#         ("LANGCHAIN_TRACING_V2", "true"),
#         ("LANGCHAIN_ENDPOINT", f"http://127.0.0.1:{_port}"),
#     )
#     with ExitStack() as stack:
#         stack.enter_context(mock.patch.dict(os.environ, values))
#         stack.enter_context(_Receiver(_port).run_in_thread())
#         yield
#         wait_for_all_tracers()


@pytest.fixture
def in_memory_span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture
def tracer_provider(in_memory_span_exporter: InMemorySpanExporter) -> TracerProvider:
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))
    return tracer_provider


@pytest.fixture(autouse=True)
def instrument(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> Iterator[None]:
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
    yield


@pytest.fixture(autouse=True)
def uninstrument() -> Iterator[None]:
    yield
    LangChainInstrumentor().uninstrument()


@pytest.fixture(autouse=True)
def openai_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-")


@pytest.fixture(autouse=True)
def force_legacy_httpx_openai_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route the OpenAI SDK through a legacy ``httpx`` transport so ``respx`` can mock it.

    ``openai>=3`` defaults its HTTP transport to ``httpx2`` (an API-identical fork of
    ``httpx``). ``respx`` only patches ``httpx``, so on ``openai>=3`` the ``respx_mock``
    routes are bypassed and requests hit the real API (surfacing as a 401). The OpenAI
    SDK still accepts a legacy ``httpx`` client, so we inject one for every client that
    ``langchain_openai`` constructs. This is a no-op on ``openai<3`` (already ``httpx``).
    """
    import openai

    if int(openai.__version__.split(".")[0]) < 3:
        return

    import httpx

    real_sync = openai.OpenAI
    real_async = openai.AsyncOpenAI

    def sync_client(*args: Any, **kwargs: Any) -> Any:
        kwargs["http_client"] = httpx.Client()
        return real_sync(*args, **kwargs)

    def async_client(*args: Any, **kwargs: Any) -> Any:
        kwargs["http_client"] = httpx.AsyncClient()
        return real_async(*args, **kwargs)

    monkeypatch.setattr(openai, "OpenAI", sync_client)
    monkeypatch.setattr(openai, "AsyncOpenAI", async_client)


@pytest.fixture(scope="module")
def seed() -> Iterator[int]:
    """
    Use rolling seeds to help debugging, because the rolling pseudo-random values
    allow conditional breakpoints to be hit precisely (and repeatably).
    """
    return count()


@pytest.fixture(autouse=True)
def set_seed(seed: Iterator[int]) -> Iterator[None]:
    random.seed(next(seed))
    yield

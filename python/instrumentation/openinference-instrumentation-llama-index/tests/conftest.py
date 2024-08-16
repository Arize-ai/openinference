import asyncio
import logging
import random
from asyncio.events import BaseDefaultEventLoopPolicy
from itertools import count
from typing import Any, Dict, Iterator

import pytest
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from openinference.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import TracerProvider

for name, logger in logging.root.manager.loggerDict.items():
    if name.startswith("openinference.") and isinstance(logger, logging.Logger):
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()
        logger.addHandler(logging.StreamHandler())


@pytest.fixture
def in_memory_span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture
def tracer_provider(
    in_memory_span_exporter: InMemorySpanExporter,
) -> TracerProvider:
    tracer_provider = trace_sdk.TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))
    return tracer_provider


@pytest.fixture(scope="session")
def event_loop_policy() -> BaseDefaultEventLoopPolicy:
    try:
        import uvloop
    except ImportError:
        return asyncio.DefaultEventLoopPolicy()
    return uvloop.EventLoopPolicy()


@pytest.fixture(scope="session")
def vcr_config() -> Dict[str, Any]:
    return dict(
        before_record_request=lambda _: _.headers.clear() or _,
        before_record_response=lambda _: {**_, "headers": {}},
        decode_compressed_response=True,
        ignore_localhost=True,
    )


@pytest.fixture(autouse=True)
def instrument(
    tracer_provider: TracerProvider,
) -> Iterator[None]:
    LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
    OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    OpenAIInstrumentor().uninstrument()
    LlamaIndexInstrumentor().uninstrument()


@pytest.fixture(autouse=True)
def api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-")


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

import asyncio
import logging
from asyncio.events import BaseDefaultEventLoopPolicy
from typing import Any, Dict

import pytest
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

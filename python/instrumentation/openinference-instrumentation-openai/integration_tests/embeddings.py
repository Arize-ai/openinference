"""
Phoenix collector should be running in the background.
"""
import asyncio
import inspect
import logging
from contextlib import suppress
from itertools import chain
from time import sleep

from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


def default_tracer_provider() -> trace_sdk.TracerProvider:
    resource = Resource(attributes={})
    tracer_provider = trace_sdk.TracerProvider(resource=resource)
    span_exporter = OTLPSpanExporter(endpoint="http://127.0.0.1:6006/v1/traces")
    span_processor = SimpleSpanProcessor(span_exporter=span_exporter)
    tracer_provider.add_span_processor(span_processor=span_processor)
    return tracer_provider


# Instrument httpx to show that it can show up as a child span.
# Note that it must be instrumented before it's imported by openai.
HTTPXClientInstrumentor().instrument()

# To instrument httpx, it must be monkey-patched before it is imported by openai, so we do it
# like this to prevent the imports from being re-formatted to the top of file.
if True:
    import openai
    from openinference.instrumentation.openai import OpenAIInstrumentor
    from openinference.semconv.trace import SpanAttributes

CLIENT = openai.AsyncOpenAI()

tracer_provider = default_tracer_provider()
in_memory_span_exporter = InMemorySpanExporter()
tracer_provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))
trace_api.set_tracer_provider(tracer_provider=tracer_provider)

OpenAIInstrumentor().instrument()

KWARGS = {
    "model": "text-embedding-ada-002",
}

for k, v in logging.root.manager.loggerDict.items():
    if k.startswith("openinference.instrumentation.openai") and isinstance(v, logging.Logger):
        v.setLevel(logging.DEBUG)
        v.handlers.clear()
        v.addHandler(logging.StreamHandler())

logger = logging.getLogger(__name__)

_EXPECTED_SPAN_COUNT = 0


def _print_span_count(kwargs):
    spans = in_memory_span_exporter.get_finished_spans()
    llm_spans = [
        span
        for span in spans
        if span.attributes.get(SpanAttributes.OPENINFERENCE_SPAN_KIND) == "EMBEDDING"
    ]
    actual = len(llm_spans)
    global _EXPECTED_SPAN_COUNT
    _EXPECTED_SPAN_COUNT += 1
    mark = "✅" if _EXPECTED_SPAN_COUNT <= actual else "❌"
    name = inspect.stack()[1][3]
    print(f"{mark} expected {_EXPECTED_SPAN_COUNT}; actual {actual}; {name}({kwargs})")


async def embeddings(**kwargs):
    try:
        with suppress(openai.BadRequestError):
            await CLIENT.embeddings.create(**{**KWARGS, **kwargs})
    except Exception:
        logger.exception(f"{inspect.stack()[0][3]}({kwargs})")
    finally:
        _print_span_count(kwargs)


async def embeddings_with_raw_response(**kwargs):
    try:
        with suppress(openai.BadRequestError):
            await CLIENT.embeddings.with_raw_response.create(**{**KWARGS, **kwargs})
    except Exception:
        logger.exception(f"{inspect.stack()[0][3]}({kwargs})")
    finally:
        _print_span_count(kwargs)


async def main(*tasks):
    await asyncio.gather(*chain(tasks))


if __name__ == "__main__":
    asyncio.run(
        main(
            embeddings(input="hello world"),
            embeddings(input="hello world", encoding_format="float"),
            embeddings(input="hello world", encoding_format="base64"),
            embeddings(input=["hello", "world"]),
            embeddings(input=["hello", "world"], encoding_format="float"),
            embeddings(input=["hello", "world"], encoding_format="base64"),
            embeddings(input=[15339, 1917]),
            embeddings(input=[15339, 1917], encoding_format="float"),
            embeddings(input=[15339, 1917], encoding_format="base64"),
            embeddings(input=[[15339], [14957]]),
            embeddings(input=[[15339], [14957]], encoding_format="float"),
            embeddings(input=[[15339], [14957]], encoding_format="base64"),
            embeddings_with_raw_response(input="hello world"),
            embeddings_with_raw_response(input="hello world", encoding_format="float"),
            embeddings_with_raw_response(input="hello world", encoding_format="base64"),
            embeddings_with_raw_response(input=["hello", "world"]),
            embeddings_with_raw_response(input=["hello", "world"], encoding_format="float"),
            embeddings_with_raw_response(input=["hello", "world"], encoding_format="base64"),
            embeddings_with_raw_response(input=[15339, 1917]),
            embeddings_with_raw_response(input=[15339, 1917], encoding_format="float"),
            embeddings_with_raw_response(input=[15339, 1917], encoding_format="base64"),
            embeddings_with_raw_response(input=[[15339], [14957]]),
            embeddings_with_raw_response(input=[[15339], [14957]], encoding_format="float"),
            embeddings_with_raw_response(input=[[15339], [14957]], encoding_format="base64"),
        )
    )
    spans = in_memory_span_exporter.get_finished_spans()
    llm_spans = [
        span
        for span in spans
        if span.attributes.get(SpanAttributes.OPENINFERENCE_SPAN_KIND) == "EMBEDDING"
    ]
    actual = len(llm_spans)
    mark = "✅" if _EXPECTED_SPAN_COUNT == actual else "❌"
    print(f"\n{mark} expected {_EXPECTED_SPAN_COUNT}; actual {actual};")
    assert _EXPECTED_SPAN_COUNT == actual
    sleep(1)  # (if applicable) let the old exporter finish sending traces

import asyncio
import inspect
import logging
from contextlib import suppress
from importlib.metadata import version
from itertools import chain

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

_OPENAI_VERSION = tuple(map(int, version("openai").split(".")[:3]))

N = 3  # iteration i = 0 results in intentional BadRequestError
HAIKU = "Write a haiku."
HAIKU_TOKENS = [8144, 264, 6520, 39342, 13]
RESUME = "Write a résumé."
RESUME_TOKENS = [8144, 264, 9517, 1264, 978, 13]
CHAT_KWARGS = {
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": HAIKU}],
    "max_tokens": 20,
    "temperature": 2,
    **(
        {
            "logprobs": True,
            "top_logprobs": 5,
        }
        if _OPENAI_VERSION >= (1, 5, 0)
        else {}
    ),
}
COMP_KWARGS = {
    "model": "gpt-3.5-turbo-instruct",
    "prompt": HAIKU,
    "max_tokens": 20,
    "temperature": 2,
    "logprobs": 5,
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
        if span.attributes.get(SpanAttributes.OPENINFERENCE_SPAN_KIND) == "LLM"
    ]
    actual = len(llm_spans)
    global _EXPECTED_SPAN_COUNT
    _EXPECTED_SPAN_COUNT += 1
    mark = "✅" if _EXPECTED_SPAN_COUNT <= actual else "❌"
    name = inspect.stack()[1][3]
    print(f"{mark} expected {_EXPECTED_SPAN_COUNT}; actual {actual}; {name}({kwargs})")


async def chat_completions(**kwargs):
    try:
        with suppress(openai.BadRequestError):
            response = await CLIENT.chat.completions.create(**{**CHAT_KWARGS, **kwargs})
            if kwargs.get("stream"):
                async for _ in response:
                    await asyncio.sleep(0.005)
    except Exception:
        logger.exception(f"{inspect.stack()[0][3]}({kwargs})")
    finally:
        _print_span_count(kwargs)


async def completions(**kwargs):
    try:
        with suppress(openai.BadRequestError):
            response = await CLIENT.completions.create(**{**COMP_KWARGS, **kwargs})
            if kwargs.get("stream"):
                async for _ in response:
                    await asyncio.sleep(0.005)
    except Exception:
        logger.exception(f"{inspect.stack()[0][3]}({kwargs})")
    finally:
        _print_span_count(kwargs)


async def chat_completions_with_raw_response(**kwargs):
    try:
        with suppress(openai.BadRequestError):
            response = await CLIENT.chat.completions.with_raw_response.create(
                **{**CHAT_KWARGS, **kwargs}
            )
            if kwargs.get("stream"):
                async for _ in response.parse():
                    await asyncio.sleep(0.005)
    except Exception:
        logger.exception(f"{inspect.stack()[0][3]}({kwargs})")
    finally:
        _print_span_count(kwargs)


async def completions_with_raw_response(**kwargs):
    try:
        with suppress(openai.BadRequestError):
            response = await CLIENT.completions.with_raw_response.create(
                **{**COMP_KWARGS, **kwargs}
            )
            if kwargs.get("stream"):
                async for _ in response.parse():
                    await asyncio.sleep(0.005)
    except Exception:
        logger.exception(f"{inspect.stack()[0][3]}({kwargs})")
    finally:
        _print_span_count(kwargs)


def tasks(n, task, **kwargs):
    return [task(n=i, **kwargs) for i in range(n)]  # i = 0 results in intentional BadRequestError


async def main(*tasks):
    await asyncio.gather(*chain.from_iterable(tasks))


if __name__ == "__main__":
    asyncio.run(
        main(
            tasks(N, completions),
            tasks(N, completions_with_raw_response),
            tasks(N, completions, stream=True),
            tasks(N, completions_with_raw_response, stream=True),
            tasks(N, completions, prompt=[HAIKU, RESUME]),
            tasks(N, completions_with_raw_response, prompt=[HAIKU, RESUME]),
            tasks(N, completions, prompt=[HAIKU, RESUME], stream=True),
            tasks(N, completions_with_raw_response, prompt=[HAIKU, RESUME], stream=True),
            tasks(N, completions, prompt=HAIKU_TOKENS),
            tasks(N, completions_with_raw_response, prompt=HAIKU_TOKENS),
            tasks(N, completions, prompt=HAIKU_TOKENS, stream=True),
            tasks(N, completions_with_raw_response, prompt=HAIKU_TOKENS, stream=True),
            tasks(N, completions, prompt=[HAIKU_TOKENS, RESUME_TOKENS]),
            tasks(N, completions_with_raw_response, prompt=[HAIKU_TOKENS, RESUME_TOKENS]),
            tasks(N, completions, prompt=[HAIKU_TOKENS, RESUME_TOKENS], stream=True),
            tasks(
                N, completions_with_raw_response, prompt=[HAIKU_TOKENS, RESUME_TOKENS], stream=True
            ),
            tasks(N, chat_completions),
            tasks(N, chat_completions_with_raw_response),
            tasks(N, chat_completions, stream=True),
            tasks(N, chat_completions_with_raw_response, stream=True),
        )
    )
    spans = in_memory_span_exporter.get_finished_spans()
    llm_spans = [
        span
        for span in spans
        if span.attributes.get(SpanAttributes.OPENINFERENCE_SPAN_KIND) == "LLM"
    ]
    actual = len(llm_spans)
    mark = "✅" if _EXPECTED_SPAN_COUNT == actual else "❌"
    print(f"\n{mark} expected {_EXPECTED_SPAN_COUNT}; actual {actual};")
    assert _EXPECTED_SPAN_COUNT == actual

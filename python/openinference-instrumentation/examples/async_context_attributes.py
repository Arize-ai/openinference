"""Async usage of OpenInference context attributes.

The context managers exported by ``openinference.instrumentation`` (``using_session``,
``using_user``, ``using_attributes``, ``suppress_tracing``) work in async code in every
shape demonstrated below:

1. ``@using_session(...)`` on an ``async def`` — attributes are attached for the whole
   awaited call, including across ``await`` suspension points.
2. ``@using_session(...)`` on an async generator — attributes stay attached for the
   entire iteration (e.g. a streaming handler), not just generator creation.
3. ``async with`` — attributes scope to the awaited block, and concurrent tasks each
   keep their own attributes without interfering.
4. ``async with suppress_tracing()`` — spans created inside are not exported.

Run a local Phoenix (https://github.com/Arize-ai/phoenix) at http://localhost:6006,
then:

    python async_context_attributes.py

and open the "async-context-attributes" project in Phoenix to see the spans, each
carrying the session/user/metadata attributes from its enclosing context.
"""

import asyncio
from typing import AsyncIterator

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation import (
    TracerProvider,
    suppress_tracing,
    using_attributes,
    using_session,
)
from openinference.semconv.resource import ResourceAttributes

endpoint = "http://localhost:6006/v1/traces"
resource = Resource(attributes={ResourceAttributes.PROJECT_NAME: "async-context-attributes"})
tracer_provider = TracerProvider(resource=resource)
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer = tracer_provider.get_tracer(__name__)


@using_session("decorated-async-session")
async def answer(question: str) -> str:
    """Shape 1: a decorated ``async def``."""
    with tracer.start_as_current_span("answer", openinference_span_kind="chain") as span:
        span.set_input(question)
        await asyncio.sleep(0.01)  # stand-in for an awaited LLM call
        reply = f"echo: {question}"
        span.set_output(reply)
        return reply


@using_session("streaming-async-session")
async def stream_tokens(prompt: str) -> AsyncIterator[str]:
    """Shape 2: a decorated async generator."""
    for index, token in enumerate(prompt.split()):
        with tracer.start_as_current_span(
            f"chunk-{index}", openinference_span_kind="chain"
        ) as span:
            span.set_input(prompt)
            await asyncio.sleep(0.01)
            span.set_output(token)
        yield token


async def handle_user(user_id: str, session_id: str, question: str) -> str:
    """Shape 3: ``async with``."""
    async with using_attributes(
        session_id=session_id,
        user_id=user_id,
        metadata={"example": "async_context_attributes"},
    ):
        with tracer.start_as_current_span("handle_user", openinference_span_kind="agent") as span:
            span.set_input(question)
            await asyncio.sleep(0.01)
            reply = f"{user_id}: {question}"
            span.set_output(reply)
            return reply


async def main() -> None:
    await answer("What is OpenInference?")

    async for _ in stream_tokens("context attributes survive streaming"):
        pass

    await asyncio.gather(
        handle_user("alice", "session-alice", "hello from alice"),
        handle_user("bob", "session-bob", "hello from bob"),
    )

    # Shape 4: this span is not exported.
    async with suppress_tracing():
        with tracer.start_as_current_span("not-exported", openinference_span_kind="chain"):
            await asyncio.sleep(0.01)

    tracer_provider.force_flush()
    print("done: open the 'async-context-attributes' project at http://localhost:6006")


if __name__ == "__main__":
    asyncio.run(main())

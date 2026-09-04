import asyncio
import os
from typing import AsyncIterator, List

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation import (
    TracerProvider,
    using_attributes,
    using_session,
    using_tags,
    using_user,
)
from openinference.semconv.resource import ResourceAttributes

PHOENIX_BASE_URL = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006")
COLLECTOR_ENDPOINT = f"{PHOENIX_BASE_URL.rstrip('/')}/v1/traces"
PROJECT_NAME = os.environ.get("PHOENIX_PROJECT_NAME", "async-context-attribute-decorators")

tracer_provider = TracerProvider(
    resource=Resource({ResourceAttributes.PROJECT_NAME: PROJECT_NAME}),
)
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(COLLECTOR_ENDPOINT)))
tracer = tracer_provider.get_tracer(__name__)


@tracer.tool
async def look_up_order(order_id: str) -> str:
    await asyncio.sleep(0.05)
    return f"order {order_id} shipped yesterday"


async def stream_answer(order_status: str) -> AsyncIterator[str]:
    for sentence in (f"Good news: {order_status}.", "Anything else I can help with?"):
        await asyncio.sleep(0.02)
        with tracer.start_as_current_span("emit_chunk", openinference_span_kind="chain") as span:
            span.set_output(sentence)
        yield sentence


async def answer_order_question(order_id: str) -> str:
    order_status = await look_up_order(order_id)
    answer_chunks: List[str] = []
    async for chunk in stream_answer(order_status):
        answer_chunks.append(chunk)
    return " ".join(answer_chunks)


@using_session("session-billing-42")
@using_user("customer-billing")
@using_tags(["billing"])
@tracer.agent
async def billing_agent(order_id: str) -> str:
    return await answer_order_question(order_id)


@using_attributes(
    session_id="session-shipping-7",
    user_id="customer-shipping",
    metadata={"region": "eu-west", "tier": "gold"},
    tags=["shipping"],
)
@tracer.agent
async def shipping_agent(order_id: str) -> str:
    return await answer_order_question(order_id)


async def main() -> None:
    billing_answer, shipping_answer = await asyncio.gather(
        billing_agent("B-42"),
        shipping_agent("S-7"),
    )
    print(f"billing agent:  {billing_answer}")
    print(f"shipping agent: {shipping_answer}")


if __name__ == "__main__":
    asyncio.run(main())

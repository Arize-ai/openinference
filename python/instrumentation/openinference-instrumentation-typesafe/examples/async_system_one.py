"""
Traces two concurrent ``AsyncTypeSafeClient.system_one`` calls, each with one Noul question.
Produces two LLM spans named ``AsyncTypeSafeClient``.

1. Run a local OTLP collector such as Phoenix: `uvx arize-phoenix serve`
2. Install dependencies: `pip install -r requirements.txt`
3. Set `TYPESAFE_API_KEY` and run this example: `python async_system_one.py`
4. View the traces at http://localhost:6006 under the `typesafe-async-system-one` project.
"""

import asyncio

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from typesafe_sdk import AsyncTypeSafeClient, Noul

from openinference.instrumentation.typesafe import TypeSafeAIInstrumentor
from openinference.semconv.resource import ResourceAttributes

endpoint = "http://localhost:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider(
    resource=Resource({ResourceAttributes.PROJECT_NAME: "typesafe-async-system-one"})
)
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
TypeSafeAIInstrumentor().instrument(tracer_provider=tracer_provider)


async def main() -> None:
    question = {"is_bug_report": Noul(instructions="Is this message a bug report?")}
    async with AsyncTypeSafeClient() as client:
        bug, sales = await asyncio.gather(
            client.system_one(
                "The API returns 500 whenever I pass an emoji in the name field.", question
            ),
            client.system_one("Can I get a discount if I pay annually?", question),
        )
    for label, response in (("bug ticket", bug), ("sales ticket", sales)):
        print(f"{label}: is_bug_report =", round(response.nouls["is_bug_report"].noul, 3))


if __name__ == "__main__":
    asyncio.run(main())

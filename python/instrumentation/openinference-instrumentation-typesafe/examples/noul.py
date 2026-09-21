"""
Traces a single ``TypeSafeClient.system_one`` call asking one Noul, a yes/no probability. Produces
one LLM span whose assistant message is the answers map with a single ``noul`` entry.

1. Run a local OTLP collector such as Phoenix: `uvx arize-phoenix serve`
2. Install dependencies: `pip install -r requirements.txt`
3. Set `TYPESAFE_API_KEY` and run this example: `python noul.py`
4. View the traces at http://localhost:6006 under the `typesafe-noul` project.
"""

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from typesafe_sdk import Noul, TypeSafeClient

from openinference.instrumentation.typesafe import TypeSafeAIInstrumentor
from openinference.semconv.resource import ResourceAttributes

endpoint = "http://localhost:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider(
    resource=Resource({ResourceAttributes.PROJECT_NAME: "typesafe-noul"})
)
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
TypeSafeAIInstrumentor().instrument(tracer_provider=tracer_provider)


def main() -> None:
    client = TypeSafeClient()
    state = "Hi, I was charged twice for order #4821. Can you refund the duplicate charge?"
    response = client.system_one(
        state,
        {
            "billing": Noul(
                instructions="Is this message about billing?",
                criteria={"true": "Payments, invoices, refunds", "false": "Anything else"},
            ),
        },
    )
    print("billing P(yes) =", round(response.nouls["billing"].noul, 3))
    print("usage          =", response.usage)


if __name__ == "__main__":
    main()

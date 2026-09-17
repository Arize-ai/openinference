"""
Traces a single ``TypeSafeClient.system_one`` call that asks all three TypeSafe primitives
(Noul, Choice, Score) over structured state. Produces one LLM span named ``TypeSafeClient``
whose ``llm.invocation_parameters`` carries the questions and whose output is the answers map.

1. Run a local OTLP collector such as Phoenix: `uvx arize-phoenix serve`
2. Install dependencies: `pip install -r requirements.txt`
3. Set `TYPESAFE_API_KEY` and run this example: `python system_one.py`
4. View the traces at http://localhost:6006 under the `typesafe-system-one` project.
"""

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

from openinference.instrumentation.typesafe import TypeSafeAIInstrumentor
from openinference.semconv.resource import ResourceAttributes

endpoint = "http://localhost:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider(
    resource=Resource({ResourceAttributes.PROJECT_NAME: "typesafe-system-one"})
)
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
TypeSafeAIInstrumentor().instrument(tracer_provider=tracer_provider)


def main() -> None:
    client = TypeSafeClient()
    state = {
        "ticket": {
            "subject": "Charged twice for one order",
            "messages": [
                {"from": "customer", "text": "I was charged twice. Please fix this ASAP."},
            ],
        },
        "account": {"plan": "pro", "tenure_months": 14},
    }
    response = client.system_one(
        state,
        {
            "billing": Noul(
                instructions="Is `ticket` about billing?",
                criteria={"true": "Payments, invoices, refunds", "false": "Anything else"},
            ),
            "department": Choice(
                instructions="Which team should handle `ticket`?",
                criteria={
                    "billing": "Payments, invoicing, refunds",
                    "technical": "Bugs, outages, integrations",
                    "sales": "Pricing, upgrades, new accounts",
                },
            ),
            "urgency": Score(
                instructions="How urgent is `ticket`?",
                criteria=["can wait", "this week", "today"],
            ),
        },
    )
    print("billing    P(yes) =", round(response.nouls["billing"].noul, 3))
    print("department        =", response.choices["department"].choice)
    print("urgency    score  =", round(response.scores["urgency"].score, 3))
    print("usage             =", response.usage)


if __name__ == "__main__":
    main()

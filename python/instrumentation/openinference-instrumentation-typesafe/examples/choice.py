"""
Traces a single ``TypeSafeClient.system_one`` call asking one Choice, a pick from labelled options.
Produces one LLM span whose assistant message holds the chosen label with its confidence and the
probability of every option.

1. Run a local OTLP collector such as Phoenix: `uvx arize-phoenix serve`
2. Install dependencies: `pip install -r requirements.txt`
3. Set `TYPESAFE_API_KEY` and run this example: `python choice.py`
4. View the traces at http://localhost:6006 under the `typesafe-choice` project.
"""

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from typesafe_sdk import Choice, TypeSafeClient

from openinference.instrumentation.typesafe import TypeSafeAIInstrumentor
from openinference.semconv.resource import ResourceAttributes

endpoint = "http://localhost:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider(
    resource=Resource({ResourceAttributes.PROJECT_NAME: "typesafe-choice"})
)
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
TypeSafeAIInstrumentor().instrument(tracer_provider=tracer_provider)


def main() -> None:
    client = TypeSafeClient()
    state = (
        "The webhook endpoint returns 500 since your last deploy and our orders are not syncing."
    )
    response = client.system_one(
        state,
        {
            "department": Choice(
                instructions="Which team should handle this message?",
                criteria={
                    "billing": "Payments, invoicing, refunds",
                    "technical": "Bugs, outages, integrations",
                    "sales": "Pricing, upgrades, new accounts",
                },
            ),
        },
    )
    answer = response.choices["department"]
    print("department    =", answer.choice)
    print("confidence    =", round(answer.confidence, 3))
    print("probabilities =", {k: round(v, 3) for k, v in answer.probabilities.items()})


if __name__ == "__main__":
    main()

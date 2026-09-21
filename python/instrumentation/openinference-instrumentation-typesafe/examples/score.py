"""
Traces a single ``TypeSafeClient.system_one`` call asking one Score, a position on an ordered
scale. Produces one LLM span whose assistant message holds the score, its confidence, the legend
mapping positions to labels, and the probability of every position.

1. Run a local OTLP collector such as Phoenix: `uvx arize-phoenix serve`
2. Install dependencies: `pip install -r requirements.txt`
3. Set `TYPESAFE_API_KEY` and run this example: `python score.py`
4. View the traces at http://localhost:6006 under the `typesafe-score` project.
"""

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from typesafe_sdk import Score, TypeSafeClient

from openinference.instrumentation.typesafe import TypeSafeAIInstrumentor
from openinference.semconv.resource import ResourceAttributes

endpoint = "http://localhost:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider(
    resource=Resource({ResourceAttributes.PROJECT_NAME: "typesafe-score"})
)
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
TypeSafeAIInstrumentor().instrument(tracer_provider=tracer_provider)


def main() -> None:
    client = TypeSafeClient()
    state = "Production is down for all customers and we are losing orders every minute."
    response = client.system_one(
        state,
        {
            "urgency": Score(
                instructions="How urgent is this message?",
                criteria=["can wait", "this week", "today", "right now"],
            ),
        },
    )
    answer = response.scores["urgency"]
    print("urgency score =", round(answer.score, 3))
    print("confidence    =", round(answer.confidence, 3))
    print("legend        =", answer.legend)


if __name__ == "__main__":
    main()

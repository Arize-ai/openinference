"""
Traces one ``system_one`` call inside ``using_attributes`` (session, user, metadata, tags),
then makes the same call inside ``suppress_tracing``. Exactly one LLM span is exported, and it
carries the context attributes.

1. Run a local OTLP collector such as Phoenix: `uvx arize-phoenix serve`
2. Install dependencies: `pip install -r requirements.txt`
3. Set `TYPESAFE_API_KEY` and run this example: `python context_attributes.py`
4. View the traces at http://localhost:6006 under the `typesafe-context-attributes` project.
"""

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from typesafe_sdk import Choice, Noul, TypeSafeClient

from openinference.instrumentation import suppress_tracing, using_attributes
from openinference.instrumentation.typesafe import TypeSafeAIInstrumentor
from openinference.semconv.resource import ResourceAttributes

endpoint = "http://localhost:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider(
    resource=Resource({ResourceAttributes.PROJECT_NAME: "typesafe-context-attributes"})
)
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
TypeSafeAIInstrumentor().instrument(tracer_provider=tracer_provider)


def main() -> None:
    client = TypeSafeClient()
    state = "I was charged twice. Please help ASAP."
    questions = {
        "billing": Noul(instructions="Is this about billing?"),
        "tone": Choice(instructions="What is the tone?", criteria={"calm": None, "angry": None}),
    }
    with using_attributes(
        session_id="session-42",
        user_id="user-1",
        metadata={"env": "demo"},
        tags=["example"],
    ):
        response = client.system_one(state, questions)
    print("tone =", response.choices["tone"].choice)

    with suppress_tracing():
        client.system_one(state, questions)  # not traced


if __name__ == "__main__":
    main()

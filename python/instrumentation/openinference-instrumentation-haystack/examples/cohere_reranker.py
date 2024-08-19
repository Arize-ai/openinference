"""
Requires a Cohere API request set with the `COHERE_API_KEY` environment variable.
"""

from haystack import Document, Pipeline
from haystack_integrations.components.rankers.cohere import CohereRanker
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.haystack import HaystackInstrumentor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))


HaystackInstrumentor().instrument(tracer_provider=tracer_provider)

ranker = CohereRanker()
pipe = Pipeline()
pipe.add_component("ranker", ranker)
response = pipe.run(
    {
        "ranker": {
            "query": "Who won the World Cup in 2022?",
            "documents": [
                Document(
                    content="Paul Graham is the founder of Y Combinator.",
                ),
                Document(
                    content=(
                        "Lionel Messi, captain of the Argentinian national team, "
                        " won his first World Cup in 2022."
                    ),
                ),
                Document(
                    content="France lost the 2022 World Cup.",
                ),  # Cohere consistently ranks this document last
            ],
            "top_k": 2,
        }
    }
)
print(f"{response=}")

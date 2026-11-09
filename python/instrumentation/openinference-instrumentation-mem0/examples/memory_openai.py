"""Traces mem0's default OpenAI-backed memory, including the LLM and embedding calls it makes.

Requires OPENAI_API_KEY. Spans are exported to Phoenix at http://localhost:6006 under the
project "mem0-openai"; the OpenAI spans nest under the mem0 operation that issued them.
"""

import os

from mem0 import Memory
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.mem0 import Mem0Instrumentor
from openinference.instrumentation.openai import OpenAIInstrumentor
from openinference.semconv.resource import ResourceAttributes

endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://127.0.0.1:6006") + "/v1/traces"
tracer_provider = TracerProvider(
    resource=Resource({ResourceAttributes.PROJECT_NAME: "mem0-openai"})
)
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
Mem0Instrumentor().instrument(tracer_provider=tracer_provider)
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

if __name__ == "__main__":
    memory = Memory()
    memory.add(
        [
            {"role": "user", "content": "I'm planning a trip to Japan and I'm vegetarian."},
            {"role": "assistant", "content": "Great, I'll keep vegetarian options in mind."},
        ],
        user_id="alice",
    )
    results = memory.search("What should I eat in Tokyo?", filters={"user_id": "alice"})
    for result in results["results"]:
        print(f"{result['score']:.2f}  {result['memory']}")

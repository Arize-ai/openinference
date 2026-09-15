from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from together import Together

from openinference.instrumentation import using_attributes
from openinference.instrumentation.together import TogetherInstrumentor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

TogetherInstrumentor().instrument(tracer_provider=tracer_provider)


if __name__ == "__main__":
    client = Together()
    # Session, user, metadata, and tags are propagated onto every span
    # created inside the context manager.
    with using_attributes(
        session_id="session-42",
        user_id="user-1",
        metadata={"env": "demo"},
        tags=["example"],
    ):
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=[{"role": "user", "content": "In one sentence, what is a trace?"}],
        )
    print(response.choices[0].message.content)

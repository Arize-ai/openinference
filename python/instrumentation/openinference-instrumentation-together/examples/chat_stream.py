from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from together import Together

from openinference.instrumentation.together import TogetherInstrumentor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

TogetherInstrumentor().instrument(tracer_provider=tracer_provider)


if __name__ == "__main__":
    client = Together()
    stream = client.chat.completions.create(
        model="MiniMaxAI/MiniMax-M3",
        messages=[{"role": "user", "content": "What are the top 3 things to do in New York?"}],
        stream=True,
    )

    printed_answer_header = False
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta

        # Reasoning models return their thinking in a separate `reasoning` field.
        if getattr(delta, "reasoning", None):
            print(delta.reasoning, end="", flush=True)

        # The final answer arrives in `content`.
        if getattr(delta, "content", None):
            if not printed_answer_header:
                print("\n\n--- Answer ---\n", flush=True)
                printed_answer_header = True
            print(delta.content, end="", flush=True)
    print()

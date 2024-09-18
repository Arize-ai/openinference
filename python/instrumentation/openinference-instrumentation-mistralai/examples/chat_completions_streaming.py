from mistralai import Mistral
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from openinference.instrumentation.mistralai import MistralAIInstrumentor

tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

MistralAIInstrumentor().instrument(tracer_provider=tracer_provider)


if __name__ == "__main__":
    from mistralai import Mistral

    client = Mistral(api_key="redacted")

    response_stream = client.chat.stream(
        model="mistral-small-latest",
        messages=[
            {
                "content": "Who is the best French painter? Answer in one short sentence.",
                "role": "user",
            },
        ],
    )

    for chunk in response_stream:
        print(chunk)

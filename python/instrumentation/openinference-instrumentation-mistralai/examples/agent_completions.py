from mistralai import Mistral
from opentelemetry import trace as trace_api
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from openinference.instrumentation.mistralai import MistralAIInstrumentor

tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
trace_api.set_tracer_provider(tracer_provider)

MistralAIInstrumentor().instrument()

if __name__ == "__main__":
    client = Mistral(api_key="redacted")
    response = client.chat.complete(
        agent_id="ag:ad73bfd7:20240912:python-codegen-agent:0375a7cf",
        messages=[
            {
                "role": "user",
                "content":  "is it too much"
            },
        ],
    )
    print(response)

import asyncio

import openai
from openinference.instrumentation import using_attributes
from openinference.instrumentation.openai import OpenAIInstrumentor
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
trace_api.set_tracer_provider(tracer_provider)

OpenAIInstrumentor().instrument()


async def chat_completions(**kwargs):
    client = openai.AsyncOpenAI()
    with using_attributes(
        session_id="my-test-session",
        user_id="my-test-user",
        metadata={
            "test-int": 1,
            "test-str": "string",
            "test-list": [1, 2, 3],
            "test-dict": {
                "key-1": "val-1",
                "key-2": "val-2",
            },
        },
        tags=["tag-1", "tag-2"],
        prompt_template="Who won the soccer match in {city} on {date}",
        prompt_template_version="v1.0",
        prompt_template_variables={
            "city": "Johannesburg",
            "date": "July 11th",
        },
    ):
        response = await client.chat.completions.create(**kwargs)
        async for chunk in response:
            if content := chunk.choices[0].delta.content:
                print(content, end="")


if __name__ == "__main__":
    asyncio.run(
        chat_completions(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Write a haiku."}],
            max_tokens=20,
            stream=True,
        ),
    )

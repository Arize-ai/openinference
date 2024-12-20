import asyncio
import os

from groq import AsyncGroq, Groq
from groq.types.chat import ChatCompletionToolMessageParam
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.groq import GroqInstrumentor


def test():
    client = Groq(
        api_key=os.environ.get("GROQ_API_KEY"),
    )

    weather_function = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "finds the weather for a given city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city to find the weather for, e.g. 'London'",
                    }
                },
                "required": ["city"],
            },
        },
    }

    sys_prompt = "Respond to the user's query using the correct tool."
    user_msg = "What's the weather like in San Francisco?"

    messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_msg}]
    response = client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=messages,
        temperature=0.0,
        tools=[weather_function],
        tool_choice="required",
    )

    message = response.choices[0].message
    assert (tool_calls := message.tool_calls)
    tool_call_id = tool_calls[0].id
    messages.append(message)
    messages.append(
        ChatCompletionToolMessageParam(content="sunny", role="tool", tool_call_id=tool_call_id),
    )
    final_response = client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=messages,
    )
    return final_response


async def async_test():
    client = AsyncGroq(
        api_key=os.environ.get("GROQ_API_KEY"),
    )

    weather_function = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "finds the weather for a given city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city to find the weather for, e.g. 'London'",
                    }
                },
                "required": ["city"],
            },
        },
    }

    sys_prompt = "Respond to the user's query using the correct tool."
    user_msg = "What's the weather like in San Francisco?"

    messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_msg}]
    response = await client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=messages,
        temperature=0.0,
        tools=[weather_function],
        tool_choice="required",
    )

    message = response.choices[0].message
    assert (tool_calls := message.tool_calls)
    tool_call_id = tool_calls[0].id
    messages.append(message)
    messages.append(
        ChatCompletionToolMessageParam(content="sunny", role="tool", tool_call_id=tool_call_id),
    )
    final_response = await client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=messages,
    )
    return final_response


if __name__ == "__main__":
    endpoint = "http://0.0.0.0:6006/v1/traces"
    tracer_provider = trace_sdk.TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

    GroqInstrumentor().instrument(tracer_provider=tracer_provider)

    response = test()
    print("Response\n--------")
    print(response)

    async_response = asyncio.run(async_test())
    print("\nAsync Response\n--------")
    print(async_response)

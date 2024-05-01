import openai
import asyncio

from openinference.instrumentation.openai import OpenAIInstrumentor
from openinference.instrumentation import (
    using_attributes,
    using_session,
    using_user,
    using_metadata,
    using_tags,
)

from opentelemetry import trace as trace_api

# from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor


from opentelemetry.context import (
    get_current,
)

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
# tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
trace_api.set_tracer_provider(tracer_provider)

OpenAIInstrumentor().instrument()


def sync_test():
    with (
        using_attributes(
            session_id="test-session",
            metadata={
                "show": "blacklist",
                "hour": "dusk",
            },
        ),
        using_attributes(
            user_id="kiko",
            tags=["x", "y", "z"],
        ),
    ):
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Write a haiku."}],
            max_tokens=20,
        )
        with open("sync.json", "w") as f:
            print(response.choices[0].message.content, file=f)
            ctx = get_current()
            print(f"{ctx=}")


async def async_test():
    async with (
        using_attributes(
            session_id="test-session",
            metadata={
                "show": "blacklist",
                "hour": "dusk",
            },
        ),
        using_attributes(
            user_id="kiko",
            tags=["x", "y", "z"],
        ),
    ):
        client = openai.AsyncOpenAI()
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Write a haiku."}],
            max_tokens=20,
        )
        with open("async.json", "w") as f:
            print(response.choices[0].message.content, file=f)
        ctx = get_current()


def sync_test_mini():
    with (
        using_session("mini-session"),
        using_user(user_id="mini-user"),
        using_metadata(
            {
                "mini-show": "mini-blacklist",
                "mini-hour": "mini-dusk",
            },
        ),
        using_tags(["mini-x", "mini-y", "mini-z"]),
    ):
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Write a haiku."}],
            max_tokens=20,
        )
        with open("sync.json", "w") as f:
            print(response.choices[0].message.content, file=f)
        ctx = get_current()
        print(f"{ctx=}")


async def async_test_help():
    yield await async_test()


if __name__ == "__main__":
    print(" ")
    print("SYNC")
    print(" ")
    sync_test()
    print(" ")
    print("ASYNC")
    print(" ")
    loop = asyncio.get_event_loop()
    loop.run_until_complete(async_test())
    print(" ")
    print("MINI USERS SYNC")
    print(" ")
    sync_test_mini()
    print(" ")
    print("DONE")
    print(" ")

    ctx = get_current()
    print(f"{ctx=}")

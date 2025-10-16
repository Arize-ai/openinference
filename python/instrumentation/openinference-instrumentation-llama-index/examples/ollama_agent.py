import asyncio

from llama_index.core.agent.workflow import (
    AgentStream,
    FunctionAgent,
    ToolCall,
    ToolCallResult,
)
from llama_index.llms.ollama import Ollama
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

endpoint = "http://localhost:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)


def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b


async def run_async():
    print("Started RUN Method....")
    llm = Ollama(
        model="gpt-oss:20b",
        request_timeout=360,
        thinking=True,
        temperature=1.0,
        context_window=8000,
    )

    agent = FunctionAgent(
        tools=[multiply],
        llm=llm,
        system_prompt="You are a helpful assistant that can multiply and add numbers. "
        "Always rely on tools for math operations.",
    )
    print("Agent Created....")
    handler = agent.run("What is 1234 * 5678?")
    async for ev in handler.stream_events():
        if isinstance(ev, ToolCall):
            print(f"\nTool call: {ev.tool_name}({ev.tool_kwargs}")
        elif isinstance(ev, ToolCallResult):
            print(f"\nTool call: {ev.tool_name}({ev.tool_kwargs}) -> {ev.tool_output}")
        elif isinstance(ev, AgentStream):
            print(ev.delta, end="", flush=True)
    print("Functional Call executed....")
    resp = await handler
    print(resp)


if __name__ == "__main__":
    asyncio.run(run_async())

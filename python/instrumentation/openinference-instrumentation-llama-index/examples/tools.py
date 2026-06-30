from llama_index.core.tools import FunctionTool
from llama_index.llms.anthropic import Anthropic
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)


def get_weather(location: str) -> str:
    """Useful for getting the weather for a given location."""
    raise NotImplementedError


TOOL = FunctionTool.from_defaults(get_weather, name="get_weather")

llm = Anthropic(model="claude-3-5-haiku-20241022")

if __name__ == "__main__":
    response = llm.chat(
        **llm._prepare_chat_with_tools([TOOL], "what's the weather in San Francisco?"),
    )
    print(response)

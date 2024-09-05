from llama_index.core.base.llms.types import ChatMessage
from llama_index.llms.openai import OpenAI
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)

llm = OpenAI(model="gpt-3.5-turbo")

if __name__ == "__main__":
    response_gen = llm.stream_chat(
        [ChatMessage(content="hello")],
        stream_options={"include_usage": True},
    )
    for response in response_gen:
        print(response.delta, end="")

import tempfile
from urllib.request import urlretrieve

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.openai import OpenAI
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)

with tempfile.NamedTemporaryFile() as tf:
    urlretrieve(
        "https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt",
        tf.name,
    )
    documents = SimpleDirectoryReader(input_files=[tf.name]).load_data()

index = VectorStoreIndex.from_documents(documents)
chat_engine = index.as_chat_engine()
Settings.llm = OpenAI(model="gpt-3.5-turbo")

if __name__ == "__main__":
    response = chat_engine.stream_chat("What did the author do growing up?")
    response.print_response_stream()

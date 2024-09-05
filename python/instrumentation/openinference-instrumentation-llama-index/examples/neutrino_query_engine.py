"""
This script requires that your Neutrino API key is set in the NEUTRINO_API_KEY
environment variable and that some text files are placed in the `files`
directory.

https://www.neutrinoapp.com/
"""

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.neutrino import Neutrino
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)

Settings.llm = Neutrino(
    model="chat-preview",
    api_base="https://router.neutrinoapp.com/api/engines",
)
documents = SimpleDirectoryReader("files").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("summarize each document in a few sentences")

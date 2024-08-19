import tempfile
from urllib.request import urlretrieve

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.response.pprint_utils import pprint_response
from llama_index.llms.openai import OpenAI
from llama_index.postprocessor.cohere_rerank import CohereRerank
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)

Settings.llm = OpenAI(model="gpt-3.5-turbo")

essay = "https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt"
with tempfile.NamedTemporaryFile() as tf:
    urlretrieve(essay, tf.name)
    documents = SimpleDirectoryReader(input_files=[tf.name]).load_data()
index = VectorStoreIndex.from_documents(documents)

rerank = CohereRerank(top_n=2)
query_engine = index.as_query_engine(similarity_top_k=10, node_postprocessors=[rerank])

if __name__ == "__main__":
    response = query_engine.query("What did the author do growing up?")
    pprint_response(response, show_source=True)

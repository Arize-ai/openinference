import tempfile
from urllib.request import urlretrieve

from llama_index.core import SimpleDirectoryReader
from llama_index.core.extractors import SummaryExtractor, TitleExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import MetadataMode
from llama_index.embeddings.openai import OpenAIEmbedding
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

llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
pipline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=1024, chunk_overlap=20),
        TitleExtractor(llm=llm, metadata_mode=MetadataMode.EMBED, num_workers=8),
        SummaryExtractor(llm=llm, metadata_mode=MetadataMode.EMBED, num_workers=8),
        OpenAIEmbedding(),
    ]
)

if __name__ == "__main__":
    nodes = pipline.run(documents=documents)

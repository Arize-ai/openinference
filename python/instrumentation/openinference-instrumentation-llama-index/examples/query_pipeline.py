import tempfile
from typing import List
from urllib.request import urlretrieve

from llama_index.core import PromptTemplate, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.query_pipeline import QueryPipeline
from llama_index.llms.openai import OpenAI
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from openinference.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from pydantic import BaseModel, Field

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
LlamaIndexInstrumentor().instrument(
    tracer_provider=tracer_provider,
    # use_legacy_callback_handler=True,
)

tracer = tracer_provider.get_tracer(__name__)

essay = "https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt"
with tempfile.NamedTemporaryFile() as tf:
    urlretrieve(essay, tf.name)
    documents = SimpleDirectoryReader(input_files=[tf.name]).load_data()
index = VectorStoreIndex.from_documents(documents)


class Movie(BaseModel):
    """Object representing a single movie."""

    name: str = Field(..., description="Name of the movie.")
    year: int = Field(..., description="Year of the movie.")


class Movies(BaseModel):
    """Object representing a list of movies."""

    movies: List[Movie] = Field(..., description="List of movies.")


llm = OpenAI(model="gpt-4o-mini")
output_parser = PydanticOutputParser(Movies)
json_prompt_str = """\
Please generate related movies to {movie_name}. Output with the following JSON format: 
"""
json_prompt_str = output_parser.format(json_prompt_str)

# add JSON spec to prompt template
json_prompt_tmpl = PromptTemplate(json_prompt_str)

if __name__ == "__main__":
    p = QueryPipeline(chain=[json_prompt_tmpl, llm, output_parser], verbose=True)
    with tracer.start_as_current_span("test"):
        output = p.run(movie_name="Toy Story")
    print(output)

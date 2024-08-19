import pandas as pd
import wikipedia
from llama_index.core import Document, Settings
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.query_engine import NLSQLTableQueryEngine, RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool
from llama_index.core.utilities.sql_wrapper import SQLDatabase
from llama_index.llms.openai import OpenAI
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from sqlalchemy import create_engine

from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)

engine = create_engine("sqlite:///:memory:")
pd.read_parquet(
    "https://storage.googleapis.com/arize-phoenix-assets/datasets/structured/camera-info/cameras.parquet"
).to_sql("cameras", engine, index=False)
sql_tool = QueryEngineTool.from_defaults(
    query_engine=NLSQLTableQueryEngine(
        sql_database=SQLDatabase(engine, include_tables=["cameras"]),
        tables=["cameras"],
    ),
    description=(
        "Useful for translating a natural language query into a SQL query over"
        " a table containing technical details about specific digital camera models: Model,"
        " Release date, Max resolution, Low resolution, Effective pixels, Zoom wide (W),"
        " Zoom tele (T), Normal focus range, Macro focus range, Storage included,"
        " Weight (inc. batteries), Dimensions, Price"
    ),
)

page = wikipedia.page(pageid=52797)
vector_tool = QueryEngineTool.from_defaults(
    query_engine=VectorStoreIndex.from_documents(
        [Document(id_=page.pageid, text=page.content)]
    ).as_query_engine(),
    description="Useful for answering generic questions about digital cameras.",
)
query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[sql_tool, vector_tool],
)
Settings.llm = OpenAI(model="gpt-3.5-turbo")

if __name__ == "__main__":
    response = query_engine.query("What is the most expensive digital camera?")
    print(str(response))
    response = query_engine.query("Tell me about the history of digital camera sensors.")
    print(str(response))

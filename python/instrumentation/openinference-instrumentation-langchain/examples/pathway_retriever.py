from typing import List

from langchain_community.vectorstores import PathwayVectorClient
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever, Document
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
trace_api.set_tracer_provider(tracer_provider)
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

LangChainInstrumentor().instrument()


if __name__ == "__main__":
    client = PathwayVectorClient(url="https://demo-document-indexing.pathway.stream")
    retriever = client.as_retriever()

    retriever.invoke("what is pathway?")

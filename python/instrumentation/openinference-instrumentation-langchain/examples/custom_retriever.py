from typing import List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever, Document
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

LangChainInstrumentor().instrument(tracer_provider=tracer_provider)


class CustomRetriever(BaseRetriever):
    """
    This example is taken from langchain docs.

    https://python.langchain.com/v0.1/docs/modules/data_connection/retrievers/custom_retriever/

    A custom retriever that contains the top k documents that contain the user query.

    This retriever only implements the sync method _get_relevant_documents.

    If the retriever were to involve file access or network access, it could benefit
    from a native async implementation of `_aget_relevant_documents`.

    As usual, with Runnables, there's a default async implementation that's provided
    that delegates to the sync implementation running on another thread.
    """

    k: int
    """Number of top results to return"""

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Sync implementations for retriever."""
        matching_documents: List[Document] = []

        # Custom logic to find the top k documents that contain the query

        for index in range(self.k):
            matching_documents.append(Document(page_content=f"dummy content at {index}", score=1.0))
        return matching_documents


retriever = CustomRetriever(k=3)


if __name__ == "__main__":
    documents = retriever.invoke("what is the meaning of life?")

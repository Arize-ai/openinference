import cohere as cohere_v2
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation import TraceConfig
from openinference.instrumentation.cohere import CohereInstrumentor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

docs = [
    "Carson City is the capital city of the American state of Nevada.",
    "The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean. "
    "Its capital is Saipan.",
    "Capitalization or capitalisation in English grammar is the use of a capital letter at the "
    "start of a word. English usage varies from capitalization in other languages.",
    "Washington, D.C. (also known as simply Washington or D.C., and officially as the District of "
    "Columbia) is the capital of the United States. It is a federal district.",
    "Capital punishment has existed in the United States since before the United States was a "
    "country. As of 2017, capital punishment is legal in 30 of the 50 states.",
]
query = "What is the capital of the United States?"


def rerank_docs():
    CohereInstrumentor().instrument(tracer_provider=tracer_provider)
    co = cohere_v2.ClientV2()
    response = co.rerank(
        model="rerank-v4.0-pro",
        query=query,
        documents=docs,
        top_n=3,
    )
    print(response)
    CohereInstrumentor().uninstrument()


def rerank_docs_hide_input_output():
    CohereInstrumentor().instrument(
        tracer_provider=tracer_provider,
        config=TraceConfig(hide_inputs=True, hide_outputs=True),
    )
    co = cohere_v2.ClientV2()
    response = co.rerank(
        model="rerank-v4.0-pro",
        query=query,
        documents=docs,
        top_n=1,
    )
    print(response)
    CohereInstrumentor().uninstrument()


if __name__ == "__main__":
    rerank_docs()
    rerank_docs_hide_input_output()

from io import BytesIO

import openai
import requests
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from openinference.instrumentation.openai import OpenAIInstrumentor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)


def create_file(client, file_path):
    if file_path.startswith("http://") or file_path.startswith("https://"):
        response = requests.get(file_path)
        file_content = BytesIO(response.content)
        file_name = file_path.split("/")[-1]
        file_tuple = (file_name, file_content)
        result = client.files.create(file=file_tuple, purpose="assistants")
    else:
        with open(file_path, "rb") as file_content:
            result = client.files.create(file=file_content, purpose="assistants")
    print(result.id)
    return result.id


def create_vector_store(client, file_id):
    vector_store = client.vector_stores.create(name="knowledge_base")
    print(vector_store.id)
    result = client.vector_stores.files.create(vector_store_id=vector_store.id, file_id=file_id)
    print(result)
    return vector_store.id


client = openai.OpenAI()
file_id = create_file(client, "https://cdn.openai.com/API/docs/deep_research_blog.pdf")
vector_store_id = create_vector_store(client, file_id)
response = client.responses.create(
    model="gpt-4o-mini",
    tools=[{"type": "file_search", "vector_store_ids": [vector_store_id], "max_num_results": 20}],
    input=[{"role": "user", "content": "What are limitations of deep research?"}],
)
print(response)

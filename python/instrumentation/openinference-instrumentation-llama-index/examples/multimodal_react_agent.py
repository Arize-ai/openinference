import tempfile

import requests
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.agent import AgentRunner
from llama_index.core.agent.react_multimodal.step import MultimodalReActAgentWorker
from llama_index.core.base.agent.types import Task
from llama_index.core.schema import ImageDocument
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.openai import OpenAI
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.readers.web import SimpleWebPageReader
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)


def execute_step(agent: AgentRunner, task: Task):
    step_output = agent.run_step(task.task_id)
    if step_output.is_last:
        response = agent.finalize_response(task.task_id)
        print(f"> Agent finished: {str(response)}")
        return response
    else:
        return None


def execute_steps(agent: AgentRunner, task: Task):
    response = execute_step(agent, task)
    while response is None:
        response = execute_step(agent, task)
    return response


url = "https://openai.com/blog/new-models-and-developer-products-announced-at-devday"
reader = SimpleWebPageReader(html_to_text=True)
documents = reader.load_data(urls=[url])
vector_index = VectorStoreIndex.from_documents(documents)
Settings.llm = OpenAI(temperature=0, model="gpt-3.5-turbo")
query_tool = QueryEngineTool(
    query_engine=vector_index.as_query_engine(),
    metadata=ToolMetadata(
        name="vector_tool",
        description="Useful to lookup new features announced by OpenAI",
    ),
)

mm_llm = OpenAIMultiModal(model="gpt-4o", max_new_tokens=1000)
react_step_engine = MultimodalReActAgentWorker.from_tools(
    [query_tool],
    multi_modal_llm=mm_llm,
    verbose=True,
)
agent = react_step_engine.as_agent()
query_str = (
    "The photo shows some new features released by OpenAI. "
    "Can you pinpoint the features in the photo and give more details using relevant tools?"
)
jpg_url = "https://images.openai.com/blob/a2e49de2-ba5b-4869-9c2d-db3b4b5dcc19/new-models-and-developer-products-announced-at-devday.jpg"


if __name__ == "__main__":
    with tempfile.NamedTemporaryFile(suffix=".jpg") as tf:
        with open(tf.name, "wb") as f:
            f.write(requests.get(jpg_url).content)
        image_document = ImageDocument(image_path=tf.name)
        task = agent.create_task(query_str, extra_state={"image_docs": [image_document]})
        response = execute_steps(agent, task)
        print(str(response))

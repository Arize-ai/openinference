import tempfile

import requests
from llama_index.core import SimpleDirectoryReader
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from openinference.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

url = "https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/gpt4_experiments/llama2_mistral.png"

query = """Based on the image provided. Follow the steps and answer the query - Assuming mistral is available in 7B series. How well does mistral model compared to llama2 model?
Examine the Image: Look at the mentioned category in the query in the Image.
Identify Relevant Data: Note the respective percentages.
Evaluate: Compare if there is any comparison required as per the query.
Draw a Conclusion: Now draw the conclusion based on the whole data.
"""  # noqa: E501

llm = OpenAIMultiModal(model="gpt-4o", max_new_tokens=500)

if __name__ == "__main__":
    with tempfile.NamedTemporaryFile(suffix=".png") as tf:
        with open(tf.name, "wb") as f:
            f.write(requests.get(url).content)
        image_documents = SimpleDirectoryReader(input_files=[tf.name]).load_data()
        response_gen = llm.stream_complete(
            prompt=query,
            image_documents=image_documents,
            stream_options={"include_usage": True},
        )
        for response in response_gen:
            print(response.delta, end="")

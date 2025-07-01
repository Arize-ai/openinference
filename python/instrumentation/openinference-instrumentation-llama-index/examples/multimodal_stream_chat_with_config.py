import os

from llama_index.core.multi_modal_llms.generic_utils import load_image_urls
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.multi_modal_llms.openai.utils import (
    generate_openai_multi_modal_chat_message,
)
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from openinference.instrumentation import TraceConfig
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer_provider.add_span_processor(
    span_processor=SimpleSpanProcessor(span_exporter=ConsoleSpanExporter())
)


IMAGE_URLS = [
    # "https://www.visualcapitalist.com/wp-content/uploads/2023/10/US_Mortgage_Rate_Surge-Sept-11-1.jpg",
    # "https://www.sportsnet.ca/wp-content/uploads/2023/11/CP1688996471-1040x572.jpg",
    "https://res.cloudinary.com/hello-tickets/image/upload/c_limit,f_auto,q_auto,w_1920/v1640835927/o3pfl41q7m5bj8jardk0.jpg",  # noqa: E501
    # "https://www.cleverfiles.com/howto/wp-content/uploads/2018/03/minion.jpg",
]


def get_chat_messages():
    return [
        generate_openai_multi_modal_chat_message(
            prompt="Describe the images as an alternative text",
            role="user",
            image_documents=image_documents,
        ),
    ]


if __name__ == "__main__":
    image_documents = load_image_urls(IMAGE_URLS)

    openai_mm_llm = OpenAIMultiModal(
        model="gpt-4o",
    )
    os.environ["OPENINFERENCE_HIDE_INPUT_IMAGES"] = str(True)  # Will hide input images
    os.environ["OPENINFERENCE_HIDE_INPUTS"] = str(True)  # Will hide all inputs
    config = TraceConfig(
        hide_inputs=False,  # Overwrites the environment value setting
        hide_output_text=True,  # The text in output messages will appear as __REDACTED__
    )
    LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider, config=config)

    response_gen = openai_mm_llm.stream_chat(
        messages=get_chat_messages(),
        stream_options={
            "include_usage": True,
        },
    )
    for response in response_gen:
        print(response.delta, end="")

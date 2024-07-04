from llama_index.core.multi_modal_llms.generic_utils import load_image_urls
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.multi_modal_llms.openai.utils import (
    generate_openai_multi_modal_chat_message,
)
from openinference.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer_provider.add_span_processor(
    span_processor=SimpleSpanProcessor(span_exporter=ConsoleSpanExporter())
)

LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)


IMAGE_URLS = [
    # "https://www.visualcapitalist.com/wp-content/uploads/2023/10/US_Mortgage_Rate_Surge-Sept-11-1.jpg",
    # "https://www.sportsnet.ca/wp-content/uploads/2023/11/CP1688996471-1040x572.jpg",
    "https://res.cloudinary.com/hello-tickets/image/upload/c_limit,f_auto,q_auto,w_1920/v1640835927/o3pfl41q7m5bj8jardk0.jpg",
    # "https://www.cleverfiles.com/howto/wp-content/uploads/2018/03/minion.jpg",
]

if __name__ == "__main__":

    image_documents = load_image_urls(IMAGE_URLS)

    openai_mm_llm = OpenAIMultiModal(
        model="gpt-4o",
    )

    chat_msg_1 = generate_openai_multi_modal_chat_message(
        prompt="Describe the images as an alternative text",
        role="user",
        image_documents=image_documents,
    )

    chat_msg_2 = generate_openai_multi_modal_chat_message(
        prompt="The image is a graph showing the surge in US mortgage rates. It is a visual representation of data, with a title at the top and labels for the x and y-axes. Unfortunately, without seeing the image, I cannot provide specific details about the data or the exact design of the graph.",
        role="assistant",
    )

    chat_msg_3 = generate_openai_multi_modal_chat_message(
        prompt="can I know more?",
        role="user",
    )
    response_gen = openai_mm_llm.stream_chat(
        # prompt="Describe the images as an alternative text",
        messages=[
            chat_msg_1,
            chat_msg_2,
            chat_msg_3,
        ],
        stream_options={
            "include_usage": True,
        },
    )
    for response in response_gen:
        print(response.delta, end="")

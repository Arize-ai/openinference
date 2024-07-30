import itertools

from llama_index.core.multi_modal_llms.generic_utils import load_image_urls
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.multi_modal_llms.openai.utils import (
    generate_openai_multi_modal_chat_message,
)
from openinference.instrumentation import TraceConfig, using_attributes
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from phoenix.trace import using_project

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


def get_chat_messaged():
    return [
        generate_openai_multi_modal_chat_message(
            prompt="Describe the images as an alternative text",
            role="user",
            image_documents=image_documents,
        ),
        # generate_openai_multi_modal_chat_message(
        #     prompt="The image is a graph showing the surge in US mortgage rates. It is a visual representation of data, with a title at the top and labels for the x and y-axes. Unfortunately, without seeing the image, I cannot provide specific details about the data or the exact design of the graph.",  # noqa: E501
        #     role="assistant",
        # ),
        # generate_openai_multi_modal_chat_message(
        #     prompt="can I know more?",
        #     role="user",
        # ),
    ]


if __name__ == "__main__":
    image_documents = load_image_urls(IMAGE_URLS)

    openai_mm_llm = OpenAIMultiModal(
        model="gpt-4o",
    )
    combinations = itertools.product([False, True], repeat=3)
    # print(f"{len(combinations)=}")
    for i, combination in enumerate(combinations):
        print(f"{i=}")
        (
            # hide_inputs,
            hide_outputs,
            # hide_input_messages,
            hide_output_messages,
            # hide_input_images,
            # hide_input_text,
            hide_output_text,
            # hide_embedding_vectors,
            # base64_image_max_length,
        ) = combination
        # os.environ["OPENINFERENCE_HIDE_INPUT_IMAGES"] = str(True)  # Will hide input images
        # os.environ["OPENINFERENCE_HIDE_INPUTS"] = str(True)  # Will hide all inputs
        # config = TraceConfig(
        #     hide_inputs=False,  # Overwrites the environment value setting
        #     hide_output_text=True,  # The text in output messages will appear as __REDACTED__
        # )
        hide_inputs = True
        config = TraceConfig(
            hide_inputs=hide_inputs,
            hide_outputs=hide_outputs,
            # hide_input_messages=hide_input_messages,
            hide_output_messages=hide_output_messages,
            # hide_input_images=hide_input_images,
            # hide_input_text=hide_input_text,
            hide_output_text=hide_output_text,
            # hide_embedding_vectors=  hide_embedding_vectors,
            # base64_image_max_length, base64_image_max_length,
        )
        # config = TraceConfig()
        print(f"{config=}")

        LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider, config=config)

        with using_project("test-outputs"):
            with using_attributes(
                metadata={
                    # "hide_inputs": hide_inputs,
                    "hide_outputs": hide_outputs,
                    # "hide_input_messages": hide_input_messages,
                    "hide_output_messages": hide_output_messages,
                    # "hide_input_images": hide_input_images,
                    # "hide_input_text": hide_input_text,
                    "hide_output_text": hide_output_text,
                    # "hide_embedding_vectors": hide_embedding_vectors,
                    # "base64_image_max_length": base64_image_max_length,
                },
            ):
                response_gen = openai_mm_llm.stream_chat(
                    messages=get_chat_messaged(),
                    stream_options={
                        "include_usage": True,
                    },
                )
                for response in response_gen:
                    print(response.delta, end="")

        LlamaIndexInstrumentor().uninstrument()

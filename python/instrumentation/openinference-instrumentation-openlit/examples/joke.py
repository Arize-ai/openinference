import asyncio
import os

import grpc
import openlit
from _span_processor import OpenInferenceSpanProcessor
from arize.otel import register
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from semantic_kernel import Kernel, __version__
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.functions import KernelArguments

os.environ["GLOBAL_LLM_SERVICE"] = "OpenAI"
os.environ["OPENAI_API_KEY"] = "INSERT OPENAI API KEY HERE"
os.environ["OPENAI_CHAT_MODEL_ID"] = "gpt-4o-mini"
os.environ["OPENAI_TEXT_MODEL_ID"] = "gpt-4o-mini"

print(__version__)

SPACE_ID = "INSERT ARIZE SPACE ID HERE"
API_KEY = "INSERT ARIZE API KEY HERE"

if __name__ == "__main__":
    provider = register(
        space_id=SPACE_ID,
        api_key=API_KEY,
        project_name="semantic-kernel-example",
        set_global_tracer_provider=True,
    )

    provider.add_span_processor(OpenInferenceSpanProcessor())

    provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(
                endpoint="otlp.arize.com:443",
                headers={
                    "authorization": f"Bearer {API_KEY}",
                    "api_key": API_KEY,
                    "arize-space-id": SPACE_ID,
                    "arize-interface": "python",
                    "user-agent": "arize-python",
                },
                compression=grpc.Compression.Gzip,
            )
        )
    )

    tracer = provider.get_tracer(__name__)

    openlit.init(otel_tracer=tracer)

kernel = Kernel()

kernel.remove_all_services()

service_id = "default"
kernel.add_service(
    OpenAIChatCompletion(
        service_id=service_id,
    ),
)

plugin_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../prompt_template_samples/")
)
plugin = kernel.add_plugin(parent_directory=plugin_dir, plugin_name="FunPlugin")

joke_function = plugin["Joke"]


async def run_joke():
    joke = await kernel.invoke(
        joke_function,
        KernelArguments(input="time travel to dinosaur age", style="super silly"),
    )
    print(joke)


if __name__ == "__main__":
    asyncio.run(run_joke())

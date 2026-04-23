import io

import openai
import requests
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from PIL import Image
from smolagents import OpenAIServerModel, tool
from smolagents.agents import ToolCallingAgent
from smolagents.models import ChatMessage, MessageRole

from openinference.instrumentation.smolagents import SmolagentsInstrumentor

SAMPLE_IMAGE_URL = "https://fastly.picsum.photos/id/237/200/300.jpg?hmac=TmmQSbShHz9CdQm0NkEjx1Dyh_Y984R9LpNrpvH2D_U"
ICON_IMAGE_URL = "https://cdn-icons-png.flaticon.com/128/149/149071.png"


def setup_tracing() -> None:
    endpoint = "http://127.0.0.1:6006/v1/traces"
    tracer_provider = trace_sdk.TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
    tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    SmolagentsInstrumentor().instrument(tracer_provider=tracer_provider)


def image_url_input() -> None:
    """Pass an image URL alongside text as model input."""
    model = OpenAIServerModel(model_id="gpt-4o-mini")
    response = model(
        messages=[
            ChatMessage(
                role=MessageRole.USER,
                content=[
                    {"type": "text", "text": "What do you see in this image?"},
                    {"type": "image_url", "image_url": {"url": SAMPLE_IMAGE_URL}},
                ],
            )
        ]
    )
    print("[image_url_input]", response.content)


def generate_image() -> None:
    """Generate an image using a smolagents agent with a dall-e-3 image generation tool.

    Image generation models use /v1/images/generations, not /v1/chat/completions,
    so they cannot be used directly as a smolagents model. The correct smolagents
    pattern is to wrap image generation as a Tool and let an agent invoke it.

    Requires: OPENAI_API_KEY env variable and `pip install pillow openai`
    """

    @tool
    def generate_image_tool(prompt: str) -> str:
        """Generate an image from a text prompt using dall-e-3 and save it to disk.

        Args:
            prompt: Text description of the image to generate.

        Returns:
            Path to the saved image file.
        """
        client = openai.OpenAI()
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            response_format="b64_json",
            n=1,
        )
        import base64

        image_data = response.data[0].b64_json
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        image.save("generated_image.png")
        return "generated_image.png"

    model = OpenAIServerModel(model_id="gpt-4o")
    agent = ToolCallingAgent(tools=[generate_image_tool], model=model)
    result = agent.run(
        "Generate an image of a beautiful night sky featuring a bright full moon "
        "surrounded by sparkling stars, deep blue and black tones, soft glowing "
        "moonlight, dreamy atmosphere, highly detailed, serene and magical."
    )
    print("[generate_image]", result)


def pil_image_input() -> None:
    """Pass a PIL Image object alongside text as model input."""
    model = OpenAIServerModel(model_id="gpt-4o")
    pil_image = Image.open(requests.get(ICON_IMAGE_URL, stream=True).raw)
    response = model(
        messages=[
            ChatMessage(
                role=MessageRole.USER,
                content=[
                    {"type": "text", "text": "What do you see in this image?"},
                    {"type": "image", "image": pil_image},
                ],
            )
        ]
    )
    print("[pil_image_input]", response.content)


def image_output() -> None:
    """Ask the model to describe an image and inspect the response content."""
    model = OpenAIServerModel(model_id="gpt-4o")
    response = model(
        messages=[
            ChatMessage(
                role=MessageRole.USER,
                content=[
                    {"type": "text", "text": "Describe the colours in this image."},
                    {"type": "image_url", "image_url": {"url": SAMPLE_IMAGE_URL}},
                ],
            )
        ]
    )
    print("[image_output] role:", response.role)
    print("[image_output] content:", response.content)
    if isinstance(response.content, list):
        for element in response.content:
            if element.get("type") in ("image", "image_url"):
                print("[image_output] image element found:", element.get("type"))


def agent_with_image() -> None:
    """Run an agent with a PIL image passed via agent.run(images=[...])."""
    model = OpenAIServerModel(model_id="gpt-4o")
    agent = ToolCallingAgent(tools=[], model=model)
    pil_image = Image.open(requests.get(ICON_IMAGE_URL, stream=True).raw)
    result = agent.run("Describe what you see in the image.", images=[pil_image])
    print("[agent_with_image]", result)


if __name__ == "__main__":
    setup_tracing()
    generate_image()
    # image_url_input()
    # pil_image_input()
    # image_output()
    # agent_with_image()

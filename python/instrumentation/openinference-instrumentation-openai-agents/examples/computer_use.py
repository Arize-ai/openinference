"""Trace a computer-use agent with OpenInference.

The agent drives a tiny in-memory "display": a red button that turns green when
clicked. Each turn produces a `computer` TOOL span (the requested action) and an
LLM span whose input carries the screenshot as structured image content.

Prerequisites:
    pip install -r examples/requirements.txt
    export OPENAI_API_KEY=...        # needs access to the computer-use model
    phoenix serve                    # http://localhost:6006

Run:
    python examples/computer_use.py

Environment variables:
    COMPUTER_MODEL                         model to use (default: gpt-5.4)
    PHOENIX_PROJECT                        Phoenix project name (default: computer-use)
    OPENINFERENCE_HIDE_INPUT_IMAGES=true   drop screenshots from the trace
    OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH  redact screenshots longer than this
                                           (default 32000; real screenshots are
                                           usually larger, so raise it to keep them)

See the "Computer use" section of the package README for the attribute layout.
"""

import base64
import io
import os

from agents import Agent, Computer, ComputerTool, ModelSettings, Runner
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from PIL import Image, ImageDraw

from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor


class ButtonComputer(Computer):
    """An isolated display: clicking the button changes it from red to green."""

    clicked = False

    @property
    def environment(self):
        return "browser"

    @property
    def dimensions(self):
        return (640, 480)

    def screenshot(self) -> str:
        image = Image.new("RGB", self.dimensions, "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((200, 180, 440, 300), fill="green" if self.clicked else "red")
        draw.text((260, 230), "SUCCESS" if self.clicked else "CLICK ME", fill="white")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()

    def click(self, x, y, button):
        if button == "left" and 200 <= x <= 440 and 180 <= y <= 300:
            self.clicked = True

    def double_click(self, x, y):
        self.click(x, y, "left")

    def scroll(self, x, y, scroll_x, scroll_y):
        pass

    def type(self, text):
        pass

    def wait(self):
        pass

    def move(self, x, y):
        pass

    def keypress(self, keys):
        pass

    def drag(self, path):
        pass


def main():
    provider = TracerProvider(
        resource=Resource.create(
            {"openinference.project.name": os.getenv("PHOENIX_PROJECT", "computer-use")}
        )
    )
    provider.add_span_processor(
        SimpleSpanProcessor(OTLPSpanExporter("http://localhost:6006/v1/traces"))
    )
    # TraceConfig defaults are read from the OPENINFERENCE_* environment variables.
    OpenAIAgentsInstrumentor().instrument(tracer_provider=provider)
    computer = ButtonComputer()
    agent = Agent(
        name="Button tester",
        model=os.getenv("COMPUTER_MODEL", "gpt-5.4"),
        instructions="Click the red button, then inspect the screen and report its new color.",
        tools=[ComputerTool(computer=computer)],
        model_settings=ModelSettings(truncation="auto"),
    )
    try:
        result = Runner.run_sync(agent, "Test the button on the display.", max_turns=8)
        print(result.final_output)
        print(f"Button clicked: {computer.clicked}")
    finally:
        provider.force_flush()
        provider.shutdown()


if __name__ == "__main__":
    main()

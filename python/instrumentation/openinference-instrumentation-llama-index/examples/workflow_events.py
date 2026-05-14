import asyncio

from llama_index.core.llms import ChatMessage
from llama_index.llms.openai import OpenAI
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from workflows import Workflow, step
from workflows.events import Event, StartEvent, StopEvent

from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)

llm = OpenAI(model="gpt-4o-mini")


class KeywordsExtracted(Event):
    keywords: str


class ResearchWorkflow(Workflow):
    @step
    async def extract_keywords(self, ev: StartEvent) -> KeywordsExtracted:
        text = ev.get("text", "")
        response = await llm.achat(
            [
                ChatMessage(
                    role="system",
                    content=(
                        "Extract 5 keywords from the text. Reply with a comma-separated list only."
                    ),
                ),
                ChatMessage(role="user", content=text),
            ]
        )
        return KeywordsExtracted(keywords=str(response.message.content))

    @step
    async def summarize(self, ev: KeywordsExtracted) -> StopEvent:
        response = await llm.achat(
            [
                ChatMessage(
                    role="system", content="Write a one-sentence summary using these keywords."
                ),
                ChatMessage(role="user", content=ev.keywords),
            ]
        )
        return StopEvent(result=str(response.message.content))


TEXT = (
    "The James Webb Space Telescope is a space telescope designed to conduct "
    "infrared astronomy. Its high-resolution and high-sensitivity instruments "
    "allow it to view objects too old, distant, or faint for the Hubble Space "
    "Telescope, enabling investigations across many fields of astronomy."
)


async def run_full() -> None:
    handler = ResearchWorkflow().run(text=TEXT)
    result = await handler
    print("Summary:", result)


async def run_cancelled() -> None:
    step_started = asyncio.Event()

    class SlowResearchWorkflow(ResearchWorkflow):
        @step
        async def extract_keywords(self, ev: StartEvent) -> KeywordsExtracted:
            step_started.set()
            await asyncio.sleep(30)
            return KeywordsExtracted(keywords="")

    handler = SlowResearchWorkflow(timeout=30).run(text=TEXT)
    await step_started.wait()
    await handler.cancel_run()
    print("Run cancelled & SpanCancelledEvent recorded in trace.")


async def main() -> None:
    print("=== Workflow Events Example ===")
    print("* Full Run (WorkflowStepOutputEvent + WorkflowRunOutputEvent)")
    await run_full()
    print("* Cancelled Run (SpanCancelledEvent)")
    await run_cancelled()


if __name__ == "__main__":
    asyncio.run(main())

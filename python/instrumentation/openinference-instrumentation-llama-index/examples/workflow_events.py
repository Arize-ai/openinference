import asyncio

from llama_index.core.llms import ChatMessage
from llama_index.llms.openai import OpenAI
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from workflows import Workflow, step
from workflows.events import StartEvent, StopEvent

from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)

llm = OpenAI(model="gpt-4o-mini")


class SummarizeWorkflow(Workflow):
    @step
    async def summarize(self, ev: StartEvent) -> StopEvent:
        text = ev.get("text", "")
        response = await llm.achat(
            [
                ChatMessage(role="system", content="Summarize the following text in one sentence."),
                ChatMessage(role="user", content=text),
            ]
        )
        return StopEvent(result=str(response.message.content))


workflow = SummarizeWorkflow()

if __name__ == "__main__":
    text = (
        "The James Webb Space Telescope is a space telescope designed to conduct "
        "infrared astronomy. Its high-resolution and high-sensitivity instruments "
        "allow it to view objects too old, distant, or faint for the Hubble Space "
        "Telescope, enabling investigations across many fields of astronomy."
    )
    handler = workflow.run(text=text)
    result = asyncio.run(handler)
    print(result)

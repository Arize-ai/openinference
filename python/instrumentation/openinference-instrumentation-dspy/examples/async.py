import asyncio

import dspy
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from openinference.instrumentation.dspy import DSPyInstrumentor

endpoint = "http://localhost:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
DSPyInstrumentor().instrument(tracer_provider=tracer_provider)


dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))
predict = dspy.Predict("question->answer")


async def main() -> None:
    # Use acall() for async execution
    output = await predict.acall(question="why did a chicken cross the kitchen?")
    print(output)


asyncio.run(main())

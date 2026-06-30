import asyncio
from typing import Any

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


class MyModule(dspy.Module):
    def __init__(self) -> None:
        self.predict1 = dspy.ChainOfThought("question->answer")  # type: ignore
        self.predict2 = dspy.ChainOfThought("answer->simplified_answer")  # type: ignore

    async def aforward(self, question: str, **kwargs) -> Any:  # type: ignore
        # Execute predictions sequentially but asynchronously
        answer = await self.predict1.acall(question=question)
        return await self.predict2.acall(answer=answer)


async def main() -> None:
    mod = MyModule()
    result = await mod.acall(question="Why did a chicken cross the kitchen?")
    print(result)


asyncio.run(main())

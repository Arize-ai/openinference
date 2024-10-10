import logging
import sys

import dspy
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from openinference.instrumentation.dspy import DSPyInstrumentor

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

resource = Resource(attributes={})
tracer_provider = trace_sdk.TracerProvider(resource=resource)
span_console_exporter = ConsoleSpanExporter()
tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter=span_console_exporter))

endpoint = "http://localhost:6006/v1/traces"
span_otlp_exporter = OTLPSpanExporter(endpoint=endpoint)
tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter=span_otlp_exporter))


trace_api.set_tracer_provider(tracer_provider=tracer_provider)
DSPyInstrumentor().instrument()


class BasicQA(dspy.Signature):
    answer = dspy.OutputField(desc="often between 1 and 5 words")


class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(BasicQA)

    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)


if __name__ == "__main__":
    turbo = dspy.LM("openai/gpt-4", cache=False)
    colbertv2_wiki17_abstracts = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")
    dspy.settings.configure(
        lm=turbo,
        rm=colbertv2_wiki17_abstracts,
    )
    rag = RAG()
    output = rag("What's the capital of the united states?")
    print(output)

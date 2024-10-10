import dspy
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from openinference.instrumentation.dspy import DSPyInstrumentor

endpoint = "http://localhost:6006/v1/traces"

resource = Resource(attributes={})
tracer_provider = trace_sdk.TracerProvider(resource=resource)
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

trace_api.set_tracer_provider(tracer_provider=tracer_provider)


DSPyInstrumentor().instrument()


class BasicQA(dspy.Signature):
    """Answer questions with short factoid answers."""

    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")


if __name__ == "__main__":
    turbo = dspy.LM(model="openai/gpt-4")

    dspy.settings.configure(lm=turbo)

    # Define the predictor.
    generate_answer = dspy.Predict(BasicQA)

    # Call the predictor on a particular input.
    pred = generate_answer(
        question="What is the capital of the united states?"  # noqa: E501
    )  # noqa: E501
    print(f"Predicted Answer: {pred.answer}")

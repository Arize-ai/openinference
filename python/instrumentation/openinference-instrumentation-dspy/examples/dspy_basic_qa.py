import os

import dspy
from openinference.instrumentation.dspy import DSPyInstrumentor
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

resource = Resource(attributes={})
tracer_provider = trace_sdk.TracerProvider(resource=resource)
span_console_exporter = ConsoleSpanExporter()
tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter=span_console_exporter))
# Logs to the Phoenix Collector if running locally
if os.environ.get("PHOENIX_COLLECTOR_ENDPOINT"):
    endpoint = os.environ["PHOENIX_COLLECTOR_ENDPOINT"] + "/v1/traces"
    span_otlp_exporter = OTLPSpanExporter(endpoint=endpoint)
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter=span_otlp_exporter))

trace_api.set_tracer_provider(tracer_provider=tracer_provider)


DSPyInstrumentor().instrument()


class BasicQA(dspy.Signature):
    """Answer questions with short factoid answers."""

    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")


if __name__ == "__main__":
    turbo = dspy.OpenAI(model="gpt-3.5-turbo")

    dspy.settings.configure(lm=turbo)

    # Define the predictor.
    generate_answer = dspy.Predict(BasicQA)

    # Call the predictor on a particular input.
    pred = generate_answer(
        question="What is the capital of the united states?"  # noqa: E501
    )  # noqa: E501
    print(f"Predicted Answer: {pred.answer}")

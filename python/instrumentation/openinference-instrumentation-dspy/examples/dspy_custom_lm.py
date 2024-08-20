import os

import dspy
import requests
from dsp import LM
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from openinference.instrumentation.dspy import DSPyInstrumentor


class CustomLM(LM):
    """A Fake LM to test instrumentation"""

    def __init__(self, model, api_key):
        self.model = model
        self.api_key = api_key
        self.provider = "default"
        self.history = []

    def basic_request(self, prompt, **kwargs):
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "messages-2023-12-15",
            "content-type": "application/json",
        }

        data = {**kwargs, "model": self.model, "messages": [{"role": "user", "content": prompt}]}

        response = requests.post(self.base_url, headers=headers, json=data)
        response = response.json()

        self.history.append(
            {
                "prompt": prompt,
                "response": response,
                "kwargs": kwargs,
            }
        )

        return response

    def __call__(self, prompt, only_completed=True, return_sorted=False, **kwargs):
        response = self.request(prompt, **kwargs)

        completions = [result["text"] for result in response["content"]]

        return completions


# Logs to the Phoenix Collector if running locally
if os.environ.get("PHOENIX_COLLECTOR_ENDPOINT"):
    endpoint = os.environ["PHOENIX_COLLECTOR_ENDPOINT"] + "/v1/traces"
else:
    # Assume a local collector
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
    turbo = dspy.OpenAI(model="gpt-3.5-turbo")

    dspy.settings.configure(lm=CustomLM("fake-model", "fake-api-key"))

    # Define the predictor.
    generate_answer = dspy.Predict(BasicQA)

    # Call the predictor on a particular input.
    pred = generate_answer(
        question="What is the capital of the united states?"  # noqa: E501
    )  # noqa: E501
    print(f"Predicted Answer: {pred.answer}")

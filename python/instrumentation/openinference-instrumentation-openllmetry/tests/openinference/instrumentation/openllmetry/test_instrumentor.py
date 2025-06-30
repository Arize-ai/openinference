import os
from typing import Mapping, cast

import openai
from openai.types.chat import ChatCompletionUserMessageParam
from opentelemetry.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.util.types import AttributeValue
from phoenix.otel import register

from openinference.instrumentation.openllmetry import OpenInferenceSpanProcessor
from openinference.semconv.trace import SpanAttributes


def is_openinference_span(span: ReadableSpan) -> bool:
    """Check if a span is an OpenInference span."""
    if span.attributes is None:
        return False
    return SpanAttributes.OPENINFERENCE_SPAN_KIND in span.attributes


# Set your OpenAI API key
api_key: str = os.getenv("OPENAI_API_KEY") or ""
os.environ["OPENAI_API_KEY"] = api_key


class TestOpenLLMetryInstrumentor:
    def test_openllmetry_instrumentor(self) -> None:
        in_memory_span_exporter = InMemorySpanExporter()
        in_memory_span_exporter.clear()

        # Set up the tracer provider
        tracer_provider = register(
            project_name="default"  # Phoenix project name
        )

        tracer_provider.add_span_processor(OpenInferenceSpanProcessor())

        tracer_provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))

        OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

        # Define and invoke your OpenAI model
        client = openai.OpenAI()

        messages: list[ChatCompletionUserMessageParam] = [
            {"role": "user", "content": "What is the capital of Yemen?"}
        ]

        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=messages,
        )

        # Basic assertion on the result
        assert response.choices[0].message.content is not None
        assert (
            "Sanaa" in response.choices[0].message.content
            or "Sana'a" in response.choices[0].message.content
        )

        # Get spans
        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) > 0
        for span in spans:
            # Check span attributes
            attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))

            # Verify it's an OpenInference span
            assert is_openinference_span(span)

            # Check that we have input and output values
            assert SpanAttributes.INPUT_VALUE in attributes
            assert SpanAttributes.OUTPUT_VALUE in attributes

            # Verify the model name is captured (exact attribute may vary)
            model_name_attr = next(
                (k for k in attributes if "model" in k.lower() and "name" in k.lower()), None
            )
            assert model_name_attr is not None, "Model name attribute not found"
            assert "gpt-4" in str(attributes[model_name_attr])

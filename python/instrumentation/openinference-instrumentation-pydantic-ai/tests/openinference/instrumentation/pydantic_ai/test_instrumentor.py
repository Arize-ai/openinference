from typing import Mapping, cast

import pytest
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.util.types import AttributeValue
from pydantic import BaseModel

# Import necessary Pydantic AI components
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from openinference.instrumentation.pydantic_ai.span_processor import (
    OpenInferenceSpanProcessor,
)
from openinference.instrumentation.pydantic_ai.utils import is_openinference_span
from openinference.semconv.trace import (
    MessageAttributes,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)


class TestPydanticAIInstrumentation:
    @pytest.mark.vcr(
        decode_compressed_response=True,
        before_record_request=lambda _: _.headers.clear() or _,
        before_record_response=lambda _: {**_, "headers": {}},
    )
    def test_pydantic_ai_instrumentation(
        self, in_memory_span_exporter: InMemorySpanExporter, tracer_provider: TracerProvider
    ) -> None:
        """Test that Pydantic AI with instrument=True is properly traced."""
        in_memory_span_exporter.clear()
        trace.set_tracer_provider(tracer_provider)
        exporter = OTLPSpanExporter()
        tracer_provider.add_span_processor(OpenInferenceSpanProcessor())
        tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))

        class LocationModel(BaseModel):
            city: str
            country: str

        model = OpenAIModel("gpt-4o", provider=OpenAIProvider(api_key="sk-"))
        agent = Agent(model, output_type=LocationModel, instrument=True)
        result = agent.run_sync("The windy city in the US of A.")
        assert result is not None
        spans = in_memory_span_exporter.get_finished_spans()

        # Find the openinference span
        openinference_span = next((span for span in spans if is_openinference_span(span)), None)
        assert openinference_span is not None

        # Verify span attributes
        attributes = dict(cast(Mapping[str, AttributeValue], openinference_span.attributes))
        assert (
            attributes.get(SpanAttributes.OPENINFERENCE_SPAN_KIND)
            == OpenInferenceSpanKindValues.LLM.value
        )
        assert attributes.get(SpanAttributes.LLM_SYSTEM) == "openai"
        assert attributes.get(SpanAttributes.LLM_MODEL_NAME) == "gpt-4o"

        assert attributes.get(SpanAttributes.INPUT_VALUE) == "The windy city in the US of A."
        assert (
            attributes.get(SpanAttributes.OUTPUT_VALUE)
            == '{"city":"Chicago","country":"United States of America"}'
        )

        prompt_tokens = attributes.get(SpanAttributes.LLM_TOKEN_COUNT_PROMPT)
        completion_tokens = attributes.get(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION)
        total_tokens = attributes.get(SpanAttributes.LLM_TOKEN_COUNT_TOTAL)
        assert isinstance(prompt_tokens, int)
        assert isinstance(completion_tokens, int)
        assert isinstance(total_tokens, int)
        assert total_tokens == prompt_tokens + completion_tokens
        # Check that tools are present
        assert (
            attributes.get(f"{SpanAttributes.LLM_TOOLS}.0.{SpanAttributes.TOOL_NAME}")
            == "final_result"
        )
        assert (
            attributes.get(f"{SpanAttributes.LLM_TOOLS}.0.{SpanAttributes.TOOL_DESCRIPTION}")
            == "The final response which ends this conversation"
        )
        assert (
            attributes.get(
                f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}"
            )
            == "user"
        )

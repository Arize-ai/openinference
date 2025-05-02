from typing import Dict, Any, cast, Mapping
import pytest
from pydantic import BaseModel

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.util.types import AttributeValue
from openinference.semconv.trace import SpanAttributes

from openinference.instrumentation.pydantic.utils import is_openinference_span
from openinference.instrumentation.pydantic.span_processor import OpenInferenceSimpleSpanProcessor

from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

# Import necessary Pydantic AI components
from pydantic_ai import Agent


class TestPydanticAIInstrumentation:
    
    def test_pydantic_ai_instrumentation(
        self, in_memory_span_exporter: InMemorySpanExporter, tracer_provider: TracerProvider
    ) -> None:
        """Test that Pydantic AI with instrument=True is properly traced."""
        # Clear any previous spans
        in_memory_span_exporter.clear()
        
        # Set the tracer provider
        trace.set_tracer_provider(tracer_provider)
        
        # Add OpenInferenceSimpleSpanProcessor to handle spans
        tracer_provider.add_span_processor(OpenInferenceSimpleSpanProcessor(in_memory_span_exporter))
        
        # Define a simple Pydantic model
        class LocationModel(BaseModel):
            city: str
            country: str
        
        model = OpenAIModel(
            'gpt-4o', 
            provider=OpenAIProvider(api_key='sk-')
        )
        agent = Agent(
            model,
            output_type=LocationModel,
            instrument=True
        )
        
        # Run the agent with a simple prompt
        result = agent.run_sync('The windy city in the US of A.')
        
        # Basic assertion on the result
        assert result.output is not None
        assert result.output.city == "Chicago"
        assert result.output.country == "USA" or result.output.country == "United States"
        
        # Get spans
        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) > 0
        
        # Find the LLM span
        llm_span = None
        for span in spans:
            if "llm" in span.name.lower() or "model" in span.name.lower():
                llm_span = span
                break
        
        assert llm_span is not None, "No LLM span found"
        
        # Check span attributes
        attributes = dict(cast(Mapping[str, AttributeValue], llm_span.attributes))
        
        # Verify it's an OpenInference span
        assert is_openinference_span(llm_span)
        
        # Check that we have input and output values
        assert SpanAttributes.INPUT_VALUE in attributes
        assert SpanAttributes.OUTPUT_VALUE in attributes
        
        # Verify the model name is captured (exact attribute may vary)
        model_name_attr = next((k for k in attributes.keys() if "model" in k.lower() and "name" in k.lower()), None)
        assert model_name_attr is not None, "Model name attribute not found"
        assert "gpt-4o" in str(attributes[model_name_attr])

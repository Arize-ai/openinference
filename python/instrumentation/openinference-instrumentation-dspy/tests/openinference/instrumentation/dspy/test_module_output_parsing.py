import json
from unittest.mock import Mock, patch

import dspy
import pytest
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation.dspy import DSPyInstrumentor
from openinference.semconv.trace import SpanAttributes


class DisappointmentScorer(dspy.Module):
    """A DSPy module that scores disappointment level"""
    
    def __init__(self):
        super().__init__()
        self.scorer = dspy.ChainOfThought("recording_file_name -> disappointment_score, disappointment_reasoning")
    
    def forward(self, recording_file_name):
        # Mock the scorer to return a prediction with the expected fields
        prediction = dspy.Prediction(
            disappointment_score=8,
            disappointment_reasoning="The customer expressed frustration multiple times"
        )
        return prediction


@pytest.fixture
def in_memory_span_exporter():
    return InMemorySpanExporter()


@pytest.fixture
def tracer_provider(in_memory_span_exporter):
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))
    return tracer_provider


@pytest.fixture
def setup_tracing(tracer_provider):
    trace_api.set_tracer_provider(tracer_provider)
    DSPyInstrumentor().instrument()
    yield
    DSPyInstrumentor().uninstrument()


def test_module_forward_output_parsing(setup_tracing, in_memory_span_exporter):
    """Test that Module.forward output is properly parsed as JSON with individual fields"""
    
    # Create and run the module
    scorer = DisappointmentScorer()
    result = scorer(recording_file_name="customer_call_123.wav")
    
    # Get the spans
    spans = in_memory_span_exporter.get_finished_spans()
    
    # Find the Module.forward span
    module_span = None
    for span in spans:
        if span.name == "DisappointmentScorer.forward":
            module_span = span
            break
    
    assert module_span is not None, "Module.forward span not found"
    
    # Get the output value
    output_value = module_span.attributes.get(SpanAttributes.OUTPUT_VALUE)
    assert output_value is not None, "Output value not found in span attributes"
    
    # Parse the output value as JSON
    output_dict = json.loads(output_value)
    
    # Print for debugging
    print(f"Output value type: {type(output_value)}")
    print(f"Output value: {output_value}")
    print(f"Parsed output dict: {output_dict}")
    
    # The bug: output_dict should contain the individual fields, not be a string representation
    # Currently it's likely something like: '{"disappointment_score": 8, "disappointment_reasoning": "..."}'
    # But it should be a proper dict that can be accessed
    assert isinstance(output_dict, dict), "Output should be a dictionary, not a string"
    assert "disappointment_score" in output_dict, "disappointment_score field should be in output"
    assert "disappointment_reasoning" in output_dict, "disappointment_reasoning field should be in output"
    assert output_dict["disappointment_score"] == 8
    assert output_dict["disappointment_reasoning"] == "The customer expressed frustration multiple times" 
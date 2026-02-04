import os
import logging
from dotenv import load_dotenv
load_dotenv()

# Enable logging to see instrumentation messages
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

from anthropic import Anthropic
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation.anthropic import AnthropicInstrumentor

# Set your Anthropic API key as an environment variable
# Option 1: Set it in your shell before running:
#   export ANTHROPIC_API_KEY="your-api-key-here"
# Option 2: Set it directly in code (not recommended for production):
#   os.environ["ANTHROPIC_API_KEY"] = "your-api-key-here"
# Option 3: Use python-dotenv to load from .env file:
#   from dotenv import load_dotenv
#   load_dotenv()

# Configure AnthropicInstrumentor with Phoenix endpoint
endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()

# Add both Phoenix exporter and in-memory exporter for verification
phoenix_exporter = OTLPSpanExporter(endpoint)
memory_exporter = InMemorySpanExporter()
tracer_provider.add_span_processor(SimpleSpanProcessor(phoenix_exporter))
tracer_provider.add_span_processor(SimpleSpanProcessor(memory_exporter))

AnthropicInstrumentor().instrument(tracer_provider=tracer_provider)

# Optional: Verify instrumentation was applied
try:
    from anthropic.resources.beta.messages import Messages as BetaMessages
    from wrapt import BoundFunctionWrapper, FunctionWrapper
    is_wrapped = isinstance(BetaMessages.parse, (BoundFunctionWrapper, FunctionWrapper))
    print(f"✓ Beta API instrumentation {'enabled' if is_wrapped else 'not detected'}")
except Exception:
    pass  # Silently skip if beta API not available

client = Anthropic()

# Example usage of beta.messages.parse() for structured outputs
# Note: The beta API and parse() method may have specific requirements:
# - Check Anthropic documentation for which models support beta.parse()
# - You may need to specify additional parameters like response_format or schema
# - The API endpoint and parameters may differ from the standard messages.create()

try:
    response = client.beta.messages.parse(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "Extract the key information from this text.",
            }
        ],
        model="claude-sonnet-4-5-20250929",  # Update with a model that supports beta.parse()
        # Additional parameters for structured parsing may be required:
        # response_format={"type": "json_schema", ...},
        # or other beta-specific parameters
    )
    print("Response:", response)
    
    # Verify instrumentation worked
    spans = memory_exporter.get_finished_spans()
    if spans:
        span = spans[0]
        print(f"\n Instrumentation verified!")
        print(f"   Span name: {span.name}")
        print(f"   Number of spans: {len(spans)}")
        if span.attributes:
            attrs = dict(span.attributes)
            print(f"   Model: {attrs.get('llm.model_name', 'N/A')}")
            print(f"   Provider: {attrs.get('llm.provider', 'N/A')}")
            print(f"   Input messages captured: {'llm.input.messages.0.content' in str(attrs)}")
            print(f"   Total attributes: {len(attrs)}")
        print(f"\nView traces at: http://localhost:6006")
    else:
        print("\No spans found - instrumentation may not have worked")
        
except Exception as e:
    print(f"Error calling beta.messages.parse(): {e}")
    print("\nNote: The beta.parse() API may:")
    print("  1. Require specific model versions")
    print("  2. Need additional parameters (response_format, schema, etc.)")
    print("  3. Not be available in your Anthropic SDK version")
    print("  4. Require API access permissions")
    print("\nCheck Anthropic documentation for beta API requirements.")

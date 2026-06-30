from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from smolagents import LiteLLMModel
from smolagents.agents import CodeAgent

from openinference.instrumentation.smolagents import SmolagentsInstrumentor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

SmolagentsInstrumentor().instrument(tracer_provider=tracer_provider)

model_params = {"thinking": {"type": "enabled", "budget_tokens": 4000}}
model = LiteLLMModel(model_id="anthropic/claude-3-7-sonnet-20250219", **model_params)
agent = CodeAgent(tools=[], model=model, add_base_tools=False)
print(agent.run("What's the weather like in Paris?"))

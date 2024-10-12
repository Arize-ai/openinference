from configs.agents import triage_agent
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from swarm.repl import run_demo_loop

from openinference.instrumentation.openai import OpenAIInstrumentor
from openinference.instrumentation.swarm import SwarmInstrumentor
from openinference.semconv.resource import ResourceAttributes

project_name = "swarm"

endpoint = "http://localhost:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider(
    resource=Resource({ResourceAttributes.PROJECT_NAME: project_name})
)
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
SwarmInstrumentor().instrument(tracer_provider=tracer_provider)


context_variables = {
    "customer_context": """Here is what you know about the customer's details:
1. CUSTOMER_ID: customer_12345
2. NAME: John Doe
3. PHONE_NUMBER: (123) 456-7890
4. EMAIL: johndoe@example.com
5. STATUS: Premium
6. ACCOUNT_STATUS: Active
7. BALANCE: $0.00
8. LOCATION: 1234 Main St, San Francisco, CA 94123, USA
""",
    "flight_context": """The customer has an upcoming flight from LGA (Laguardia) in NYC to LAX in Los Angeles.
The flight # is 1919. The flight departure date is 3pm ET, 5/21/2024.""",
}
if __name__ == "__main__":
    run_demo_loop(triage_agent, context_variables=context_variables, debug=True)

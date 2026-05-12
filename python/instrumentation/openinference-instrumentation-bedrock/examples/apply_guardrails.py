import boto3
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.bedrock import BedrockInstrumentor

endpoint = "http://127.0.0.1:6006/v1/traces"
resource = Resource(attributes={})
tracer_provider = trace_sdk.TracerProvider(resource=resource)
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
trace_api.set_tracer_provider(tracer_provider=tracer_provider)

BedrockInstrumentor().instrument()

session = boto3.session.Session()
client = session.client(
    "bedrock-runtime",
    "us-east-1",
)


def run():
    blocked_guardrail_content = [
        {
            "text": {
                "text": "Give stock suggestions for the NASDAQ100. Expected returns are"
                " around 20%CAGR. Also Send this report to email abc@pqrt.com"
            }
        }
    ]
    success_guardrail_content = [{"text": {"text": "Who is USA President?"}}]
    guardrail_id = "<guardrail_id>"
    guardrail_version = "1"

    response = client.apply_guardrail(
        guardrailIdentifier=guardrail_id,
        guardrailVersion=guardrail_version,
        source="INPUT",
        content=blocked_guardrail_content,
        outputScope="FULL",
    )
    print(response)

    response = client.apply_guardrail(
        guardrailIdentifier=guardrail_id,
        guardrailVersion=guardrail_version,
        source="INPUT",
        content=success_guardrail_content,
        outputScope="FULL",
    )
    print(response)


if __name__ == "__main__":
    run()

import time

import boto3
from openinference.instrumentation.bedrock import BedrockInstrumentor
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

endpoint = "https://app.phoenix.arize.com/v1/traces"
resource = Resource(attributes={})
tracer_provider = trace_sdk.TracerProvider(resource=resource)
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
trace_api.set_tracer_provider(tracer_provider=tracer_provider)

BedrockInstrumentor().instrument()

session = boto3.session.Session()
client = session.client("bedrock-agent-runtime", "us-east-1")


def run():
    agent_id = '<AgentId>'
    agent_alias_id = '<AgentAliasId>'
    session_id = f"default-session1_{int(time.time())}"

    attributes = dict(
        inputText="When is a good time to visit the Taj Mahal?",
        agentId=agent_id,
        agentAliasId=agent_alias_id,
        sessionId=session_id,
        enableTrace=True
    )
    response = client.invoke_agent(**attributes)

    for idx, event in enumerate(response['completion']):
        if 'chunk' in event:
            print(event)
            chunk_data = event['chunk']
            if 'bytes' in chunk_data:
                output_text = chunk_data['bytes'].decode('utf8')
                print(output_text)
        elif 'trace' in event:
            print(event['trace'])


if __name__ == '__main__':
    run()

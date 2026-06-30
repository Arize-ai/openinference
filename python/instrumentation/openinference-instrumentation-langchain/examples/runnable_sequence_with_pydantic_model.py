from typing import List, Optional

import boto3
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, SystemMessage
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from pydantic import BaseModel, Field

from openinference.instrumentation.langchain import LangChainInstrumentor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

LangChainInstrumentor().instrument(tracer_provider=tracer_provider)


class ExtractForm(BaseModel):
    description: str = Field(..., description="The description if the issue or work required")
    status_name: Optional[str] = Field(
        None,
        description="The status name or priority to assign the issue.  "
        "Only use if explicitly specified.",
    )
    tag_names: Optional[List[str]] = Field(
        None, description="The tag name to assign the issue.  Only use if explicitly specified."
    )
    assignee: Optional[str] = Field(
        None, description="The individual or team assigned to complete the task, if specified"
    )


llm = ChatBedrock(
    client=boto3.client(service_name="bedrock-runtime", region_name="us-east-1"),
    model="anthropic.claude-3-haiku-20240307-v1:0",
)

messages = [
    SystemMessage(content="Extract fields from the users message"),
    HumanMessage(
        content="This is the message: The pipe is broken, set status to Open tag as Plumbing"
    ),
]
output = llm.with_structured_output(ExtractForm).invoke(messages)

import json
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

import vcr
from brotli import brotli
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

endpoint = "http://127.0.0.1:4317"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

LangChainInstrumentor().instrument(tracer_provider=tracer_provider)


class Joke(BaseModel):
    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline to the joke")


llm = ChatOpenAI(model="gpt-4o-mini", temperature=1)
with ExitStack() as stack:
    stack.enter_context(d := TemporaryDirectory())
    cass = stack.enter_context(
        vcr.use_cassette(
            Path(d.name) / Path(__file__).with_suffix(".yaml").name,
            filter_headers=["authorization"],
            decode_compressed_response=True,
            ignore_localhost=True,
        )
    )
    llm.with_structured_output(Joke).invoke("Tell me a joke.")
pairs = [
    {
        f"REQUEST-{i}": json.loads(cass.requests[i].body),
        f"RESPONSE-{i}": json.loads(brotli.decompress(cass.responses[i]["body"]["string"])),
    }
    for i in range(len(cass.requests))
]
with open(Path(__file__).with_suffix(".vcr.json"), "w") as f:
    json.dump(pairs, f, indent=2)

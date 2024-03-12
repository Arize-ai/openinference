"""
This example demonstrates how to add metadata on a per-request basis. Here we record the
referer from request headers, but a more common use case to record the user_id as shown in
the following example:
https://github.com/langchain-ai/langserve/blob/4bf7b0f4386f9bab0026d28e0003584f654ddbbe/examples/chat_with_persistence_and_user/server.py#L107
"""  # noqa: E501

from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, Request
from langchain_core.runnables import ConfigurableField
from langchain_openai import ChatOpenAI
from langserve import add_routes
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
trace_api.set_tracer_provider(tracer_provider)
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

LangChainInstrumentor().instrument()


async def per_req_config_modifier(config: Dict[str, Any], request: Request) -> Dict[str, Any]:
    config["configurable"] = {"metadata": {"referer": request.headers["referer"]}}
    return config


app = FastAPI()
add_routes(
    app,
    ChatOpenAI().configurable_fields(metadata=ConfigurableField(id="metadata")),
    per_req_config_modifier=per_req_config_modifier,
    path="/openai",
)
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)

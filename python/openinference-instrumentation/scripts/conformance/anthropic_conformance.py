# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "anthropic",
#     "opentelemetry-sdk",
#     "opentelemetry-exporter-otlp-proto-grpc",
#     "openinference-instrumentation",
#     "openinference-instrumentation-anthropic",
#     "openinference-semantic-conventions",
# ]
#
# [tool.uv.sources.openinference-instrumentation]
# path = "../../"
# editable = true
#
# [tool.uv.sources.openinference-instrumentation-anthropic]
# path = "../../../instrumentation/openinference-instrumentation-anthropic"
# editable = true
#
# [tool.uv.sources.openinference-semantic-conventions]
# path = "../../../openinference-semantic-conventions"
# editable = true
# ///
"""Run the OpenInference Anthropic instrumentor against the mock server.

Exports OTel traces over OTLP gRPC to whichever endpoint is in
`OTEL_EXPORTER_OTLP_ENDPOINT`. The dual-write to OTel GenAI semantic
conventions is enabled via `TraceConfig(enable_genai_semconv=True)`.
"""

import json
import os
import sys

import anthropic
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from openinference.instrumentation import OITracer, TraceConfig, using_session
from openinference.instrumentation.anthropic import AnthropicInstrumentor
from openinference.semconv.trace import (
    OpenInferenceMimeTypeValues,
    SpanAttributes,
)


def main() -> None:
    endpoint = os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"]
    base_url = os.environ["MOCK_LLM_URL"]

    tp = TracerProvider()
    tp.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True)))
    trace.set_tracer_provider(tp)

    AnthropicInstrumentor().instrument(
        tracer_provider=tp,
        config=TraceConfig(enable_genai_semconv=True),
    )

    client = anthropic.Anthropic(base_url=base_url, api_key="mock-key")

    print("[chat] basic message")
    with using_session("conformance-session-001"):
        resp = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=64,
            temperature=0.5,
            top_p=0.9,
            top_k=40,
            stop_sequences=["END"],
            system="You are a helpful conformance test assistant.",
            messages=[{"role": "user", "content": "Hello!"}],
        )
    print(f"  -> {resp.content[0].text[:60]}")

    print("[chat] tool use")
    tool_resp = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=64,
        tools=[
            {
                "name": "get_weather",
                "description": "Get the current weather for a city",
                "input_schema": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            }
        ],
        messages=[{"role": "user", "content": "What's the weather in Seattle?"}],
    )
    tool_use = next(b for b in tool_resp.content if b.type == "tool_use")
    print(f"  -> stop_reason={tool_resp.stop_reason}")

    oi_tracer = OITracer(
        trace.get_tracer(__name__),
        TraceConfig(enable_genai_semconv=True),
    )

    print("[tool] simulated execution")
    tool_result = {"forecast": "sunny", "temperature_f": 72}
    with oi_tracer.start_as_current_span(
        tool_use.name,
        openinference_span_kind="tool",
        attributes={
            SpanAttributes.TOOL_NAME: tool_use.name,
            SpanAttributes.TOOL_DESCRIPTION: "Get the current weather for a city",
            SpanAttributes.TOOL_ID: tool_use.id,
            SpanAttributes.TOOL_PARAMETERS: json.dumps(tool_use.input),
            SpanAttributes.OUTPUT_VALUE: json.dumps(tool_result),
            SpanAttributes.OUTPUT_MIME_TYPE: OpenInferenceMimeTypeValues.JSON.value,
        },
    ):
        pass
    print(f"  -> {tool_result}")

    print("[embedding] simulated vector embedding")
    with oi_tracer.start_as_current_span(
        "embed",
        openinference_span_kind="embedding",
        attributes={
            SpanAttributes.EMBEDDING_MODEL_NAME: "voyage-3",
            SpanAttributes.EMBEDDING_INVOCATION_PARAMETERS: json.dumps(
                {"encoding_format": "float"}
            ),
            f"{SpanAttributes.EMBEDDING_EMBEDDINGS}.0.embedding.text": "Hello, world!",
            f"{SpanAttributes.EMBEDDING_EMBEDDINGS}.0.embedding.vector": [0.1] * 8,
        },
    ):
        pass

    print("[retrieval] simulated document fetch")
    with oi_tracer.start_as_current_span(
        "retrieve_docs",
        openinference_span_kind="retriever",
        attributes={
            SpanAttributes.INPUT_VALUE: "What's the weather in Seattle?",
            SpanAttributes.INPUT_MIME_TYPE: OpenInferenceMimeTypeValues.TEXT.value,
            f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.0.document.id": "doc-001",
            f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.0.document.score": 0.95,
            f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.0.document.content": (
                "Seattle weather is typically rainy in winter and mild in summer."
            ),
        },
    ):
        pass

    tp.force_flush(timeout_millis=5000)
    tp.shutdown()


if __name__ == "__main__":
    if "--prewarm" in sys.argv[1:]:
        sys.exit(0)
    main()

# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "openai",
#     "opentelemetry-sdk",
#     "opentelemetry-exporter-otlp-proto-grpc",
#     "openinference-instrumentation",
#     "openinference-instrumentation-openai",
#     "openinference-semantic-conventions",
# ]
#
# [tool.uv.sources.openinference-instrumentation]
# path = "../../"
# editable = true
#
# [tool.uv.sources.openinference-instrumentation-openai]
# path = "../../../instrumentation/openinference-instrumentation-openai"
# editable = true
#
# [tool.uv.sources.openinference-semantic-conventions]
# path = "../../../openinference-semantic-conventions"
# editable = true
# ///
"""Run the OpenInference OpenAI instrumentor against the mock server.

Exercises chat (with all OpenAI-supported request parameters), tool calling, and
embeddings. The dual-write to OTel GenAI semantic conventions is enabled via
`TraceConfig(enable_genai_semconv=True)`.
"""

import os
import sys

from openai import OpenAI
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from openinference.instrumentation import TraceConfig, using_session
from openinference.instrumentation.openai import OpenAIInstrumentor


def main() -> None:
    endpoint = os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"]
    base_url = os.environ["MOCK_LLM_URL"] + "/v1"

    tp = TracerProvider()
    tp.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True)))
    trace.set_tracer_provider(tp)

    OpenAIInstrumentor().instrument(
        tracer_provider=tp,
        config=TraceConfig(enable_genai_semconv=True),
    )

    client = OpenAI(base_url=base_url, api_key="mock-key")

    print("[chat] basic message")
    with using_session("conformance-session-openai-001"):
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=64,
            temperature=0.5,
            top_p=0.9,
            n=2,
            seed=42,
            frequency_penalty=0.1,
            presence_penalty=0.1,
            stop=["END"],
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful conformance test assistant. Reply with JSON.",
                },
                {"role": "user", "content": "Hello!"},
            ],
        )
    print(f"  -> {resp.choices[0].message.content[:60]}")

    print("[chat] tool use")
    tool_resp = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=64,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                },
            }
        ],
        messages=[{"role": "user", "content": "What's the weather in Seattle?"}],
    )
    print(f"  -> finish_reason={tool_resp.choices[0].finish_reason}")

    print("[embeddings] embed")
    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input="Hello, world!",
        encoding_format="float",
    )
    print(f"  -> dim={len(emb.data[0].embedding)}")

    print("[responses] basic")
    resp_api = client.responses.create(
        model="gpt-4o-mini",
        input="Hello!",
        instructions="You are a helpful conformance test assistant.",
        max_output_tokens=64,
        temperature=0.5,
        top_p=0.9,
    )
    print(f"  -> {(resp_api.output_text or '')[:60]}")

    print("[responses] tool use")
    tool_resp_api = client.responses.create(
        model="gpt-4o-mini",
        input="What's the weather in Seattle?",
        max_output_tokens=64,
        tools=[
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get the current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
                "strict": False,
            }
        ],
    )
    print(f"  -> status={tool_resp_api.status}")

    tp.force_flush(timeout_millis=5000)
    tp.shutdown()


if __name__ == "__main__":
    if "--prewarm" in sys.argv[1:]:
        sys.exit(0)
    main()

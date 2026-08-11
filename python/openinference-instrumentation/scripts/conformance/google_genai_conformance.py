# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "google-genai",
#     "opentelemetry-sdk",
#     "opentelemetry-exporter-otlp-proto-grpc",
#     "openinference-instrumentation",
#     "openinference-instrumentation-google-genai",
#     "openinference-semantic-conventions",
# ]
#
# [tool.uv.sources.openinference-instrumentation]
# path = "../../"
# editable = true
#
# [tool.uv.sources.openinference-instrumentation-google-genai]
# path = "../../../instrumentation/openinference-instrumentation-google-genai"
# editable = true
#
# [tool.uv.sources.openinference-semantic-conventions]
# path = "../../../openinference-semantic-conventions"
# editable = true
# ///
"""Run the OpenInference Google GenAI instrumentor against the mock server."""

import os
import sys

from google import genai
from google.genai import types as gt
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from openinference.instrumentation import TraceConfig, using_session
from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor


def main() -> None:
    endpoint = os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"]
    base_url = os.environ["MOCK_LLM_URL"]

    tp = TracerProvider()
    tp.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True)))
    trace.set_tracer_provider(tp)

    GoogleGenAIInstrumentor().instrument(
        tracer_provider=tp,
        config=TraceConfig(enable_genai_semconv=True),
    )

    client = genai.Client(
        api_key="mock-key",
        http_options=gt.HttpOptions(base_url=base_url, api_version="v1beta"),
    )

    print("[generate_content] basic")
    with using_session("conformance-session-google-001"):
        resp = client.models.generate_content(
            model="gemini-2.0-flash",
            contents="Hello!",
            config=gt.GenerateContentConfig(
                temperature=0.5,
                top_p=0.9,
                top_k=40,
                max_output_tokens=64,
                stop_sequences=["END"],
                seed=42,
                candidate_count=1,
                system_instruction="You are a helpful conformance test assistant.",
            ),
        )
    print(f"  -> {(resp.text or '')[:60]}")

    print("[generate_content] tool use")
    tool_resp = client.models.generate_content(
        model="gemini-2.0-flash",
        contents="What's the weather in Seattle?",
        config=gt.GenerateContentConfig(
            max_output_tokens=64,
            tools=[
                gt.Tool(
                    function_declarations=[
                        gt.FunctionDeclaration(
                            name="get_weather",
                            description="Get the current weather for a city",
                            parameters=gt.Schema(
                                type="OBJECT",
                                properties={"location": gt.Schema(type="STRING")},
                                required=["location"],
                            ),
                        )
                    ]
                )
            ],
        ),
    )
    finish_reason = tool_resp.candidates[0].finish_reason if tool_resp.candidates else None
    print(f"  -> finish_reason={finish_reason}")

    tp.force_flush(timeout_millis=5000)
    tp.shutdown()


if __name__ == "__main__":
    if "--prewarm" in sys.argv[1:]:
        sys.exit(0)
    main()

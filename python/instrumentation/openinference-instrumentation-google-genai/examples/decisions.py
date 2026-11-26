from google import genai
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter,  # type: ignore[import-not-found]
)
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor

endpoint = "http://localhost:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
GoogleGenAIInstrumentor().instrument(tracer_provider=tracer_provider)

# Make sure to set the GEMINI_API_KEY environment variable.
#
# The Gemini Decisions API (gemini-decision-1) requires google-genai >= 3.2 and the
# v1beta API version. Each call is traced as a single OpenInference DECISION span.


if __name__ == "__main__":
    client = genai.Client(http_options={"api_version": "v1beta"})

    result = client.decisions.create(
        model="gemini-decision-1",
        input="Hi, I was charged twice for order #4242 and would like my money back.",
        questions={
            "is_refund": {
                "type": "noul",
                "instructions": "Is this a refund request?",
            },
            "department": {
                "type": "choice",
                "instructions": "Which department should handle this?",
                "criteria": {
                    "billing": "Charges and refunds",
                    "shipping": "Delivery and returns",
                    "other": "",
                },
            },
            "urgency": {
                "type": "score",
                "instructions": "How urgent is this request?",
                "criteria": ["Not urgent", "Somewhat urgent", "Very urgent"],
            },
        },
    )

    print("is_refund:", result.answers["is_refund"].noul)
    print("department:", result.answers["department"].choice)
    print("urgency:", result.answers["urgency"].score)

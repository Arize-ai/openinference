"""
Trace an AG2 agent that returns a structured, schema-validated reply.

Passing a pydantic model as `response_format` on `LLMConfig` makes the agent reply with
JSON matching that schema, as described in the AG2 structured outputs guide. The agent
span's output value is the serialized model, so the trace shows exactly what downstream
code will parse.

1. Run Phoenix locally: `pip install arize-phoenix && phoenix serve`
2. Install dependencies: `pip install -r requirements.txt`
3. Set your API key: `export OPENAI_API_KEY=<your-key>`
4. Run this example: `python structured_output.py`
5. View the traces at http://localhost:6006 under the `ag2-structured-output` project.
"""

import json
import os

from autogen import ConversableAgent, LLMConfig
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from pydantic import BaseModel

from openinference.instrumentation.ag2 import AG2Instrumentor
from openinference.instrumentation.openai import OpenAIInstrumentor
from openinference.semconv.resource import ResourceAttributes

endpoint = "http://localhost:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider(
    resource=Resource({ResourceAttributes.PROJECT_NAME: "ag2-structured-output"})
)
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
AG2Instrumentor().instrument(tracer_provider=tracer_provider)
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)


class TransactionAuditEntry(BaseModel):
    vendor: str
    amount: float
    memo: str
    status: str
    reason: str


class AuditLogSummary(BaseModel):
    total_transactions: int
    approved_count: int
    rejected_count: int
    transactions: list[TransactionAuditEntry]


llm_config = LLMConfig(
    {"api_type": "openai", "model": "gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"]},
    response_format=AuditLogSummary,
)

TRANSACTIONS = """
Transaction: $500 to Staples. Memo: Quarterly supplies.
Transaction: $23,000 to CyberSins Ltd. Memo: Confidential.
Transaction: $1,500 to Initech. Memo: Routine payment.
"""


def main() -> None:
    summary_bot = ConversableAgent(
        name="summary_bot",
        system_message=(
            "You are a financial summary assistant that generates audit logs. Reject "
            "transactions over $10,000 or with a vague memo, and approve the rest."
        ),
        llm_config=llm_config,
    )

    response = summary_bot.run(
        message=f"Produce the audit log for these transactions:\n{TRANSACTIONS}",
        max_turns=1,
        user_input=False,
    )
    response.process()

    audit_log = AuditLogSummary.model_validate_json(response.messages[-1]["content"])
    print("\nparsed audit log:")
    print(json.dumps(audit_log.model_dump(), indent=2))
    print("\nView the traces at http://localhost:6006")


if __name__ == "__main__":
    main()

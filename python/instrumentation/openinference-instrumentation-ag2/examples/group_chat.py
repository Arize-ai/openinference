"""
Trace an AG2 group chat where a manager routes between specialist agents.

This follows the group chat orchestration in the AG2 guide
(https://docs.ag2.ai/latest/docs/user-guide/basic-concepts/introducing-group-chat/):
an `AutoPattern` lets the group manager pick the next speaker, so the trace shows the
manager's routing decisions interleaved with each specialist's reply. The human
oversight agent from the guide is left out so the example runs unattended.

1. Run Phoenix locally: `pip install arize-phoenix && phoenix serve`
2. Install dependencies: `pip install -r requirements.txt`
3. Set your API key: `export OPENAI_API_KEY=<your-key>`
4. Run this example: `python group_chat.py`
5. View the traces at http://localhost:6006 under the `ag2-group-chat` project.
"""

import os
from typing import Any

from autogen import ConversableAgent, LLMConfig
from autogen.agentchat import initiate_group_chat
from autogen.agentchat.group.patterns import AutoPattern
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.ag2 import AG2Instrumentor
from openinference.instrumentation.openai import OpenAIInstrumentor
from openinference.semconv.resource import ResourceAttributes

endpoint = "http://localhost:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider(
    resource=Resource({ResourceAttributes.PROJECT_NAME: "ag2-group-chat"})
)
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
AG2Instrumentor().instrument(tracer_provider=tracer_provider)
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

llm_config = LLMConfig(
    {"api_type": "openai", "model": "gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"]}
)

TRANSACTIONS = [
    "Transaction: $500 to Staples. Memo: Quarterly supplies.",
    "Transaction: $23,000 to CyberSins Ltd. Memo: Confidential.",
    "Transaction: $1,500 to Initech. Memo: Routine payment.",
]

FINANCE_SYSTEM_MESSAGE = """
You are a financial compliance assistant reviewing transactions.
Flag a transaction as suspicious when the amount is over $10,000 or the memo is vague.
Approve the rest. Review every transaction in one reply, then hand off to summary_bot.
"""

SUMMARY_SYSTEM_MESSAGE = """
You are a financial summary assistant. Summarize the reviewed transactions as a markdown
table with Vendor, Memo, Amount, and Status columns, followed by the approved and
rejected counts. End your reply with "==== SUMMARY GENERATED ====".
"""


def is_termination_msg(message: dict[str, Any]) -> bool:
    return "==== SUMMARY GENERATED ====" in (message.get("content") or "")


def main() -> None:
    finance_bot = ConversableAgent(
        name="finance_bot",
        system_message=FINANCE_SYSTEM_MESSAGE,
        llm_config=llm_config,
    )
    summary_bot = ConversableAgent(
        name="summary_bot",
        system_message=SUMMARY_SYSTEM_MESSAGE,
        llm_config=llm_config,
    )

    pattern = AutoPattern(
        initial_agent=finance_bot,
        agents=[finance_bot, summary_bot],
        group_manager_args={
            "llm_config": llm_config,
            "is_termination_msg": is_termination_msg,
        },
    )

    result, _, _ = initiate_group_chat(
        pattern=pattern,
        messages="Please review these transactions:\n" + "\n".join(TRANSACTIONS),
        max_rounds=6,
    )
    print("\nfinal message:", result.chat_history[-1]["content"])
    print("\nView the traces at http://localhost:6006")


if __name__ == "__main__":
    main()

"""
Trace an LLM-driven AG2 tool call end to end.

The assistant decides to call `get_exchange_rate`, the user proxy executes it, and the
assistant summarizes the result. Instrumenting the OpenAI client as well nests the LLM
spans under the AG2 agent spans.

1. Run Phoenix locally: `pip install arize-phoenix && phoenix serve`
2. Install dependencies: `pip install -r requirements.txt`
3. Set your API key: `export OPENAI_API_KEY=<your-key>`
4. Run this example: `python openai_tool_calling.py`
5. View the traces at http://localhost:6006
"""

import os

from autogen import ConversableAgent
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.ag2 import AG2Instrumentor
from openinference.instrumentation.openai import OpenAIInstrumentor

endpoint = "http://localhost:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
AG2Instrumentor().instrument(tracer_provider=tracer_provider)
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

llm_config = {"config_list": [{"model": "gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"]}]}


def main() -> None:
    assistant = ConversableAgent(
        "assistant",
        llm_config=llm_config,
        human_input_mode="NEVER",
        system_message="Use the provided tools to answer currency questions, then reply TERMINATE.",
    )
    user_proxy = ConversableAgent(
        "user_proxy",
        llm_config=False,
        human_input_mode="NEVER",
        is_termination_msg=lambda message: "TERMINATE" in (message.get("content") or ""),
    )

    @user_proxy.register_for_execution()
    @assistant.register_for_llm(description="Convert an amount between two currencies.")
    def get_exchange_rate(amount: float, base: str, quote: str) -> str:
        rate = 0.92 if (base, quote) == ("USD", "EUR") else 1.0
        return f"{amount} {base} is {amount * rate:.2f} {quote}."

    chat = user_proxy.initiate_chat(
        assistant,
        message="How much is 250 USD in EUR?",
        max_turns=4,
        silent=True,
    )
    print("assistant:", chat.chat_history[-1]["content"])
    print("\nView the traces at http://localhost:6006")


if __name__ == "__main__":
    main()

"""
Trace an LLM-driven AG2 tool call end to end.

One agent is given the tool to call and a second executes it, which is the registration
split AG2 uses throughout its tools guide. The assistant decides to call
`get_exchange_rate`, the user proxy runs it, and the assistant summarizes the result.
Instrumenting the OpenAI client as well nests the LLM spans under the agent spans.

1. Run Phoenix locally: `pip install arize-phoenix && phoenix serve`
2. Install dependencies: `pip install -r requirements.txt`
3. Set your API key: `export OPENAI_API_KEY=<your-key>`
4. Run this example: `python tool_calling.py`
5. View the traces at http://localhost:6006 under the `ag2-tool-calling` project.
"""

import os
from typing import Annotated

from autogen import ConversableAgent, LLMConfig
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.ag2 import AG2Instrumentor
from openinference.instrumentation.openai import OpenAIInstrumentor
from openinference.semconv.resource import ResourceAttributes

endpoint = "http://localhost:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider(
    resource=Resource({ResourceAttributes.PROJECT_NAME: "ag2-tool-calling"})
)
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
AG2Instrumentor().instrument(tracer_provider=tracer_provider)
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

llm_config = LLMConfig(
    {"api_type": "openai", "model": "gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"]}
)

RATES = {("USD", "EUR"): 0.92, ("EUR", "USD"): 1.09, ("USD", "JPY"): 157.0}


def main() -> None:
    assistant = ConversableAgent(
        name="assistant",
        system_message=(
            "You convert currencies using the provided tool. Once you have the answer, "
            "state it and reply TERMINATE."
        ),
        llm_config=llm_config,
    )
    user_proxy = ConversableAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        is_termination_msg=lambda message: "TERMINATE" in (message.get("content") or ""),
    )

    @user_proxy.register_for_execution()
    @assistant.register_for_llm(description="Convert an amount between two currencies.")
    def get_exchange_rate(
        amount: Annotated[float, "The amount to convert"],
        base: Annotated[str, "The currency code to convert from, e.g. USD"],
        quote: Annotated[str, "The currency code to convert to, e.g. EUR"],
    ) -> str:
        rate = RATES.get((base.upper(), quote.upper()))
        if rate is None:
            return f"No exchange rate available for {base} to {quote}."
        return f"{amount} {base.upper()} is {amount * rate:.2f} {quote.upper()}."

    user_proxy.initiate_chat(
        assistant,
        message="How much is 250 USD in EUR?",
        max_turns=4,
    )

    print("\nView the traces at http://localhost:6006")


if __name__ == "__main__":
    main()

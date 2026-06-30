"""
Email Composer Agent with Observability

This script defines a LangGraph-based multi-step agent that generates,
formats, and tones email drafts based on bullet points and a desired tone.
It includes OpenLIT instrumentation and OpenLIT → OpenInference trace conversion
for real-time tracing and observability in Phoenix or Arize.
"""

import json
import os
import sys

import grpc
import openlit
from langchain.chat_models import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from phoenix.otel import register
from typing_extensions import Literal, TypedDict

from openinference.instrumentation.openlit import OpenInferenceSpanProcessor

# --------------------------------------------------------------------------------
# Setup OpenTelemetry Tracing
# --------------------------------------------------------------------------------


class DebugPrintProcessor(SpanProcessor):
    """Optional span processor for debugging."""

    def on_end(self, span):
        print(f"\n=== RAW OpenLLMetry span: {span.name} ===", file=sys.stderr)
        print(json.dumps(dict(span.attributes), default=str, indent=2), file=sys.stderr)

    def on_start(self, span, parent_context=None):
        pass

    def shutdown(self):
        return True

    def force_flush(self, timeout_millis=None):
        return True


provider = register(
    project_name="email-agent-observability",
    set_global_tracer_provider=True,
)

provider.add_span_processor(OpenInferenceSpanProcessor())

provider.add_span_processor(
    BatchSpanProcessor(
        OTLPSpanExporter(
            endpoint="http://localhost:4317",
            headers={},
            compression=grpc.Compression.Gzip,
        )
    )
)

tracer = provider.get_tracer(__name__)
openlit.init(otel_tracer=tracer)

# --------------------------------------------------------------------------------
# LLM Setup
# --------------------------------------------------------------------------------

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# --------------------------------------------------------------------------------
# State Definition
# --------------------------------------------------------------------------------


class EmailState(TypedDict, total=False):
    subject: str
    bullet_points: str
    desired_tone: str  # e.g., "formal", "friendly"
    outline: str
    draft_email: str
    final_email: str


# --------------------------------------------------------------------------------
# Node Functions
# --------------------------------------------------------------------------------


def generate_outline(state: EmailState) -> EmailState:
    """Node 1: Generate outline from bullet points."""
    prompt = (
        "Create a concise outline for an email.\n"
        f"Subject: {state['subject']}\n"
        f"Bullet points:\n{state['bullet_points']}\n"
        "Return the outline as numbered points."
    )
    outline = llm.invoke(prompt).content
    return {"outline": outline}


def write_email(state: EmailState) -> EmailState:
    """Node 2: Write email based on the outline and tone."""
    prompt = (
        f"Write a complete email using this outline:\n{state['outline']}\n\n"
        f"Tone: {state['desired_tone']}\n"
        "Start with a greeting, use professional formatting, and keep it concise."
    )
    email = llm.invoke(prompt).content
    return {"draft_email": email}


def tone_gate(state: EmailState) -> Literal["Pass", "Fail"]:
    """Heuristic: Does email match the desired tone?"""
    prompt = (
        f"Check whether the following email matches the desired tone {state['desired_tone']}:\n\n"
        f"{state['draft_email']}\n\n"
        "If it does, return 'Pass'. Otherwise, return 'Fail'."
    )
    return llm.invoke(prompt).content.strip()


def reform_tone(state: EmailState) -> EmailState:
    """Node 3: Rewrite the email to match the desired tone."""
    prompt = (
        f"Reform the following email to match a {state['desired_tone']} tone:\n\n"
        f"{state['draft_email']}\n\n"
        "Keep the content unchanged, but adjust phrasing, tone, and sign-off."
    )
    final_email = llm.invoke(prompt).content
    return {"final_email": final_email}


# --------------------------------------------------------------------------------
# Build LangGraph Workflow
# --------------------------------------------------------------------------------

graph = StateGraph(EmailState)

graph.add_node("outline_generator", generate_outline)
graph.add_node("email_writer", write_email)
graph.add_node("tone_reformer", reform_tone)

graph.add_edge(START, "outline_generator")
graph.add_edge("outline_generator", "email_writer")
graph.add_conditional_edges("email_writer", tone_gate, {"Pass": END, "Fail": "tone_reformer"})
graph.add_edge("tone_reformer", END)

email_chain = graph.compile()

# --------------------------------------------------------------------------------
# Run Example
# --------------------------------------------------------------------------------

if __name__ == "__main__":
    input_state = {
        "subject": "Quarterly Sales Recap & Next Steps",
        "bullet_points": (
            "- Q1 revenue up 18%\n- Need feedback on new pricing tiers\n"
            "- Reminder: submit pipeline forecasts by Friday"
        ),
        "desired_tone": "friendly",
    }

    result = email_chain.invoke(input_state)
    output = result.get("final_email", result.get("draft_email"))

    print("\n========== FINAL EMAIL ==========")
    print(output)

"""Advanced example: Multi-agent RealtimeSession with handoffs, tools, guardrails, and history.

This example demonstrates a customer support assistant that exercises all the major
OpenInference span types produced by the realtime tracer:

  Agent: triage_agent        [AGENT - root, first to respond]
    Tool: get_account_info   [TOOL  - looks up the caller's account]
    Handoff: triage -> billing_agent  [TOOL - transfer to specialist]
  Agent: billing_agent       [AGENT - handles payment/subscription questions]
    Tool: get_order_status   [TOOL  - retrieves order details]
    Tool: create_support_ticket [TOOL - escalates when needed]
  Agent: tech_agent          [AGENT - handles technical issues]
    Tool: create_support_ticket [TOOL - escalates when needed]
  Guardrail: profanity_check [CHAIN - fires when the agent's output contains disallowed content]

Span hierarchy observed in Phoenix:

  RealtimeSession: triage_agent   (root AGENT span, wraps the full session)
    Agent: triage_agent
      Tool: get_account_info
      Handoff: triage_agent -> billing_agent
    Agent: billing_agent
      Tool: get_order_status
    Guardrail: profanity_check    (fires if the model produces disallowed content)

Prerequisites:
    pip install "openai-agents[realtime]"
    pip install openinference-instrumentation-openai-agents
    pip install opentelemetry-exporter-otlp-proto-http

    # Start Phoenix locally:
    pip install arize-phoenix
    python -m phoenix.server.main &

Usage:
    OPENAI_API_KEY=<your-key> python realtime_agent_advanced.py
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from agents import GuardrailFunctionOutput, OutputGuardrail, RunContextWrapper, function_tool
from agents.realtime import RealtimeAgent, RealtimeRunner
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor

# ---------------------------------------------------------------------------
# Tracing setup — send spans to a local Phoenix instance
# ---------------------------------------------------------------------------
endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

OpenAIAgentsInstrumentor().instrument(tracer_provider=tracer_provider)

# ---------------------------------------------------------------------------
# Simulated back-end data
# ---------------------------------------------------------------------------
_ACCOUNTS: dict[str, dict[str, Any]] = {
    "alice@example.com": {"name": "Alice", "plan": "Pro", "status": "active"},
    "bob@example.com": {"name": "Bob", "plan": "Basic", "status": "past_due"},
}

_ORDERS: dict[str, dict[str, Any]] = {
    "ORD-001": {"item": "Pro Plan - Monthly", "amount": "$29.00", "status": "paid"},
    "ORD-002": {"item": "Basic Plan - Monthly", "amount": "$9.00", "status": "overdue"},
}

_TICKET_COUNTER = [1000]  # mutable so the closure can increment it


# ---------------------------------------------------------------------------
# Tools — shared across agents
# ---------------------------------------------------------------------------


@function_tool
def get_account_info(email: str) -> str:
    """Look up a customer account by email address."""
    account = _ACCOUNTS.get(email.lower())
    if account is None:
        return json.dumps({"error": f"No account found for {email}"})
    return json.dumps(account)


@function_tool
def get_order_status(order_id: str) -> str:
    """Return the current status and details of an order."""
    order = _ORDERS.get(order_id.upper())
    if order is None:
        return json.dumps({"error": f"Order {order_id} not found"})
    return json.dumps(order)


@function_tool
def create_support_ticket(category: str, description: str) -> str:
    """Open a support ticket and return its ID.

    Use this when the customer's issue cannot be resolved immediately.
    category: one of 'billing', 'technical', 'general'
    description: a concise summary of the issue
    """
    _TICKET_COUNTER[0] += 1
    ticket_id = f"TKT-{_TICKET_COUNTER[0]}"
    return json.dumps(
        {
            "ticket_id": ticket_id,
            "category": category,
            "status": "open",
            "message": f"Ticket {ticket_id} created. A support engineer will contact you within 24 hours.",
        }
    )


# ---------------------------------------------------------------------------
# Guardrail — blocks clearly inappropriate input
# ---------------------------------------------------------------------------

_BLOCKED_WORDS = {"badword1", "badword2"}  # replace with real moderation logic


async def profanity_check(
    ctx: RunContextWrapper[None], agent: Any, output: Any
) -> GuardrailFunctionOutput:
    """Block output that contains disallowed words."""
    text = output if isinstance(output, str) else json.dumps(output)
    triggered = any(word in text.lower() for word in _BLOCKED_WORDS)
    return GuardrailFunctionOutput(
        output_info={"blocked": triggered},
        tripwire_triggered=triggered,
    )


# ---------------------------------------------------------------------------
# Specialist agents
# ---------------------------------------------------------------------------

billing_agent = RealtimeAgent(
    name="billing_agent",
    instructions=(
        "You are a billing specialist for Acme Corp. "
        "Help customers with invoices, payments, and subscription changes. "
        "Use get_order_status to look up order details. "
        "If an issue cannot be resolved, create a support ticket with category='billing'."
    ),
    tools=[get_order_status, create_support_ticket],
)

tech_agent = RealtimeAgent(
    name="tech_agent",
    instructions=(
        "You are a technical support engineer for Acme Corp. "
        "Help customers troubleshoot product issues. "
        "If an issue cannot be resolved, create a support ticket with category='technical'."
    ),
    tools=[create_support_ticket],
)

# ---------------------------------------------------------------------------
# Triage agent — entry point, routes to specialists via handoffs
# ---------------------------------------------------------------------------

triage_agent = RealtimeAgent(
    name="triage_agent",
    instructions=(
        "You are the first point of contact for Acme Corp customer support. "
        "Greet the customer, use get_account_info to look up their account if they provide an email, "
        "then route them to the right specialist:\n"
        "  - Billing questions → billing_agent\n"
        "  - Technical issues  → tech_agent\n"
        "  - General queries   → answer directly and briefly."
    ),
    tools=[get_account_info],
    handoffs=[billing_agent, tech_agent],
    output_guardrails=[OutputGuardrail(guardrail_function=profanity_check, name="profanity_check")],
)


# ---------------------------------------------------------------------------
# Event printer — logs every interesting realtime event to stdout
# ---------------------------------------------------------------------------

_in_transcript = False  # tracks whether we're mid-transcript line


def _print_event(event: Any) -> tuple[bool, bool]:
    """Print a summary of the event.

    Returns (done, fatal):
      done  — turn is fully complete (agent_end seen); safe to send next message
      fatal — unrecoverable error; caller should stop the session
    """
    global _in_transcript
    t = getattr(event, "type", None)

    # Flush any open transcript line before printing a structured event
    def _flush_transcript() -> None:
        global _in_transcript
        if _in_transcript:
            print()  # newline after the accumulated delta chars
            _in_transcript = False

    if t == "agent_start":
        _flush_transcript()
        print(f"[agent_start]      agent={event.agent.name}")
    elif t == "agent_end":
        _flush_transcript()
        print(f"[agent_end]        agent={event.agent.name}")
        return True, False  # turn is done; safe to send next message
    elif t == "handoff":
        _flush_transcript()
        print(f"[handoff]          {event.from_agent.name} → {event.to_agent.name}")
    elif t == "tool_start":
        _flush_transcript()
        print(f"[tool_start]       tool={event.tool.name}")
    elif t == "tool_end":
        _flush_transcript()
        print(f"[tool_end]         tool={event.tool.name}  output={event.output!r}")
    elif t == "history_added":
        _flush_transcript()
        item = getattr(event, "item", None)
        role = getattr(item, "role", "?")
        print(f"[history_added]    role={role}")
    elif t == "history_updated":
        _flush_transcript()
        history = getattr(event, "history", [])
        print(f"[history_updated]  {len(history)} item(s) in history")
    elif t == "guardrail_tripped":
        _flush_transcript()
        results = getattr(event, "guardrail_results", [])
        names = [getattr(getattr(r, "guardrail", None), "name", "?") for r in results]
        print(f"[guardrail_tripped] guardrails={names}")
    elif t == "audio":
        data = getattr(event, "audio", None)
        nbytes = len(getattr(data, "data", b"")) if data else 0
        _flush_transcript()
        print(f"[audio]            {nbytes} bytes received")
    elif t == "audio_end":
        _flush_transcript()
        print("[audio_end]        model finished speaking")
        # Do NOT stop here — wait for agent_end so the turn is fully settled
        # before the next send_message call.
    elif t == "error":
        _flush_transcript()
        print(f"[error]            {event.error}")
        return False, True  # fatal — stop the session
    elif t == "raw_model_event":
        data = getattr(event, "data", None)
        sub_type = getattr(data, "type", None)
        if sub_type == "transcript_delta":
            delta = getattr(data, "delta", "")
            if not _in_transcript:
                print("[transcript]       ", end="", flush=True)
                _in_transcript = True
            print(delta, end="", flush=True)
        # suppress noisy low-level events
    return False, False


# ---------------------------------------------------------------------------
# Conversation script — demonstrates each feature
# ---------------------------------------------------------------------------

TURNS = [
    (
        "Billing inquiry with account lookup",
        "Hi, I'm alice@example.com. I have a question about my invoice ORD-001.",
    ),
    (
        "Tech issue — triggers handoff to tech_agent",
        "Actually, the app keeps crashing when I upload files. Can you help?",
    ),
    (
        "Escalation — agent creates a support ticket",
        "I've already tried reinstalling. Nothing works. Please escalate this.",
    ),
]


async def run_turn(session: Any, description: str, message: str) -> bool:
    """Send one message and drain events until the spoken response is fully settled.

    The SDK fires agent_end after *each internal model turn*, which may be a
    tool-calling turn with no audio.  We must wait until agent_end fires AND
    we have already seen audio_end — that combination signals that the agent
    finished speaking and the server-side response is closed.

    Returns False if a fatal error was encountered.
    """
    print(f"\n{'=' * 60}")
    print(f"Turn: {description}")
    print(f"User: {message!r}")
    print("=" * 60)

    await session.send_message(message)

    audio_done = False
    async for event in session:
        t = getattr(event, "type", None)
        done, fatal = _print_event(event)
        if fatal:
            return False
        if t == "audio_end":
            audio_done = True
        # agent_end after tool-only turns (no audio) arrives before the spoken
        # response starts — keep draining until we've also seen audio_end.
        if done and audio_done:
            break

    print()  # blank line between turns
    return True


async def main() -> None:
    runner = RealtimeRunner(triage_agent)

    print("Starting advanced realtime session (press Ctrl+C to stop)...")
    print("Traces will appear in Phoenix at http://localhost:6006")

    async with await runner.run() as session:
        for description, message in TURNS:
            ok = await run_turn(session, description, message)
            if not ok:
                break

    print("Session ended. Check Phoenix for the full span hierarchy.")


if __name__ == "__main__":
    asyncio.run(main())

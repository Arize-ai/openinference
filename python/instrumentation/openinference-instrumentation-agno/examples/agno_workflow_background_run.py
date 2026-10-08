"""
This example shows how to instrument your agno agent with OpenInference
and send traces to Arize Phoenix.

Install dependencies:
pip install openai opentelemetry-sdk opentelemetry-exporter-otlp
pip install openinference-instrumentation-agno
"""

import asyncio

from agno.agent.agent import Agent
from agno.models.openai import OpenAIChat
from agno.workflow.step import Step
from agno.workflow.workflow import Workflow
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.agno import AgnoInstrumentor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

AgnoInstrumentor().instrument(tracer_provider=tracer_provider)


summariser = Agent(
    name="Summariser",
    model=OpenAIChat(id="gpt-4o-mini"),
    role="Summarise the user's input in one sentence.",
)
step = Step(name="summarise", agent=summariser)
workflow = Workflow(
    name="Background Workflow",
    description="Demonstrates correct span lifetime for background=True runs.",
    steps=[step],
)


def _print_section(title: str) -> None:
    width = 60
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


async def _wait_for_background_tasks(before: set, timeout: float = 30.0) -> None:
    """
    Await every asyncio task that was created after `before` was captured.
    """
    new_tasks = asyncio.all_tasks() - before - {asyncio.current_task()}
    if not new_tasks:
        print("[wait] No new background tasks detected. Workflow may have already finished.")
        return

    print(f"[wait] Waiting for {len(new_tasks)} background task(s) to finish …")
    done, pending = await asyncio.wait(new_tasks, timeout=timeout)

    for task in pending:
        print(f"[warn] Task {task.get_name()!r} did not finish within {timeout}s.")

    for task in done:
        exc = task.exception()
        if exc:
            print(f"[warn] Background task raised: {exc}")


async def demo_foreground_run() -> None:
    _print_section("Demo Foreground Run")

    response = await workflow.arun(
        input="Explain why the sky is blue in plain English.",
        session_id="demo-fg-session",
        user_id="demo-user",
    )

    print(f"status: {response.status}")
    print(f"run_id: {response.run_id}")
    print(f"content: {response.content!r}")


async def demo_background_run() -> None:
    _print_section("Demo Background Run")

    # Snapshot running tasks before calling arun so we can wait for the
    # background task agno will create inside `_arun_background`.
    before = asyncio.all_tasks()

    placeholder = await workflow.arun(
        input="Explain why the sky is blue in plain English.",
        session_id="demo-bg-session",
        user_id="demo-user",
        background=True,
    )

    # arun returns immediately with a pending placeholder.
    print(f"Placeholder status: {placeholder.status} | expected 'pending'")
    print(f"Placeholder run_id: {placeholder.run_id}")
    print(f"Placeholder content: {placeholder.content!r} | expected None")
    print()
    print("Waiting for background execution to complete …")

    await _wait_for_background_tasks(before)

    print()
    print(f"Final status: {placeholder.status} | expected 'completed'")
    print(f"Final content: {placeholder.content!r}")


async def main() -> None:
    print("\nBackground Workflow Instrumentation Demo")
    print("View Traces: http://127.0.0.1:6006\n")

    await demo_foreground_run()
    await asyncio.sleep(2)
    await demo_background_run()


if __name__ == "__main__":
    asyncio.run(main())

"""
Map the Agents SDK's conversation grouping (`group_id`) to OpenInference `session.id`.

Two turns of one conversation run as two separate traces that share
`RunConfig(group_id=...)`, then a third run sits inside an enclosing
`trace(..., group_id=...)`. Each trace's root AGENT span carries `session.id`, so Phoenix
groups the turns under one session.

The agent uses a canned-reply model, so no API key is needed. Like the SDK's built-in models,
it records each call as a generation span, which becomes the LLM span.

1. Run a local OTLP collector such as Phoenix: `uvx arize-phoenix serve`
2. Install dependencies: `pip install -r requirements.txt`
3. Run this example: `python session_group_id.py`
4. View the traces at http://localhost:6006 under the `openai-agents-session-group-id`
   project, and the grouped turns under its Sessions tab.
"""

import sys
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

from agents import Agent, RunConfig, Runner, Usage, trace
from agents.items import ModelResponse, TResponseStreamEvent
from agents.models.interface import Model
from agents.tracing import generation_span
from openai.types.responses import ResponseOutputMessage, ResponseOutputText
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor
from openinference.semconv.resource import ResourceAttributes

project_name = sys.argv[1] if len(sys.argv) > 1 else "openai-agents-session-group-id"
endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = TracerProvider(resource=Resource({ResourceAttributes.PROJECT_NAME: project_name}))
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
OpenAIAgentsInstrumentor().instrument(tracer_provider=tracer_provider)


class CannedModel(Model):
    """Replies with a fixed sentence and records the call as a generation span."""

    def __init__(self, reply: str) -> None:
        self._reply = reply

    async def get_response(
        self, system_instructions: str | None, input: Any, *args: Any, **kwargs: Any
    ) -> ModelResponse:
        message = ResponseOutputMessage(
            id="msg_canned",
            type="message",
            role="assistant",
            status="completed",
            content=[ResponseOutputText(type="output_text", text=self._reply, annotations=[])],
        )
        with generation_span(
            input=[{"role": "user", "content": str(input)}],
            output=[{"role": "assistant", "content": self._reply}],
            model="canned-model",
        ):
            return ModelResponse(output=[message], usage=Usage(), response_id=None)

    def stream_response(self, *args: Any, **kwargs: Any) -> AsyncIterator[TResponseStreamEvent]:
        raise NotImplementedError("this example only runs without streaming")


agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant. Reply in one short sentence.",
    model=CannedModel("Happy to help!"),
)


def main() -> None:
    # Phoenix keys sessions by id across projects, so use a fresh id per conversation.
    run_id = uuid4().hex[:8]
    # Two turns of one conversation: each run starts its own trace, and both share a group_id.
    conversation = RunConfig(group_id=f"conversation-{run_id}", workflow_name="Chat turn")
    first = Runner.run_sync(agent, "My favorite color is teal.", run_config=conversation)
    print(f"turn 1: {first.final_output}")
    second = Runner.run_sync(
        agent,
        first.to_input_list() + [{"role": "user", "content": "What is my favorite color?"}],
        run_config=conversation,
    )
    print(f"turn 2: {second.final_output}")

    # Inside an enclosing trace, that trace's group_id applies; a RunConfig.group_id would not.
    with trace("Support workflow", group_id=f"ticket-{run_id}"):
        third = Runner.run_sync(agent, "Say hello to a new customer.")
    print(f"workflow: {third.final_output}")


if __name__ == "__main__":
    main()

"""
Group AG2 traces into a Phoenix session and attach user, metadata, and tag attributes.

Phoenix uses `session.id` to group related traces into a conversation view, and
`user.id`, metadata, and tags to filter them. No LLM API key is required.

1. Run Phoenix locally: `pip install arize-phoenix && phoenix serve`
2. Install dependencies: `pip install -r requirements.txt`
3. Run this example: `python sessions_and_metadata.py`
4. Open http://localhost:6006, select the `ag2-sessions-and-metadata` project, and look
   under Sessions for `weekend-planning`.
"""

from autogen import ConversableAgent
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation import using_attributes
from openinference.instrumentation.ag2 import AG2Instrumentor
from openinference.semconv.resource import ResourceAttributes

endpoint = "http://localhost:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider(
    resource=Resource({ResourceAttributes.PROJECT_NAME: "ag2-sessions-and-metadata"})
)
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
AG2Instrumentor().instrument(tracer_provider=tracer_provider)


def main() -> None:
    assistant = ConversableAgent(
        "assistant",
        llm_config=False,
        human_input_mode="NEVER",
        default_auto_reply="Pack a rain jacket.",
    )
    user_proxy = ConversableAgent("user_proxy", llm_config=False, human_input_mode="NEVER")

    # Every span emitted inside this block carries the session, user, metadata, and tags.
    with using_attributes(
        session_id="weekend-planning",
        user_id="user-42",
        metadata={"itinerary": "portland", "tier": "pro"},
        tags=["example", "ag2"],
    ):
        for question in ("What should I pack?", "Anything else?"):
            chat = user_proxy.initiate_chat(assistant, message=question, max_turns=1, silent=True)
            print(f"{question} -> {chat.chat_history[-1]['content']}")

    print("\nView the session at http://localhost:6006")


if __name__ == "__main__":
    main()

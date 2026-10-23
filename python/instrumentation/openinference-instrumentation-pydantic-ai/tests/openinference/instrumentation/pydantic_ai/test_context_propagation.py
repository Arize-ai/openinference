import json
from typing import Mapping, Sequence, cast

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.util.types import AttributeValue
from pydantic_ai import Agent
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.test import TestModel

from openinference.instrumentation import using_metadata, using_session, using_tags, using_user
from openinference.semconv.trace import SpanAttributes


def test_context_attributes_propagate_to_all_spans(
    in_memory_span_exporter: InMemorySpanExporter, tracer_provider: TracerProvider
) -> None:
    """session.id, user.id, metadata and tags set via OpenInference context managers must
    land on every pydantic-ai span, the same as for every other OpenInference instrumentor.

    pydantic-ai creates spans with its own tracer, not an OITracer, so
    OpenInferenceSpanProcessor must copy these over itself.
    """
    trace.set_tracer_provider(tracer_provider)
    agent = Agent(TestModel(), output_type=str)
    agent.instrument = InstrumentationSettings(version=5)

    with (
        using_session("my-app-session-42"),
        using_user("user-123"),
        using_metadata({"env": "prod"}),
        using_tags(["billing"]),
    ):
        agent.run_sync("Say hello.")

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) >= 2, "expected at least an AGENT span and an LLM span"

    for span in spans:
        attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))
        assert attributes.get(SpanAttributes.SESSION_ID) == "my-app-session-42", span.name
        assert attributes.get(SpanAttributes.USER_ID) == "user-123", span.name
        metadata = attributes.get(SpanAttributes.METADATA)
        assert isinstance(metadata, str) and json.loads(metadata) == {"env": "prod"}, span.name
        assert tuple(cast(Sequence[str], attributes.get(SpanAttributes.TAG_TAGS))) == (
            "billing",
        ), span.name


def test_context_session_id_overrides_pydantic_ai_conversation_id(
    in_memory_span_exporter: InMemorySpanExporter, tracer_provider: TracerProvider
) -> None:
    """A caller-supplied session.id must win over pydantic-ai's own gen_ai.conversation.id,
    which get_attributes() would otherwise map to session.id and silently overwrite it with.
    """
    trace.set_tracer_provider(tracer_provider)
    agent = Agent(TestModel(), output_type=str)
    agent.instrument = InstrumentationSettings(version=5)

    with using_session("my-app-session-42"):
        agent.run_sync("Say hello.")

    spans = in_memory_span_exporter.get_finished_spans()
    assert spans
    for span in spans:
        attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))
        assert attributes.get(SpanAttributes.SESSION_ID) == "my-app-session-42", span.name


def test_session_id_falls_back_to_pydantic_ai_conversation_id_without_context(
    in_memory_span_exporter: InMemorySpanExporter, tracer_provider: TracerProvider
) -> None:
    """Without an OpenInference context session, session.id keeps falling back to
    pydantic-ai's own per-run conversation id (pre-existing behavior)."""
    trace.set_tracer_provider(tracer_provider)
    agent = Agent(TestModel(), output_type=str)
    agent.instrument = InstrumentationSettings(version=5)

    agent.run_sync("Say hello.")

    spans = in_memory_span_exporter.get_finished_spans()
    assert spans
    for span in spans:
        attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))
        session_id = attributes.get(SpanAttributes.SESSION_ID)
        assert isinstance(session_id, str) and session_id, span.name
        assert attributes.get(SpanAttributes.USER_ID) is None, span.name

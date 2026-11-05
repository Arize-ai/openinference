import datetime
import json
import types
import uuid
from decimal import Decimal

import pytest
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation.crewai._wrappers import _CrewKickoffWrapper


@pytest.mark.no_autoinstrument
def test_crew_kickoff_does_not_crash_on_non_serializable_inputs(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """`Crew.kickoff(inputs=...)` is frequently called with datetimes, UUIDs,
    Decimals, etc. The kickoff wrapper records those inputs as a span attribute
    *before* invoking the wrapped call, so a bare ``json.dumps`` there raises
    ``TypeError`` straight out of the wrapper and aborts the user's kickoff.
    The wrapper must serialize defensively (it already does for the same inputs
    via ``get_input_attributes``) so instrumentation never breaks the call.
    """
    wrapper = _CrewKickoffWrapper(tracer_provider.get_tracer(__name__))

    crew = types.SimpleNamespace(
        name=None,
        key="test-crew-key",
        id="11111111-1111-1111-1111-111111111111",
        agents=[],
        tasks=[],
    )
    inputs = {
        "as_of": datetime.datetime(2024, 1, 1, 12, 0, 0),
        "request_id": uuid.uuid4(),
        "amount": Decimal("19.99"),
    }

    def wrapped(*args: object, **kwargs: object) -> str:
        return "crew-output"

    # Pre-fix this raised TypeError before `wrapped` ever ran.
    result = wrapper(wrapped, crew, (), {"inputs": inputs})
    assert result == "crew-output"

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    crew_inputs = dict(spans[0].attributes or {})["crew_inputs"]
    # The attribute is valid JSON and preserves the non-serializable values.
    decoded = json.loads(crew_inputs)
    assert set(decoded) == {"as_of", "request_id", "amount"}

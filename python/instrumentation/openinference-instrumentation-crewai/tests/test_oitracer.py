from openinference.instrumentation import OITracer
from openinference.instrumentation.crewai import CrewAIInstrumentor


def test_oitracer() -> None:
    assert isinstance(CrewAIInstrumentor()._tracer, OITracer)

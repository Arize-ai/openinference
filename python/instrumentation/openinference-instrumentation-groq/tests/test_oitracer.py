from openinference.instrumentation import OITracer
from openinference.instrumentation.groq import GroqInstrumentor


def test_oitracer() -> None:
    assert isinstance(GroqInstrumentor()._tracer, OITracer)

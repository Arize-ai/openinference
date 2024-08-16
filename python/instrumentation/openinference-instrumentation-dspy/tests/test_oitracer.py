from openinference.instrumentation import OITracer
from openinference.instrumentation.dspy import DSPyInstrumentor


def test_oitracer() -> None:
    assert isinstance(DSPyInstrumentor()._tracer, OITracer)

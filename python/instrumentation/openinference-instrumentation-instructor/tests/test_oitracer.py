from openinference.instrumentation import OITracer
from openinference.instrumentation.instructor import InstructorInstrumentor


def test_oitracer() -> None:
    assert isinstance(InstructorInstrumentor()._tracer, OITracer)

from openinference.instrumentation import OITracer
from openinference.instrumentation.vertexai import VertexAIInstrumentor


def test_oitracer() -> None:
    assert isinstance(VertexAIInstrumentor()._tracer, OITracer)

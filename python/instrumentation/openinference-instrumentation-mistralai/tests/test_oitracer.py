from openinference.instrumentation import OITracer
from openinference.instrumentation.mistralai import MistralAIInstrumentor


def test_oitracer() -> None:
    assert isinstance(MistralAIInstrumentor()._tracer, OITracer)

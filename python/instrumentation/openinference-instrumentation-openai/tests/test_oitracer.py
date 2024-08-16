from openinference.instrumentation import OITracer
from openinference.instrumentation.openai import OpenAIInstrumentor


def test_oitracer() -> None:
    assert isinstance(OpenAIInstrumentor()._tracer, OITracer)

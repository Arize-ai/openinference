from openinference.instrumentation import OITracer
from openinference.instrumentation.litellm import LiteLLMInstrumentor


def test_oitracer() -> None:
    assert isinstance(LiteLLMInstrumentor()._tracer, OITracer)

from openinference.instrumentation import OITracer
from openinference.instrumentation.guardrails import GuardrailsInstrumentor


def test_oitracer() -> None:
    assert isinstance(GuardrailsInstrumentor()._tracer, OITracer)

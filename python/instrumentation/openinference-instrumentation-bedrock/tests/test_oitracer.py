from openinference.instrumentation import OITracer
from openinference.instrumentation.bedrock import BedrockInstrumentor


def test_oitracer() -> None:
    assert isinstance(BedrockInstrumentor()._tracer, OITracer)

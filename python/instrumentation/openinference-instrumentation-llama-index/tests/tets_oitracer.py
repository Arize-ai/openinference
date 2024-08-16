from openinference.instrumentation import OITracer
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor


def test_oitracer() -> None:
    assert isinstance(LlamaIndexInstrumentor()._tracer, OITracer)

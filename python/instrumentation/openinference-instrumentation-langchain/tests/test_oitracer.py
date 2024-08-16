from openinference.instrumentation import OITracer
from openinference.instrumentation.langchain import LangChainInstrumentor


def test_oitracer() -> None:
    assert isinstance(LangChainInstrumentor()._tracer._tracer, OITracer)

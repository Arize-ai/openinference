from opentelemetry.util._importlib_metadata import entry_points

from openinference.instrumentation import OITracer
from openinference.instrumentation.autogen_agentchat import AutogenAgentChatInstrumentor


class TestInstrumentor:
    def test_entrypoint_for_opentelemetry_instrument(self) -> None:
        (instrumentor_entrypoint,) = entry_points(
            group="opentelemetry_instrumentor", name="autogen_agentchat"
        )
        instrumentor = instrumentor_entrypoint.load()()
        assert isinstance(instrumentor, AutogenAgentChatInstrumentor)

    def test_oitracer(self) -> None:
        assert isinstance(AutogenAgentChatInstrumentor()._tracer, OITracer)

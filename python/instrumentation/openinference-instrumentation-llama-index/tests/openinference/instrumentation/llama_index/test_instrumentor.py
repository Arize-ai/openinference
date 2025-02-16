from opentelemetry.util._importlib_metadata import entry_points

from openinference.instrumentation.llama_index import LlamaIndexInstrumentor


class TestInstrumentor:
    def test_entrypoint_for_opentelemetry_instrument(self) -> None:
        (instrumentor_entrypoint,) = entry_points(
            group="opentelemetry_instrumentor", name="llama_index"
        )
        instrumentor = instrumentor_entrypoint.load()()
        assert isinstance(instrumentor, LlamaIndexInstrumentor)

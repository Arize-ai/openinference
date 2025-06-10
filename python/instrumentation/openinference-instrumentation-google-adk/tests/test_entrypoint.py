from opentelemetry.util._importlib_metadata import entry_points

from openinference.instrumentation.google_adk import GoogleADKInstrumentor


def test_entrypoint_for_opentelemetry_instrument() -> None:
    (instrumentor_entrypoint,) = entry_points(group="opentelemetry_instrumentor", name="google_adk")
    instrumentor = instrumentor_entrypoint.load()()
    assert isinstance(instrumentor, GoogleADKInstrumentor)

"""Tests for ElevenLabsInstrumentor."""

from typing import Any

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.util._importlib_metadata import entry_points

from openinference.instrumentation import OITracer, TraceConfig
from openinference.instrumentation.elevenlabs import ElevenLabsInstrumentor


class TestInstrumentor:
    """Test ElevenLabsInstrumentor functionality."""

    def test_entrypoint_for_opentelemetry_instrument(self) -> None:
        """Test that the instrumentor is registered as an entry point."""
        (instrumentor_entrypoint,) = entry_points(  # type: ignore[no-untyped-call]
            group="opentelemetry_instrumentor", name="elevenlabs"
        )
        instrumentor = instrumentor_entrypoint.load()()
        assert isinstance(instrumentor, ElevenLabsInstrumentor)

    def test_oitracer(self, setup_elevenlabs_instrumentation: Any) -> None:
        """Test that instrumentor uses OITracer from common package."""
        assert isinstance(ElevenLabsInstrumentor()._tracer, OITracer)

    def test_instrument_and_uninstrument(self, tracer_provider: TracerProvider) -> None:
        """Test that instrument() and uninstrument() work without errors."""
        instrumentor = ElevenLabsInstrumentor()

        # Instrument with skip_dep_check since elevenlabs may not be installed
        instrumentor.instrument(tracer_provider=tracer_provider, skip_dep_check=True)

        # Verify tracer is set
        assert hasattr(instrumentor, "_tracer")
        assert isinstance(instrumentor._tracer, OITracer)

        # Uninstrument
        instrumentor.uninstrument()

    def test_instrument_with_custom_config(self, tracer_provider: TracerProvider) -> None:
        """Test instrumentation with custom TraceConfig."""
        custom_config = TraceConfig()
        instrumentor = ElevenLabsInstrumentor()

        instrumentor.instrument(
            tracer_provider=tracer_provider,
            config=custom_config,
            skip_dep_check=True,
        )

        try:
            # Verify the config is stored
            assert instrumentor._config == custom_config
        finally:
            instrumentor.uninstrument()

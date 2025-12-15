"""Tests for PipecatInstrumentor."""

from typing import Any
from unittest.mock import Mock, patch

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.util._importlib_metadata import entry_points
from pipecat.pipeline.task import PipelineTask
from wrapt import BoundFunctionWrapper

from openinference.instrumentation import OITracer, TraceConfig
from openinference.instrumentation.pipecat import PipecatInstrumentor
from openinference.instrumentation.pipecat._observer import OpenInferenceObserver


class TestInstrumentor:
    """Test PipecatInstrumentor functionality."""

    def test_entrypoint_for_opentelemetry_instrument(self) -> None:
        """Test that the instrumentor is registered as an entry point."""
        (instrumentor_entrypoint,) = entry_points(  # type: ignore[no-untyped-call]
            group="opentelemetry_instrumentor", name="pipecat"
        )
        instrumentor = instrumentor_entrypoint.load()()
        assert isinstance(instrumentor, PipecatInstrumentor)

    def test_oitracer(self, setup_pipecat_instrumentation: Any) -> None:
        """Test that instrumentor uses OITracer from common package."""
        assert isinstance(PipecatInstrumentor()._tracer, OITracer)

    def test_instrument_wraps_pipeline_task(self, tracer_provider: TracerProvider) -> None:
        """Test that instrument() wraps PipelineTask.__init__."""
        # Ensure not already instrumented
        PipecatInstrumentor().uninstrument()

        # Store original __init__
        _ = PipelineTask.__init__

        # Instrument
        PipecatInstrumentor().instrument(tracer_provider=tracer_provider)

        # Verify __init__ was wrapped
        assert isinstance(PipelineTask.__init__, BoundFunctionWrapper)

        # Clean up
        PipecatInstrumentor().uninstrument()

    def test_uninstrument_restores_original(self, tracer_provider: TracerProvider) -> None:
        """Test that uninstrument() restores original PipelineTask.__init__."""
        # Ensure clean state
        PipecatInstrumentor().uninstrument()

        # Store original __init__
        _ = PipelineTask.__init__

        # Instrument
        instrumentor = PipecatInstrumentor()
        instrumentor.instrument(tracer_provider=tracer_provider)

        # Verify wrapped
        assert isinstance(PipelineTask.__init__, BoundFunctionWrapper)

        # Uninstrument
        instrumentor.uninstrument()

        # Verify restored
        assert not isinstance(PipelineTask.__init__, BoundFunctionWrapper)

    def test_observer_injection_into_pipeline_task(
        self, setup_pipecat_instrumentation: Any
    ) -> None:
        """Test that observer is injected into PipelineTask instances."""
        # Create a mock pipeline to pass to PipelineTask
        mock_pipeline = Mock()

        # Create a PipelineTask - observer should be automatically added
        with patch.object(PipelineTask, "add_observer") as mock_add_observer:
            _ = PipelineTask(mock_pipeline)

            # Verify add_observer was called with an OpenInferenceObserver
            assert mock_add_observer.called
            observer = mock_add_observer.call_args[0][0]
            assert isinstance(observer, OpenInferenceObserver)

    def test_create_observer_before_instrument_fails(self) -> None:
        """Test that creating observer before instrumentation raises error."""
        instrumentor = PipecatInstrumentor()

        # Ensure not instrumented
        if hasattr(instrumentor, "_tracer"):
            delattr(instrumentor, "_tracer")

        with pytest.raises(RuntimeError, match="must be instrumented"):
            instrumentor.create_observer()

    def test_create_observer_after_instrument(self, tracer_provider: TracerProvider) -> None:
        """Test manual observer creation after instrumentation."""
        instrumentor = PipecatInstrumentor()
        instrumentor.instrument(tracer_provider=tracer_provider)

        try:
            observer = instrumentor.create_observer()
            assert isinstance(observer, OpenInferenceObserver)
        finally:
            instrumentor.uninstrument()

    def test_instrument_with_custom_config(self, tracer_provider: TracerProvider) -> None:
        """Test instrumentation with custom TraceConfig."""
        custom_config = TraceConfig()
        instrumentor = PipecatInstrumentor()

        instrumentor.instrument(tracer_provider=tracer_provider, config=custom_config)

        try:
            # Verify the config is stored
            assert instrumentor._config == custom_config
        finally:
            instrumentor.uninstrument()

    def test_instrument_with_debug_log_filename(self, tracer_provider: TracerProvider) -> None:
        """Test instrumentation with debug log filename."""
        debug_log = "test_debug.log"
        instrumentor = PipecatInstrumentor()

        instrumentor.instrument(tracer_provider=tracer_provider, debug_log_filename=debug_log)

        try:
            # Verify the debug log filename is stored
            assert instrumentor._debug_log_filename == debug_log
        finally:
            instrumentor.uninstrument()

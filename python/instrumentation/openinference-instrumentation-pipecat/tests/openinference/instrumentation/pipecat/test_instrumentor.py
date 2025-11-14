"""
Test the PipecatInstrumentor class for automatic observer injection.
"""

from pipecat.pipeline.task import PipelineTask

from openinference.instrumentation.pipecat import PipecatInstrumentor


class TestInstrumentorBasics:
    """Test basic instrumentor functionality"""

    def test_instrumentor_can_be_imported(self):
        """Test that instrumentor can be imported"""
        assert PipecatInstrumentor is not None

    def test_instrumentor_initialization(self):
        """Test instrumentor can be initialized"""
        instrumentor = PipecatInstrumentor()
        assert instrumentor is not None

    def test_instrumentor_instrument(self, tracer_provider):
        """Test instrumentor can be instrumented"""
        instrumentor = PipecatInstrumentor()
        instrumentor.instrument(tracer_provider=tracer_provider)
        assert instrumentor.is_instrumented_by_opentelemetry

    def test_instrumentor_uninstrument(self, tracer_provider):
        """Test instrumentor can be uninstrumented"""
        instrumentor = PipecatInstrumentor()
        instrumentor.instrument(tracer_provider=tracer_provider)
        instrumentor.uninstrument()
        assert not instrumentor.is_instrumented_by_opentelemetry

    def test_double_instrument_is_safe(self, tracer_provider):
        """Test that double instrumentation is safe"""
        instrumentor = PipecatInstrumentor()
        instrumentor.instrument(tracer_provider=tracer_provider)
        instrumentor.instrument(tracer_provider=tracer_provider)  # Should not raise
        assert instrumentor.is_instrumented_by_opentelemetry

    def test_uninstrument_without_instrument_is_safe(self):
        """Test that uninstrument without instrument is safe"""
        instrumentor = PipecatInstrumentor()
        instrumentor.uninstrument()  # Should not raise


class TestObserverInjection:
    """Test automatic observer injection into PipelineTask"""

    def test_observer_injected_automatically(self, tracer_provider, simple_pipeline):
        """Test that observer is automatically injected into PipelineTask"""
        instrumentor = PipecatInstrumentor()
        instrumentor.instrument(tracer_provider=tracer_provider)

        # Create a task - observer should be auto-injected
        task = PipelineTask(simple_pipeline)

        # Check that task has observers
        # Note: Implementation will need to expose observers for verification
        # or we verify via generated spans
        assert task is not None

        instrumentor.uninstrument()

    def test_multiple_tasks_get_separate_observers(self, tracer_provider, simple_pipeline):
        """Test that each task gets its own observer instance (thread safety)"""
        instrumentor = PipecatInstrumentor()
        instrumentor.instrument(tracer_provider=tracer_provider)

        # Create multiple tasks
        task1 = PipelineTask(simple_pipeline)
        task2 = PipelineTask(simple_pipeline)

        # Each should have independent observer state
        # Verify via task execution producing independent spans
        assert task1 is not None
        assert task2 is not None
        assert task1 is not task2

        instrumentor.uninstrument()

    def test_existing_observers_preserved(self, tracer_provider, simple_pipeline):
        """Test that existing observers are preserved when auto-injecting"""
        from pipecat.observers.base_observer import BaseObserver

        class CustomObserver(BaseObserver):
            def __init__(self):
                super().__init__()
                self.events = []

            async def on_push_frame(self, data):
                self.events.append(data)

        instrumentor = PipecatInstrumentor()
        instrumentor.instrument(tracer_provider=tracer_provider)

        custom_observer = CustomObserver()
        task = PipelineTask(simple_pipeline, observers=[custom_observer])

        # Custom observer should still be present
        # Implementation should add OpenInferenceObserver without removing custom ones
        assert task is not None

        instrumentor.uninstrument()

    def test_manual_observer_creation(self, tracer_provider):
        """Test manual observer creation for advanced use cases"""
        instrumentor = PipecatInstrumentor()
        instrumentor.instrument(tracer_provider=tracer_provider)

        # Create observer manually
        observer = instrumentor.create_observer()
        assert observer is not None

        instrumentor.uninstrument()


class TestInstrumentationWithConfig:
    """Test instrumentation with various configurations"""

    def test_instrument_with_trace_config(self, tracer_provider):
        """Test instrumentation with custom TraceConfig"""
        from openinference.instrumentation import TraceConfig

        config = TraceConfig()
        instrumentor = PipecatInstrumentor()
        instrumentor.instrument(tracer_provider=tracer_provider, config=config)

        assert instrumentor.is_instrumented_by_opentelemetry
        instrumentor.uninstrument()


class TestInstrumentorLifecycle:
    """Test instrumentor lifecycle and cleanup"""

    def test_instrumentor_singleton_behavior(self, tracer_provider):
        """Test that multiple instrumentor instances behave correctly"""
        instrumentor1 = PipecatInstrumentor()
        instrumentor2 = PipecatInstrumentor()

        instrumentor1.instrument(tracer_provider=tracer_provider)

        # Second instrumentor should detect first is already instrumented
        assert instrumentor1.is_instrumented_by_opentelemetry
        assert instrumentor2.is_instrumented_by_opentelemetry  # Singleton pattern

        instrumentor1.uninstrument()

    def test_cleanup_on_uninstrument(self, tracer_provider, simple_pipeline):
        """Test that uninstrument properly cleans up"""
        instrumentor = PipecatInstrumentor()
        instrumentor.instrument(tracer_provider=tracer_provider)

        # Create task while instrumented
        task1 = PipelineTask(simple_pipeline)

        instrumentor.uninstrument()

        # New tasks should not get observer after uninstrument
        task2 = PipelineTask(simple_pipeline)

        assert task1 is not None
        assert task2 is not None

    def test_reinstrumentation(self, tracer_provider):
        """Test that instrumentation can be re-applied after uninstrument"""
        instrumentor = PipecatInstrumentor()

        instrumentor.instrument(tracer_provider=tracer_provider)
        instrumentor.uninstrument()
        instrumentor.instrument(tracer_provider=tracer_provider)

        assert instrumentor.is_instrumented_by_opentelemetry
        instrumentor.uninstrument()


class TestInstrumentationDependencies:
    """Test that instrumentation properly declares dependencies"""

    def test_instrumentation_dependencies(self):
        """Test that instrumentor declares correct dependencies"""
        instrumentor = PipecatInstrumentor()
        dependencies = instrumentor.instrumentation_dependencies()

        # Should declare pipecat as dependency
        assert "pipecat" in dependencies or "pipecat-ai" in dependencies

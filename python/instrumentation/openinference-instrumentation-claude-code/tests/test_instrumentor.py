import pytest
from openinference.instrumentation.claude_code import ClaudeCodeInstrumentor


def test_instrumentor_can_be_instantiated():
    """Test that instrumentor can be created."""
    instrumentor = ClaudeCodeInstrumentor()
    assert instrumentor is not None


def test_instrumentor_instrument():
    """Test that instrumentation can be applied."""
    instrumentor = ClaudeCodeInstrumentor()
    instrumentor.instrument()
    # Verify it doesn't crash
    instrumentor.uninstrument()


def test_instrumentor_instrument_twice_is_idempotent():
    """Test that calling instrument() twice is safe."""
    instrumentor = ClaudeCodeInstrumentor()
    instrumentor.instrument()
    instrumentor.instrument()  # Should not crash
    instrumentor.uninstrument()

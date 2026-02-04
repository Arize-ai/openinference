"""Integration tests requiring actual SDK installation."""

import pytest

from openinference.instrumentation.claude_code import ClaudeCodeInstrumentor


@pytest.mark.integration
@pytest.mark.asyncio
async def test_end_to_end_query_instrumentation(tracer_provider, in_memory_span_exporter):
    """
    End-to-end test of query instrumentation.

    This test requires claude-agent-sdk to be installed and may require
    actual API credentials (or mocking at SDK level).

    Mark as @pytest.mark.integration and skip in normal test runs.
    """
    instrumentor = ClaudeCodeInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider)

    try:
        from claude_agent_sdk import AssistantMessage, TextBlock, query

        # This would call real SDK - mock or skip if no credentials
        pytest.skip("Integration test - requires SDK credentials")

        # If credentials available:
        # async for message in query(prompt="What is 2+2?"):
        #     if isinstance(message, AssistantMessage):
        #         pass

        # spans = in_memory_span_exporter.get_finished_spans()
        # assert len(spans) >= 1

    finally:
        instrumentor.uninstrument()


def test_package_metadata():
    """Test that package metadata is correct."""
    from openinference.instrumentation.claude_code import __version__

    assert __version__ == "0.1.0"

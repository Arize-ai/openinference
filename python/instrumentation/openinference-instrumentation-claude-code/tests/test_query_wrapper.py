import pytest
from claude_agent_sdk import AssistantMessage, TextBlock

from openinference.instrumentation.claude_code import ClaudeCodeInstrumentor


@pytest.mark.asyncio
async def test_query_creates_agent_span(tracer_provider, in_memory_span_exporter):
    """Test that query() creates root AGENT span."""
    instrumentor = ClaudeCodeInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider)

    try:
        from claude_agent_sdk import query

        # Mock the query to return a simple message
        # (This test will need mocking in real implementation)
        async def mock_query(*args, **kwargs):
            yield AssistantMessage(
                content=[TextBlock(type="text", text="4")]
            )

        # Temporarily replace query for testing
        import claude_agent_sdk
        original_query = claude_agent_sdk.query
        claude_agent_sdk.query = mock_query

        async for message in claude_agent_sdk.query(prompt="What is 2+2?"):
            pass

        claude_agent_sdk.query = original_query

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) >= 1

        root_span = spans[-1]  # Last span should be root
        assert root_span.attributes["openinference.span.kind"] == "AGENT"
        assert "Claude Code" in root_span.name

    finally:
        instrumentor.uninstrument()

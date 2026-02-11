import pytest
from openinference.instrumentation import using_session, using_user

from openinference.instrumentation.claude_code import ClaudeCodeInstrumentor


@pytest.mark.asyncio
async def test_context_attributes_propagation(tracer_provider, in_memory_span_exporter):
    """Test that context attributes are attached to spans."""
    instrumentor = ClaudeCodeInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider)

    try:
        from claude_agent_sdk import AssistantMessage, TextBlock, query

        # Mock query
        async def mock_query(*args, **kwargs):
            yield AssistantMessage(content=[TextBlock(type="text", text="context test")])

        import claude_agent_sdk

        original = claude_agent_sdk.query
        claude_agent_sdk.query = mock_query

        # Use context attributes
        with using_session("test-session-123"):
            with using_user("user-456"):
                async for message in claude_agent_sdk.query(prompt="test"):
                    pass

        claude_agent_sdk.query = original

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) >= 1

        root_span = spans[-1]
        assert root_span.attributes.get("session.id") == "test-session-123"
        assert root_span.attributes.get("user.id") == "user-456"

    finally:
        instrumentor.uninstrument()

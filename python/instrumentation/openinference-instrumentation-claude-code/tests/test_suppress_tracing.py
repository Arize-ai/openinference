import pytest
from opentelemetry import context as context_api
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY

from openinference.instrumentation.claude_code import ClaudeCodeInstrumentor


@pytest.mark.asyncio
async def test_suppress_tracing_query(tracer_provider, in_memory_span_exporter):
    """Test that tracing is suppressed when context flag is set."""
    instrumentor = ClaudeCodeInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider)

    try:
        from claude_agent_sdk import AssistantMessage, TextBlock, query

        # Mock query
        async def mock_query(*args, **kwargs):
            yield AssistantMessage(
                content=[TextBlock(type="text", text="suppressed")]
            )

        import claude_agent_sdk
        original = claude_agent_sdk.query
        claude_agent_sdk.query = mock_query

        # Set suppression context
        token = context_api.attach(
            context_api.set_value(_SUPPRESS_INSTRUMENTATION_KEY, True)
        )

        try:
            async for message in claude_agent_sdk.query(prompt="test"):
                pass
        finally:
            context_api.detach(token)
            claude_agent_sdk.query = original

        # Assert no spans were created
        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 0

    finally:
        instrumentor.uninstrument()

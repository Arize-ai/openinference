import pytest
from claude_agent_sdk import AssistantMessage, TextBlock, ToolUseBlock

from openinference.instrumentation.claude_code import ClaudeCodeInstrumentor


@pytest.mark.asyncio
async def test_tool_use_creates_tool_span(tracer_provider, in_memory_span_exporter):
    """Test that ToolUseBlock creates TOOL span."""
    instrumentor = ClaudeCodeInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider)

    try:
        from claude_agent_sdk import query

        # Mock query to return message with tool use
        async def mock_query(*args, **kwargs):
            yield AssistantMessage(
                content=[
                    TextBlock(type="text", text="I'll read the file"),
                    ToolUseBlock(
                        type="tool_use",
                        id="tool_123",
                        name="Read",
                        input={"file_path": "test.py"}
                    ),
                ]
            )

        import claude_agent_sdk
        original = claude_agent_sdk.query
        claude_agent_sdk.query = mock_query

        async for message in claude_agent_sdk.query(prompt="Read test.py"):
            pass

        claude_agent_sdk.query = original

        spans = in_memory_span_exporter.get_finished_spans()
        tool_spans = [
            s for s in spans
            if s.attributes.get("openinference.span.kind") == "TOOL"
        ]

        assert len(tool_spans) >= 1
        tool_span = tool_spans[0]
        assert tool_span.attributes["tool.name"] == "Read"
        assert "Read" in tool_span.name

    finally:
        instrumentor.uninstrument()

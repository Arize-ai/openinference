from opentelemetry.util._importlib_metadata import entry_points

from openinference.instrumentation.claude_agent_sdk import ClaudeAgentSDKInstrumentor


def test_entrypoint_opentelemetry_instrumentor() -> None:
    (ep,) = entry_points(
        group="opentelemetry_instrumentor",
        name="claude_agent_sdk",
    )
    instrumentor = ep.load()()
    assert isinstance(instrumentor, ClaudeAgentSDKInstrumentor)


def test_entrypoint_openinference_instrumentor() -> None:
    (ep,) = entry_points(
        group="openinference_instrumentor",
        name="claude_agent_sdk",
    )
    instrumentor = ep.load()()
    assert isinstance(instrumentor, ClaudeAgentSDKInstrumentor)

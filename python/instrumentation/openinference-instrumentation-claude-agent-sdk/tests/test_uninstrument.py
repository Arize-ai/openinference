"""Tests that instrument/uninstrument correctly patch and restore query and client."""

import importlib

from openinference.instrumentation.claude_agent_sdk import ClaudeAgentSDKInstrumentor


def test_instrument_uninstrument_restores_query_and_client() -> None:
    query_module = importlib.import_module("claude_agent_sdk.query")
    client_module = importlib.import_module("claude_agent_sdk.client")
    ClaudeSDKClient = getattr(client_module, "ClaudeSDKClient", None)

    original_query = query_module.query
    if ClaudeSDKClient is not None:
        original_connect = ClaudeSDKClient.connect
        original_client_query = ClaudeSDKClient.query
        original_receive_response = ClaudeSDKClient.receive_response

    instrumentor = ClaudeAgentSDKInstrumentor()
    instrumentor.instrument()

    assert query_module.query is not original_query
    if ClaudeSDKClient is not None:
        assert ClaudeSDKClient.connect is not original_connect
        assert ClaudeSDKClient.query is not original_client_query
        assert ClaudeSDKClient.receive_response is not original_receive_response

    instrumentor.uninstrument()

    assert query_module.query is original_query
    if ClaudeSDKClient is not None:
        assert ClaudeSDKClient.connect is original_connect
        assert ClaudeSDKClient.query is original_client_query
        assert ClaudeSDKClient.receive_response is original_receive_response

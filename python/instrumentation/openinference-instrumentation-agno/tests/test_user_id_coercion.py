"""Tests for user_id string coercion in Agno instrumentation.

Verifies that:
1. Integer user_id values are coerced to str (OpenInference spec requires string).
2. Falsy-but-valid user_id values (e.g. numeric zero) are NOT silently dropped.
3. None user_id is correctly omitted from the span attributes.
4. String user_id values pass through unchanged.

Fixes #3404.
"""

from typing import Any, Mapping
from unittest.mock import MagicMock

import pytest

from openinference.instrumentation.agno._runs_wrapper import (
    _run_arguments,
    _agent_run_attributes,
)
from openinference.semconv.trace import SpanAttributes

USER_ID = SpanAttributes.USER_ID


class _FakeAgent:
    """Minimal stand-in for agno.agent.Agent with just the fields we need."""

    def __init__(self, user_id: Any = None, name: str = "TestAgent") -> None:
        self.user_id = user_id
        self.name = name
        self.id = "agent-id-123"
        self.metadata = None


def _attrs_from_arguments(arguments: Mapping[str, Any]) -> dict:
    return dict(_run_arguments(arguments))


class TestRunArgumentsUserIdCoercion:
    def test_string_user_id_passes_through(self) -> None:
        attrs = _attrs_from_arguments({"user_id": "alice"})
        assert attrs[USER_ID] == "alice"

    def test_integer_user_id_coerced_to_str(self) -> None:
        attrs = _attrs_from_arguments({"user_id": 123})
        assert attrs[USER_ID] == "123", "int user_id must be cast to str per OpenInference spec"

    def test_zero_user_id_not_dropped(self) -> None:
        """user_id=0 is falsy but valid; must not be silently dropped."""
        attrs = _attrs_from_arguments({"user_id": 0})
        assert USER_ID in attrs, "user_id=0 must not be silently dropped"
        assert attrs[USER_ID] == "0"

    def test_none_user_id_omitted(self) -> None:
        attrs = _attrs_from_arguments({"user_id": None})
        assert USER_ID not in attrs

    def test_missing_user_id_omitted(self) -> None:
        attrs = _attrs_from_arguments({})
        assert USER_ID not in attrs


class TestAgentRunAttributesUserIdCoercion:
    """Tests for _agent_run_attributes with various user_id types on the agent object."""

    def _get_user_id_attr(self, agent: _FakeAgent) -> Any:
        # Import Agent and Team to satisfy isinstance checks
        try:
            from agno.agent import Agent
            from agno.team import Team
        except ImportError:
            pytest.skip("agno not installed")

        # Build a real Agent with our user_id
        real_agent = Agent.__new__(Agent)
        real_agent.name = agent.name
        real_agent.id = agent.id
        real_agent.user_id = agent.user_id
        real_agent.metadata = None
        real_agent.team_id = None

        attrs = dict(_agent_run_attributes(real_agent))
        return attrs.get(USER_ID)

    def test_string_user_id_on_agent(self) -> None:
        result = self._get_user_id_attr(_FakeAgent(user_id="bob"))
        assert result == "bob"

    def test_integer_user_id_on_agent_coerced(self) -> None:
        result = self._get_user_id_attr(_FakeAgent(user_id=42))
        assert result == "42", "Integer user_id on Agent must be coerced to str"

    def test_zero_user_id_on_agent_not_dropped(self) -> None:
        result = self._get_user_id_attr(_FakeAgent(user_id=0))
        assert result == "0", "user_id=0 on Agent must not be silently dropped"

    def test_none_user_id_on_agent_omitted(self) -> None:
        result = self._get_user_id_attr(_FakeAgent(user_id=None))
        assert result is None

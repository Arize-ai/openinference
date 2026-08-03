"""Tests for run-argument attribute extraction.

`user.id` used to be emitted behind a truthiness check, so an identifier that is
present but falsy (`0`, `""`) was dropped from the span with no error and no log
line -- the user simply saw a blank `user.id`.
"""

from typing import Any, Dict, Mapping

from openinference.instrumentation.agno._runs_wrapper import _run_arguments
from openinference.semconv.trace import SpanAttributes


def _attributes(arguments: Mapping[str, Any]) -> Dict[str, Any]:
    return dict(_run_arguments(arguments))


def test_falsy_user_id_is_still_recorded() -> None:
    assert _attributes({"user_id": 0})[SpanAttributes.USER_ID] == 0
    assert _attributes({"user_id": ""})[SpanAttributes.USER_ID] == ""


def test_truthy_user_id_is_unchanged() -> None:
    assert _attributes({"user_id": "user-1"})[SpanAttributes.USER_ID] == "user-1"


def test_absent_user_id_is_omitted() -> None:
    assert SpanAttributes.USER_ID not in _attributes({})
    assert SpanAttributes.USER_ID not in _attributes({"user_id": None})

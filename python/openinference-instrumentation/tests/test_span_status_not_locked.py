"""
Regression test for span status OK locking bug (closes #3415).

Before the fix, _chain_context, _llm_context, and _tool_context
unconditionally set StatusCode.OK after yield. On ADK 2.x, this
prevented callers from setting ERROR on failure paths because
OpenTelemetry ignores status changes once OK is set.

The fix removes the unconditional OK — OTel defaults to UNSET which
is treated as success for non-error spans and does not block later
ERROR updates.
"""

import pytest
from unittest.mock import MagicMock
from opentelemetry.trace import StatusCode, Status


def _make_mock_tracer():
    """Create a mock tracer whose start_as_current_span returns a real
    context manager that yields a mock span."""
    mock_span = MagicMock()
    mock_tracer = MagicMock()

    from contextlib import contextmanager

    @contextmanager
    def _fake_span(*args, **kwargs):
        yield mock_span

    mock_tracer.start_as_current_span = _fake_span
    return mock_tracer, mock_span


class TestSpanStatusNotLockedAtOK:
    """Verify context managers do not lock span status at OK."""

    def test_chain_context_does_not_set_ok(self):
        from openinference.instrumentation._tracers import _chain_context

        mock_tracer, mock_span = _make_mock_tracer()

        def dummy_fn(x):
            return x

        with _chain_context(
            tracer=mock_tracer,
            name="test-span",
            kind="chain",
            wrapped=dummy_fn,
            instance=None,
            args=("hello",),
            kwargs={},
        ):
            pass  # simulate normal execution

        # set_status must NOT have been called with OK
        for call in mock_span.set_status.call_args_list:
            args, kwargs = call
            status = args[0] if args else kwargs.get("status")
            if isinstance(status, Status):
                assert status.status_code != StatusCode.OK, (
                    "_chain_context must not set OK — blocks ERROR on failure"
                )

    def test_llm_context_does_not_set_ok(self):
        from openinference.instrumentation._tracers import _llm_context

        mock_tracer, mock_span = _make_mock_tracer()

        def dummy_fn(prompt):
            return prompt

        with _llm_context(
            tracer=mock_tracer,
            name="test-llm",
            process_input=None,
            process_output=None,
            wrapped=dummy_fn,
            instance=None,
            args=("hello",),
            kwargs={},
        ):
            pass

        for call in mock_span.set_status.call_args_list:
            args, kwargs = call
            status = args[0] if args else kwargs.get("status")
            if isinstance(status, Status):
                assert status.status_code != StatusCode.OK, (
                    "_llm_context must not set OK — blocks ERROR on failure"
                )

    def test_tool_context_does_not_set_ok(self):
        from openinference.instrumentation._tracers import _tool_context

        mock_tracer, mock_span = _make_mock_tracer()

        def dummy_fn(arg1):
            return arg1

        with _tool_context(
            tracer=mock_tracer,
            name="test-tool",
            description="A test tool",
            parameters={"type": "string"},
            wrapped=dummy_fn,
            instance=None,
            args=("hello",),
            kwargs={},
        ):
            pass

        for call in mock_span.set_status.call_args_list:
            args, kwargs = call
            status = args[0] if args else kwargs.get("status")
            if isinstance(status, Status):
                assert status.status_code != StatusCode.OK, (
                    "_tool_context must not set OK — blocks ERROR on failure"
                )

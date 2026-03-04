import json
import os
import re
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Generator, Iterator

import pytest
import yaml
from _pytest.monkeypatch import MonkeyPatch
from opentelemetry import trace as trace_api
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation.claude_agent_sdk import ClaudeAgentSDKInstrumentor

try:
    from claude_agent_sdk._internal.transport import Transport
except ImportError:
    Transport = object  # type: ignore[assignment,misc]

_FAKE_KEY = "test-key-no-real-call"
_USER_PATH_RE = re.compile(r"/Users/[^/]+|/home/[^/]+")


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, val in value.items():
            if key == "cwd" and isinstance(val, str):
                sanitized[key] = "<redacted>"
            else:
                sanitized[key] = _sanitize_value(val)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_value(item) for item in value)
    if isinstance(value, str):
        return _USER_PATH_RE.sub(
            lambda match: (
                "/Users/REDACTED" if match.group(0).startswith("/Users/") else "/home/REDACTED"
            ),
            value,
        )
    return value


def _sanitize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_sanitize_value(message) for message in messages]


def _real_api_key() -> str | None:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    return key if key and key != _FAKE_KEY else None


class ReplayTransport(Transport):
    """Replays pre-recorded subprocess messages from a YAML cassette.

    Handles request-ID substitution: the SDK generates a new random request_id
    for each control_request. We intercept outgoing writes to capture the actual
    IDs, then substitute them into the cassette's control_response messages so the
    SDK's pending_control_responses dict matches correctly.
    """

    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self._messages = messages
        # SDK-originated control_request IDs, captured in write() order
        self._sdk_control_request_ids: list[str] = []

    async def connect(self) -> None:
        pass

    async def write(self, data: str) -> None:
        """Capture SDK-originated control_request IDs for ID substitution."""
        try:
            msg = json.loads(data.strip())
        except json.JSONDecodeError:
            return
        if msg.get("type") == "control_request":
            self._sdk_control_request_ids.append(msg["request_id"])

    def read_messages(self) -> AsyncIterator[dict[str, Any]]:  # sync def, NOT async
        return self._gen()

    async def _gen(self) -> AsyncIterator[dict[str, Any]]:
        response_idx = 0  # index into SDK control_request IDs (one per control_response)
        for msg in self._messages:
            if msg.get("type") == "control_response":
                # Substitute cassette placeholder ID with the actual SDK-generated ID
                if response_idx < len(self._sdk_control_request_ids):
                    actual_id = self._sdk_control_request_ids[response_idx]
                    response_idx += 1
                    msg = {**msg, "response": {**msg["response"], "request_id": actual_id}}
            yield msg

    async def close(self) -> None:
        pass

    async def end_input(self) -> None:
        pass

    def is_ready(self) -> bool:
        return True


def _cassette_path(test_name: str) -> Path:
    return Path(__file__).parent / "cassettes" / "test_instrumentor" / f"{test_name}.yaml"


@pytest.fixture
def cassette_transport(
    request: pytest.FixtureRequest,
) -> Generator[Any, None, None]:
    """
    record_mode="once": replay if cassette exists; record if missing + real key set;
    fail if missing + no real key.

    In record mode, yields None so tests pass transport=None to the SDK, letting
    InternalClient create the real SubprocessCLITransport (patched to capture messages).
    """
    test_name = request.node.name
    path = _cassette_path(test_name)

    if path.exists():
        # --- REPLAY ---
        data = yaml.safe_load(path.read_text())
        messages = data.get("messages") or []
        if not messages:
            pytest.fail(
                f"Cassette at {path} is empty. Delete it and re-record:\n"
                "  env -u CLAUDECODE ANTHROPIC_API_KEY=<real-key> "
                "pytest tests/ -k <test_name>"
            )
        yield ReplayTransport(messages=messages)
        return

    # cassette missing — need to record
    key = _real_api_key()
    if not key:
        pytest.fail(
            f"Cassette not found: {path}\n"
            "Set ANTHROPIC_API_KEY to a real key (and have the claude CLI installed) "
            "to record a new cassette, then commit it."
        )

    # --- RECORD ---
    # Must patch both:
    # - subprocess_cli.SubprocessCLITransport: used by ClaudeSDKClient.connect() via
    #   a fresh local import (`from ._internal.transport.subprocess_cli import ...`)
    # - _client_mod.SubprocessCLITransport: used by InternalClient.process_query()
    #   via a module-level import (already resolved at import time)
    try:
        import claude_agent_sdk._internal.client as _client_mod
        import claude_agent_sdk._internal.transport.subprocess_cli as _sc_mod
        from claude_agent_sdk._internal.transport.subprocess_cli import SubprocessCLITransport

        OrigTransport = SubprocessCLITransport
        captured: list[dict[str, Any]] = []

        class _CapturingTransport(OrigTransport):  # type: ignore[misc,valid-type]
            def read_messages(self) -> AsyncIterator[dict[str, Any]]:
                return self._capturing_gen()

            async def _capturing_gen(self) -> AsyncIterator[dict[str, Any]]:
                async for msg in super()._read_messages_impl():
                    captured.append(msg)
                    yield msg

        setattr(_sc_mod, "SubprocessCLITransport", _CapturingTransport)
        setattr(_client_mod, "SubprocessCLITransport", _CapturingTransport)
        yield None  # test uses transport=None → SDK creates _CapturingTransport
        setattr(_sc_mod, "SubprocessCLITransport", OrigTransport)
        setattr(_client_mod, "SubprocessCLITransport", OrigTransport)
    except Exception:
        yield None
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump({"messages": _sanitize_messages(captured)}, default_flow_style=False))


@pytest.fixture
def in_memory_span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture
def tracer_provider(
    in_memory_span_exporter: InMemorySpanExporter,
) -> trace_api.TracerProvider:
    tracer_provider = trace_sdk.TracerProvider()
    span_processor = SimpleSpanProcessor(span_exporter=in_memory_span_exporter)
    tracer_provider.add_span_processor(span_processor=span_processor)
    return tracer_provider


@pytest.fixture
def instrument(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> Iterator[None]:
    ClaudeAgentSDKInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    ClaudeAgentSDKInstrumentor().uninstrument()


@pytest.fixture(autouse=True)
def api_key(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("GIT_TERMINAL_PROMPT", "0")
    monkeypatch.setenv("GIT_SSH_COMMAND", "ssh -o BatchMode=yes -o StrictHostKeyChecking=no")
    monkeypatch.setenv("SSH_ASKPASS_REQUIRE", "force")
    monkeypatch.setenv("SSH_ASKPASS", "/usr/bin/false")
    monkeypatch.setenv("DISPLAY", ":0")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-no-real-call")

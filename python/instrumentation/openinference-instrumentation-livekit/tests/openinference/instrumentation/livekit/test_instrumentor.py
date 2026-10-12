import json
import sys
import types
from pathlib import Path
from typing import Any, Iterator

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from openinference.instrumentation import (
    REDACTED_VALUE,
    TraceConfig,
    suppress_tracing,
    using_attributes,
)
from openinference.instrumentation.livekit import LiveKitInstrumentor
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes

_EXAMPLES = Path(__file__).resolve().parents[4] / "examples"
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

from _stubs import install_livekit_stubs  # noqa: E402

CONNECT_URL = "wss://example.livekit.cloud"
CONNECT_TOKEN = "super-secret-jwt-token"
RPC_METHOD = "echo"
RPC_PAYLOAD = '{"hello":"world"}'
RPC_RESULT = "pong"
TWIRP_SERVICE = "RoomService"
TWIRP_METHOD = "ListRooms"
SENTINEL_HEADER = "Bearer sentinel-auth-value"


@pytest.fixture()
def in_memory_span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture()
def tracer_provider(in_memory_span_exporter: InMemorySpanExporter) -> TracerProvider:
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))
    return tracer_provider


@pytest.fixture()
def instrumented(tracer_provider: TracerProvider) -> Iterator[None]:
    install_livekit_stubs()
    LiveKitInstrumentor().instrument(tracer_provider=tracer_provider, skip_dep_check=True)
    yield
    LiveKitInstrumentor().uninstrument()


def _require_str(value: object) -> str:
    if not isinstance(value, str):
        raise AssertionError(value)
    return value


def _require_tags(value: object) -> list[str]:
    if not isinstance(value, (list, tuple)):
        raise AssertionError(value)
    tags = [item for item in value if isinstance(item, str)]
    if len(tags) != len(value):
        raise AssertionError(value)
    return tags


def _all_attribute_values(attributes: dict[str, Any] | None) -> str:
    if not attributes:
        return ""
    return json.dumps(dict(attributes))


@pytest.mark.asyncio
async def test_livekit_instrumentor_spans(
    instrumented: None,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    from livekit.api.twirp_client import TwirpClient
    from livekit.rtc.participant import LocalParticipant
    from livekit.rtc.room import Room

    room = Room()
    await room.connect(CONNECT_URL, CONNECT_TOKEN)
    await room.disconnect()

    participant = LocalParticipant()
    result = await participant.perform_rpc(
        destination_identity="peer",
        method=RPC_METHOD,
        payload=RPC_PAYLOAD,
    )
    assert result == RPC_RESULT

    client = TwirpClient(host="https://example.livekit.cloud", pkg="livekit")
    await client.request(
        TWIRP_SERVICE,
        TWIRP_METHOD,
        data=types.SimpleNamespace(),
        headers={"Authorization": SENTINEL_HEADER},
        response_class=object,
    )

    spans = {span.name: span for span in in_memory_span_exporter.get_finished_spans()}
    assert len(spans) == 4

    connect = spans["Room.connect"]
    connect_attrs = dict(connect.attributes or {})
    assert (
        connect_attrs[SpanAttributes.OPENINFERENCE_SPAN_KIND]
        == OpenInferenceSpanKindValues.CHAIN.value
    )
    input_value = _require_str(connect_attrs[SpanAttributes.INPUT_VALUE])
    assert CONNECT_URL in input_value
    assert CONNECT_TOKEN not in _all_attribute_values(connect_attrs)

    rpc = spans["LocalParticipant.perform_rpc"]
    rpc_attrs = dict(rpc.attributes or {})
    assert (
        rpc_attrs[SpanAttributes.OPENINFERENCE_SPAN_KIND] == OpenInferenceSpanKindValues.TOOL.value
    )
    rpc_input = json.loads(_require_str(rpc_attrs[SpanAttributes.INPUT_VALUE]))
    assert rpc_input["method"] == RPC_METHOD
    assert rpc_input["payload"] == RPC_PAYLOAD
    assert rpc_attrs[SpanAttributes.OUTPUT_VALUE] == RPC_RESULT

    twirp = spans["TwirpClient.request"]
    twirp_attrs = dict(twirp.attributes or {})
    assert (
        twirp_attrs[SpanAttributes.OPENINFERENCE_SPAN_KIND]
        == OpenInferenceSpanKindValues.CHAIN.value
    )
    twirp_input = _require_str(twirp_attrs[SpanAttributes.INPUT_VALUE])
    assert TWIRP_SERVICE in twirp_input
    assert TWIRP_METHOD in twirp_input
    assert SENTINEL_HEADER not in _all_attribute_values(twirp_attrs)


@pytest.mark.asyncio
async def test_suppress_tracing_skips_room_connect(
    instrumented: None,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    from livekit.rtc.room import Room

    room = Room()
    with suppress_tracing():
        await room.connect(CONNECT_URL, CONNECT_TOKEN)

    assert len(in_memory_span_exporter.get_finished_spans()) == 0


@pytest.mark.asyncio
async def test_context_attributes_propagate(
    instrumented: None,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    from livekit.rtc.room import Room

    with using_attributes(
        session_id="session-1",
        user_id="user-1",
        metadata={"env": "test"},
        tags=["tag-1", "tag-2"],
    ):
        await Room().connect(CONNECT_URL, CONNECT_TOKEN)

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = dict(spans[0].attributes or {})
    assert attrs[SpanAttributes.SESSION_ID] == "session-1"
    assert attrs[SpanAttributes.USER_ID] == "user-1"
    assert json.loads(str(attrs[SpanAttributes.METADATA])) == {"env": "test"}
    assert _require_tags(attrs[SpanAttributes.TAG_TAGS]) == ["tag-1", "tag-2"]


@pytest.mark.asyncio
async def test_trace_config_hides_inputs(
    instrumented: None,
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    from livekit.rtc.room import Room

    LiveKitInstrumentor().uninstrument()
    LiveKitInstrumentor().instrument(
        tracer_provider=tracer_provider,
        config=TraceConfig(hide_inputs=True),
        skip_dep_check=True,
    )
    await Room().connect(CONNECT_URL, CONNECT_TOKEN)

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = dict(spans[0].attributes or {})
    assert attrs[SpanAttributes.INPUT_VALUE] == REDACTED_VALUE
    dumped = json.dumps({key: str(value) for key, value in attrs.items()})
    assert CONNECT_URL not in dumped
    assert CONNECT_TOKEN not in dumped


@pytest.mark.asyncio
async def test_trace_config_hides_outputs(
    instrumented: None,
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    from livekit.rtc.participant import LocalParticipant

    LiveKitInstrumentor().uninstrument()
    LiveKitInstrumentor().instrument(
        tracer_provider=tracer_provider,
        config=TraceConfig(hide_outputs=True),
        skip_dep_check=True,
    )
    await LocalParticipant().perform_rpc(
        destination_identity="peer",
        method=RPC_METHOD,
        payload=RPC_PAYLOAD,
    )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = dict(spans[0].attributes or {})
    assert attrs[SpanAttributes.OUTPUT_VALUE] == REDACTED_VALUE
    assert RPC_RESULT not in json.dumps({key: str(value) for key, value in attrs.items()})


@pytest.mark.asyncio
async def test_uninstrument_restores_the_original(
    instrumented: None,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    from livekit.rtc.room import Room

    LiveKitInstrumentor().uninstrument()
    in_memory_span_exporter.clear()
    await Room().connect(CONNECT_URL, CONNECT_TOKEN)
    assert len(in_memory_span_exporter.get_finished_spans()) == 0


@pytest.mark.asyncio
async def test_connect_error_sets_span_status(
    instrumented: None,
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    from livekit.rtc.room import Room

    async def boom(self: Any, url: str, token: str, options: Any = None) -> None:
        raise RuntimeError("connect failed")

    LiveKitInstrumentor().uninstrument()
    Room.connect = boom
    LiveKitInstrumentor().instrument(tracer_provider=tracer_provider, skip_dep_check=True)

    with pytest.raises(RuntimeError, match="connect failed"):
        await Room().connect(CONNECT_URL, CONNECT_TOKEN)

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code == StatusCode.ERROR
    assert spans[0].status.description == "connect failed"


@pytest.mark.asyncio
async def test_missing_twirp_module_is_skipped(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    install_livekit_stubs()
    sys.modules.pop("livekit.api.twirp_client", None)
    instrumentor = LiveKitInstrumentor()
    instrumentor.uninstrument()
    instrumentor.instrument(tracer_provider=tracer_provider, skip_dep_check=True)
    try:
        from livekit.rtc.room import Room

        await Room().connect(CONNECT_URL, CONNECT_TOKEN)
        names = {span.name for span in in_memory_span_exporter.get_finished_spans()}
        assert names == {"Room.connect"}
        assert "livekit.api.twirp_client" not in sys.modules
    finally:
        instrumentor.uninstrument()

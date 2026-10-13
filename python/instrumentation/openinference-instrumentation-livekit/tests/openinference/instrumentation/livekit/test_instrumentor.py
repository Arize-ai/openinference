import importlib
import json
import sys
from typing import Any, Iterator, cast

import pytest
from livekit import api
from livekit.api.twirp_client import TwirpClient
from livekit.rtc._ffi_client import FfiClient, FfiHandle
from livekit.rtc._proto import ffi_pb2
from livekit.rtc.room import ConnectError, Room
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
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

CONNECT_URL = "wss://example.livekit.cloud"
CONNECT_TOKEN = "super-secret-jwt-token"
FAIL_URL = "wss://down.example.livekit.cloud"
RPC_METHOD = "echo"
RPC_PAYLOAD = '{"hello":"world"}'
RPC_RESULT = "pong"
TWIRP_SERVICE = "RoomService"
TWIRP_METHOD = "ListRooms"
SENTINEL_HEADER = "Bearer sentinel-auth-value"
_ROOM_HANDLE = 11
_PARTICIPANT_HANDLE = 22


class _RecordingSession:
    timeout = None

    def __init__(self) -> None:
        self.headers: list[dict[str, str]] = []

    def post(
        self,
        url: str,
        headers: dict[str, str],
        data: bytes,
        timeout: Any,
    ) -> "_Response":
        del url, data, timeout
        self.headers.append(dict(headers))
        return _Response()


class _Response:
    status = 200

    async def read(self) -> bytes:
        return bytes(api.ListRoomsResponse().SerializeToString())

    async def __aenter__(self) -> "_Response":
        return self

    async def __aexit__(self, *args: object) -> None:
        return None


class _Transport:
    def __init__(self) -> None:
        self.tokens: list[str] = []
        self.session = _RecordingSession()
        self._next_id = 1
        self._request = FfiClient.instance.request
        original_init = FfiHandle.__init__
        self._init_handle = original_init

        def _init(handle_self: Any, handle: int) -> None:
            original_init(handle_self, handle)
            # This id was never allocated by the native library, and dropping it asserts.
            if handle in (_ROOM_HANDLE, _PARTICIPANT_HANDLE):
                handle_self._disposed = True

        setattr(FfiHandle, "__init__", _init)
        setattr(FfiClient.instance, "request", self._request_and_reply)

    def close(self) -> None:
        setattr(FfiClient.instance, "request", self._request)
        setattr(FfiHandle, "__init__", self._init_handle)

    def _request_and_reply(self, req: Any) -> Any:
        which = req.WhichOneof("message")
        response = ffi_pb2.FfiResponse()
        if which == "connect":
            self.tokens.append(req.connect.token)
            async_id = self._take_id()
            response.connect.async_id = async_id
            event = ffi_pb2.FfiEvent()
            event.connect.async_id = async_id
            if req.connect.url == FAIL_URL:
                event.connect.error = "connect failed"
            else:
                event.connect.result.room.handle.id = _ROOM_HANDLE
                event.connect.result.room.info.sid = "RM_test"
                event.connect.result.room.info.name = "room"
                event.connect.result.local_participant.handle.id = _PARTICIPANT_HANDLE
                event.connect.result.local_participant.info.identity = "caller"
                event.connect.result.local_participant.info.sid = "PA_test"
            FfiClient.instance.queue.put(event)
            return response
        if which == "disconnect":
            async_id = self._take_id()
            response.disconnect.async_id = async_id
            done = ffi_pb2.FfiEvent()
            done.disconnect.async_id = async_id
            eos = ffi_pb2.FfiEvent()
            eos.room_event.room_handle = req.disconnect.room_handle
            eos.room_event.eos.SetInParent()
            FfiClient.instance.queue.put(done)
            FfiClient.instance.queue.put(eos)
            return response
        if which == "perform_rpc":
            async_id = self._take_id()
            response.perform_rpc.async_id = async_id
            event = ffi_pb2.FfiEvent()
            event.perform_rpc.async_id = async_id
            event.perform_rpc.payload = RPC_RESULT
            FfiClient.instance.queue.put(event)
            return response
        return response

    def _take_id(self) -> int:
        async_id = self._next_id
        self._next_id += 1
        return async_id


@pytest.fixture()
def in_memory_span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture()
def tracer_provider(in_memory_span_exporter: InMemorySpanExporter) -> TracerProvider:
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))
    return tracer_provider


@pytest.fixture()
def transport() -> Iterator[_Transport]:
    installed = _Transport()
    try:
        yield installed
    finally:
        installed.close()


@pytest.fixture()
def instrumented(tracer_provider: TracerProvider, transport: _Transport) -> Iterator[None]:
    del transport
    LiveKitInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    LiveKitInstrumentor().uninstrument()


def _attributes(span: ReadableSpan) -> dict[str, Any]:
    return dict(span.attributes or {})


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


@pytest.mark.asyncio
async def test_livekit_instrumentor_spans(
    instrumented: None,
    transport: _Transport,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    room = Room()
    await room.connect(CONNECT_URL, CONNECT_TOKEN)
    result = await room.local_participant.perform_rpc(
        destination_identity="peer",
        method=RPC_METHOD,
        payload=RPC_PAYLOAD,
    )
    await room.disconnect()

    client = TwirpClient(cast(Any, transport.session), "https://example.livekit.cloud", "livekit")
    listed = await client.request(
        TWIRP_SERVICE,
        TWIRP_METHOD,
        data=api.ListRoomsRequest(),
        headers={"Authorization": SENTINEL_HEADER},
        response_class=api.ListRoomsResponse,
    )

    assert result == RPC_RESULT
    assert isinstance(listed, api.ListRoomsResponse)
    assert transport.tokens == [CONNECT_TOKEN]
    assert transport.session.headers[0]["Authorization"] == SENTINEL_HEADER

    spans = in_memory_span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "Room.connect",
        "LocalParticipant.perform_rpc",
        "Room.disconnect",
        "TwirpClient.request",
    ]
    assert all(span.parent is None for span in spans)

    connect = _attributes(spans[0])
    assert (
        connect.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND)
        == OpenInferenceSpanKindValues.CHAIN.value
    )
    assert json.loads(_require_str(connect.pop(SpanAttributes.INPUT_VALUE))) == {"url": CONNECT_URL}
    assert connect.pop(SpanAttributes.INPUT_MIME_TYPE) == "application/json"
    assert CONNECT_TOKEN not in json.dumps(connect)
    assert not connect

    rpc = _attributes(spans[1])
    assert rpc.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.TOOL.value
    assert json.loads(_require_str(rpc.pop(SpanAttributes.INPUT_VALUE))) == {
        "method": RPC_METHOD,
        "payload": RPC_PAYLOAD,
    }
    assert rpc.pop(SpanAttributes.INPUT_MIME_TYPE) == "application/json"
    assert rpc.pop(SpanAttributes.TOOL_NAME) == RPC_METHOD
    assert rpc.pop(SpanAttributes.OUTPUT_VALUE) == RPC_RESULT
    assert rpc.pop(SpanAttributes.OUTPUT_MIME_TYPE) == "text/plain"
    assert not rpc

    disconnect = _attributes(spans[2])
    assert (
        disconnect.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND)
        == OpenInferenceSpanKindValues.CHAIN.value
    )
    assert not disconnect

    twirp = _attributes(spans[3])
    assert (
        twirp.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.CHAIN.value
    )
    assert json.loads(_require_str(twirp.pop(SpanAttributes.INPUT_VALUE))) == {
        "service": TWIRP_SERVICE,
        "method": TWIRP_METHOD,
    }
    assert twirp.pop(SpanAttributes.INPUT_MIME_TYPE) == "application/json"
    assert SENTINEL_HEADER not in json.dumps(twirp)
    assert not twirp


@pytest.mark.asyncio
async def test_suppress_tracing_skips_room_connect(
    instrumented: None,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    room = Room()
    with suppress_tracing():
        await room.connect(CONNECT_URL, CONNECT_TOKEN)

    assert room.isconnected()
    assert not in_memory_span_exporter.get_finished_spans()


@pytest.mark.asyncio
async def test_context_attributes_propagate(
    instrumented: None,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    with using_attributes(
        session_id="session-1",
        user_id="user-1",
        metadata={"env": "test"},
        tags=["tag-1", "tag-2"],
    ):
        await Room().connect(CONNECT_URL, CONNECT_TOKEN)

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = _attributes(spans[0])
    assert attrs.pop(SpanAttributes.SESSION_ID) == "session-1"
    assert attrs.pop(SpanAttributes.USER_ID) == "user-1"
    assert json.loads(_require_str(attrs.pop(SpanAttributes.METADATA))) == {"env": "test"}
    assert _require_tags(attrs.pop(SpanAttributes.TAG_TAGS)) == ["tag-1", "tag-2"]
    assert (
        attrs.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.CHAIN.value
    )
    assert json.loads(_require_str(attrs.pop(SpanAttributes.INPUT_VALUE))) == {"url": CONNECT_URL}
    assert attrs.pop(SpanAttributes.INPUT_MIME_TYPE) == "application/json"
    assert not attrs


@pytest.mark.asyncio
async def test_trace_config_hides_inputs(
    instrumented: None,
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    LiveKitInstrumentor().uninstrument()
    LiveKitInstrumentor().instrument(
        tracer_provider=tracer_provider,
        config=TraceConfig(hide_inputs=True),
    )
    await Room().connect(CONNECT_URL, CONNECT_TOKEN)

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = _attributes(spans[0])
    assert attrs.pop(SpanAttributes.INPUT_VALUE) == REDACTED_VALUE
    assert (
        attrs.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.CHAIN.value
    )
    dumped = json.dumps({key: str(value) for key, value in attrs.items()})
    assert CONNECT_URL not in dumped
    assert CONNECT_TOKEN not in dumped
    assert not attrs


@pytest.mark.asyncio
async def test_trace_config_hides_outputs(
    instrumented: None,
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    LiveKitInstrumentor().uninstrument()
    LiveKitInstrumentor().instrument(
        tracer_provider=tracer_provider,
        config=TraceConfig(hide_outputs=True),
    )
    room = Room()
    await room.connect(CONNECT_URL, CONNECT_TOKEN)
    in_memory_span_exporter.clear()
    await room.local_participant.perform_rpc(
        destination_identity="peer",
        method=RPC_METHOD,
        payload=RPC_PAYLOAD,
    )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = _attributes(spans[0])
    assert attrs.pop(SpanAttributes.OUTPUT_VALUE) == REDACTED_VALUE
    assert RPC_RESULT not in json.dumps({key: str(value) for key, value in attrs.items()})


@pytest.mark.asyncio
async def test_uninstrument_restores_the_original(
    instrumented: None,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    LiveKitInstrumentor().uninstrument()
    in_memory_span_exporter.clear()
    await Room().connect(CONNECT_URL, CONNECT_TOKEN)
    assert not in_memory_span_exporter.get_finished_spans()


@pytest.mark.asyncio
async def test_connect_error_sets_span_status(
    instrumented: None,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    with pytest.raises(ConnectError, match="connect failed"):
        await Room().connect(FAIL_URL, CONNECT_TOKEN)

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code == StatusCode.ERROR
    assert spans[0].status.description == "connect failed"
    attrs = _attributes(spans[0])
    assert json.loads(_require_str(attrs[SpanAttributes.INPUT_VALUE])) == {"url": FAIL_URL}
    assert CONNECT_TOKEN not in json.dumps({key: str(value) for key, value in attrs.items()})


class _MissingTwirp(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> None:
        if fullname == "livekit.api.twirp_client":
            raise ModuleNotFoundError(fullname)
        return None


@pytest.mark.asyncio
async def test_missing_twirp_module_is_skipped(
    tracer_provider: TracerProvider,
    transport: _Transport,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    saved = sys.modules.pop("livekit.api.twirp_client")
    blocker = _MissingTwirp()
    sys.meta_path.insert(0, blocker)
    instrumentor = LiveKitInstrumentor()
    instrumentor.uninstrument()
    try:
        instrumentor.instrument(tracer_provider=tracer_provider)
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("livekit.api.twirp_client")
        await Room().connect(CONNECT_URL, CONNECT_TOKEN)
        assert [span.name for span in in_memory_span_exporter.get_finished_spans()] == [
            "Room.connect"
        ]
    finally:
        sys.meta_path.remove(blocker)
        sys.modules["livekit.api.twirp_client"] = saved
        instrumentor.uninstrument()

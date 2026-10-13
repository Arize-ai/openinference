from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Tuple

from openinference.semconv.trace import OpenInferenceSpanKindValues

InputExtractor = Callable[[Tuple[Any, ...], Mapping[str, Any]], Optional[Any]]
ToolNameExtractor = Callable[[Tuple[Any, ...], Mapping[str, Any]], Optional[str]]


@dataclass(frozen=True)
class LiveKitWrapTarget:
    module: str
    method: str
    span_name: str
    span_kind: OpenInferenceSpanKindValues
    input_extractor: Optional[InputExtractor] = None
    tool_name_extractor: Optional[ToolNameExtractor] = None
    capture_output: bool = False


def _room_connect_input(args: Tuple[Any, ...], kwargs: Mapping[str, Any]) -> dict[str, Any]:
    url = args[0] if args else kwargs.get("url")
    return {"url": url}


def _perform_rpc_input(args: Tuple[Any, ...], kwargs: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "method": kwargs.get("method"),
        "payload": kwargs.get("payload"),
    }


def _perform_rpc_tool_name(args: Tuple[Any, ...], kwargs: Mapping[str, Any]) -> Optional[str]:
    method = kwargs.get("method")
    if isinstance(method, str):
        return method
    return None


def _twirp_request_input(args: Tuple[Any, ...], kwargs: Mapping[str, Any]) -> dict[str, Any]:
    service = args[0] if len(args) > 0 else kwargs.get("service")
    method = args[1] if len(args) > 1 else kwargs.get("method")
    return {"service": service, "method": method}


LIVEKIT_WRAP_TARGETS: Tuple[LiveKitWrapTarget, ...] = (
    LiveKitWrapTarget(
        module="livekit.rtc.room",
        method="Room.connect",
        span_name="Room.connect",
        span_kind=OpenInferenceSpanKindValues.CHAIN,
        input_extractor=_room_connect_input,
    ),
    LiveKitWrapTarget(
        module="livekit.rtc.room",
        method="Room.disconnect",
        span_name="Room.disconnect",
        span_kind=OpenInferenceSpanKindValues.CHAIN,
    ),
    LiveKitWrapTarget(
        module="livekit.rtc.participant",
        method="LocalParticipant.perform_rpc",
        span_name="LocalParticipant.perform_rpc",
        span_kind=OpenInferenceSpanKindValues.TOOL,
        input_extractor=_perform_rpc_input,
        tool_name_extractor=_perform_rpc_tool_name,
        capture_output=True,
    ),
    LiveKitWrapTarget(
        module="livekit.api.twirp_client",
        method="TwirpClient.request",
        span_name="TwirpClient.request",
        span_kind=OpenInferenceSpanKindValues.CHAIN,
        input_extractor=_twirp_request_input,
    ),
)

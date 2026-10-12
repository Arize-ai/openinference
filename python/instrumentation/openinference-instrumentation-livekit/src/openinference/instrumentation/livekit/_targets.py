from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Tuple

from openinference.semconv.trace import OpenInferenceSpanKindValues

InputExtractor = Callable[[Tuple[Any, ...], Mapping[str, Any]], Optional[Any]]


@dataclass(frozen=True)
class LiveKitWrapTarget:
    module: str
    method: str
    span_name: str
    span_kind: OpenInferenceSpanKindValues
    input_extractor: Optional[InputExtractor] = None
    capture_output: bool = False


def _room_connect_input(args: Tuple[Any, ...], kwargs: Mapping[str, Any]) -> dict[str, Any]:
    # `token` authorizes the socket. It stays off the span.
    url = args[0] if args else kwargs.get("url")
    return {"url": url}


def _perform_rpc_input(args: Tuple[Any, ...], kwargs: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "method": kwargs.get("method"),
        "payload": kwargs.get("payload"),
    }


def _twirp_request_input(args: Tuple[Any, ...], kwargs: Mapping[str, Any]) -> dict[str, Any]:
    # Headers can carry Authorization. Record the service and method only.
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

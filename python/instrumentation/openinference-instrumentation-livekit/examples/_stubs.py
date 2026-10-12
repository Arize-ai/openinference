import sys
import types
from typing import Any


class _SimpleResponse:
    pass


async def _stub_connect(self: Any, url: str, token: str, options: Any = None) -> None:
    return None


async def _stub_disconnect(self: Any) -> None:
    return None


async def _stub_perform_rpc(
    self: Any,
    *,
    destination_identity: str,
    method: str,
    payload: str,
    response_timeout: float | None = None,
    max_round_trip_latency: float | None = None,
) -> str:
    return "pong"


async def _stub_request(
    self: Any,
    service: str,
    method: str,
    data: Any,
    headers: dict[str, str],
    response_class: type,
    *,
    timeout: Any = None,
) -> _SimpleResponse:
    return _SimpleResponse()


def _ensure_parent_modules(qualified: str) -> types.ModuleType:
    parts = qualified.split(".")
    for i in range(1, len(parts) + 1):
        name = ".".join(parts[:i])
        if name not in sys.modules:
            mod = types.ModuleType(name)
            if i < len(parts):
                mod.__path__ = []
            sys.modules[name] = mod
    return sys.modules[qualified]


def install_livekit_stubs() -> None:
    room_mod = _ensure_parent_modules("livekit.rtc.room")
    participant_mod = _ensure_parent_modules("livekit.rtc.participant")
    twirp_mod = _ensure_parent_modules("livekit.api.twirp_client")

    if "livekit.rtc.room" in sys.modules and hasattr(room_mod, "Room"):
        room_mod.Room.connect = _stub_connect  # type: ignore[method-assign]
        room_mod.Room.disconnect = _stub_disconnect  # type: ignore[method-assign]
    else:

        class Room:
            connect = _stub_connect
            disconnect = _stub_disconnect

        room_mod.Room = Room  # type: ignore[attr-defined]

    if "livekit.rtc.participant" in sys.modules and hasattr(participant_mod, "LocalParticipant"):
        participant_mod.LocalParticipant.perform_rpc = _stub_perform_rpc  # type: ignore[method-assign]
    else:

        class LocalParticipant:
            perform_rpc = _stub_perform_rpc

        participant_mod.LocalParticipant = LocalParticipant  # type: ignore[attr-defined]

    if "livekit.api.twirp_client" in sys.modules and hasattr(twirp_mod, "TwirpClient"):
        twirp_mod.TwirpClient.request = _stub_request  # type: ignore[method-assign]
    else:

        class TwirpClient:
            def __init__(self, host: str, pkg: str, **kwargs: Any) -> None:
                self.host = host
                self.pkg = pkg

            request = _stub_request

        twirp_mod.TwirpClient = TwirpClient  # type: ignore[attr-defined]

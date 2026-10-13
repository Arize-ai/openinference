"""Call the LiveKit Python SDK the way the official examples do.

Token grants, ``Room.connect``, and ``perform_rpc`` follow
https://github.com/livekit/python-sdks/blob/main/examples/rpc.py
and the RPC section of that repository's README.
``LiveKitAPI`` room create and list follow the "Using Server API" section
of the same README.

Set ``LIVEKIT_URL``, ``LIVEKIT_API_KEY``, and ``LIVEKIT_API_SECRET``.
``livekit-server --dev --node-ip 127.0.0.1`` accepts
``ws://localhost:7880``, ``devkey``, and ``secret``.
"""

import asyncio
import os

from livekit import api, rtc
from livekit.rtc.rpc import RpcInvocationData

_REQUIRED_ENV = ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET")


def _env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        missing = ", ".join(_REQUIRED_ENV)
        raise SystemExit(f"Missing {name}. Set {missing}.")
    return value


def create_token(identity: str, room_name: str) -> str:
    token = (
        api.AccessToken(_env("LIVEKIT_API_KEY"), _env("LIVEKIT_API_SECRET"))
        .with_identity(identity)
        .with_grants(
            api.VideoGrants(
                room=room_name,
                room_join=True,
                can_publish=True,
                can_subscribe=True,
            )
        )
    )
    return token.to_jwt()


async def connect_participant(identity: str, room_name: str) -> rtc.Room:
    room = rtc.Room()
    token = create_token(identity, room_name)
    await room.connect(_env("LIVEKIT_URL"), token)

    async def wait_for_participants() -> None:
        if room.remote_participants:
            return
        participant_connected = asyncio.Event()

        def _on_participant_connected(_participant: rtc.RemoteParticipant) -> None:
            room.off("participant_connected", _on_participant_connected)
            participant_connected.set()

        room.on("participant_connected", _on_participant_connected)
        await participant_connected.wait()

    try:
        await asyncio.wait_for(wait_for_participants(), timeout=5.0)
    except asyncio.TimeoutError as exc:
        raise TimeoutError("Timed out waiting for participants") from exc

    return room


async def perform_greeting(room: rtc.Room) -> str:
    response = await room.local_participant.perform_rpc(
        destination_identity="greeter",
        method="greet",
        payload="Hello from RPC!",
    )
    print(f'RPC response: "{response}"')
    return response


async def run_livekit_calls() -> None:
    """Create a room, connect two participants, greet over RPC, then disconnect."""
    url = _env("LIVEKIT_URL")
    room_name = f"rpc-test-{os.urandom(4).hex()}"
    print(f"Connecting participants to room: {room_name}")

    lkapi = api.LiveKitAPI(url, _env("LIVEKIT_API_KEY"), _env("LIVEKIT_API_SECRET"))
    rooms: list[rtc.Room] = []
    try:
        room_info = await lkapi.room.create_room(api.CreateRoomRequest(name=room_name))
        print(room_info)
        listed = await lkapi.room.list_rooms(api.ListRoomsRequest())
        print(listed)

        callers_room, greeters_room = await asyncio.gather(
            connect_participant("caller", room_name),
            connect_participant("greeter", room_name),
        )
        rooms = [callers_room, greeters_room]

        @greeters_room.local_participant.register_rpc_method("greet")
        async def handle_greet(data: RpcInvocationData) -> str:
            print(f"Received greeting from {data.caller_identity}: {data.payload}")
            return f"Hello, {data.caller_identity}!"

        await perform_greeting(callers_room)
    finally:
        await asyncio.gather(*(room.disconnect() for room in rooms), return_exceptions=True)
        await lkapi.aclose()

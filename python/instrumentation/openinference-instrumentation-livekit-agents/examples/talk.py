"""Join a room as a user, ask the agent one question out loud, and wait for the answer.

The question is synthesized with OpenAI TTS and published as a microphone track, so the
examples run end to end without a person. Uses LIVEKIT_URL, LIVEKIT_API_KEY,
LIVEKIT_API_SECRET, and OPENAI_API_KEY.

  python talk.py ["What's the weather in London right now?"]
"""

import asyncio
import os
import sys

from livekit import api, rtc
from livekit.plugins import openai

QUESTION = "What's the weather in London right now?"
RATE = 24_000
# Long enough for the tool call and the spoken answer.
ANSWER_SECONDS = 20


def _token(room: str) -> str:
    grants = api.VideoGrants(room=room, room_join=True, can_publish=True, can_subscribe=True)
    return (
        api.AccessToken(os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"])
        .with_identity("user")
        .with_grants(grants)
        .to_jwt()
    )


async def main(question: str) -> None:
    room = rtc.Room()
    await room.connect(os.environ["LIVEKIT_URL"], _token(f"voice-{os.urandom(3).hex()}"))
    while not any(
        p.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT
        for p in room.remote_participants.values()
    ):
        await asyncio.sleep(0.2)
    source = rtc.AudioSource(RATE, 1)
    track = rtc.LocalAudioTrack.create_audio_track("microphone", source)
    options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
    await room.local_participant.publish_track(track, options)
    await asyncio.sleep(1)

    tts = openai.TTS(model="gpt-4o-mini-tts")
    async for audio in tts.synthesize(question):
        await source.capture_frame(audio.frame)
    silence = rtc.AudioFrame(b"\x00\x00" * (RATE // 100), RATE, 1, RATE // 100)
    for _ in range(200):
        await source.capture_frame(silence)
    print(f"asked: {question}")
    await asyncio.sleep(ANSWER_SECONDS)
    await tts.aclose()
    await room.disconnect()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else QUESTION))

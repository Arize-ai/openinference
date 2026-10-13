from __future__ import annotations

import logging
import time
from typing import Any

from livekit import rtc
from livekit.agents import AgentSession
from livekit.agents.voice import io

from openinference.instrumentation.livekit_agents._audio import AudioClip, SessionAudio

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class _TapAudioInput(io.AudioInput):
    def __init__(self, source: io.AudioInput, audio: SessionAudio) -> None:
        super().__init__(label="OpenInference", source=source)
        self._audio = audio

    async def __anext__(self) -> rtc.AudioFrame:
        frame = await super().__anext__()
        try:
            self._audio.add_input(
                time.time_ns(), bytes(frame.data), frame.sample_rate, frame.num_channels
            )
        except Exception:
            logger.exception("Failed to record LiveKit input audio")
        return frame


class _TapAudioOutput(io.AudioOutput):
    """Keeps each segment cut to its playback position, so it holds only what was heard."""

    def __init__(self, next_in_chain: io.AudioOutput, audio: SessionAudio) -> None:
        super().__init__(
            label="OpenInference",
            next_in_chain=next_in_chain,
            capabilities=io.AudioOutputCapabilities(pause=True),
        )
        self._audio = audio
        self._frames: list[rtc.AudioFrame] = []
        self._started_ns: int | None = None

    @property
    def sample_rate(self) -> int | None:
        if self._sample_rate is not None:
            return self._sample_rate
        return self.next_in_chain.sample_rate if self.next_in_chain else None

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        if self.next_in_chain:
            await self.next_in_chain.capture_frame(frame)
        await super().capture_frame(frame)
        self._frames.append(frame)

    def flush(self) -> None:
        super().flush()
        if self.next_in_chain:
            self.next_in_chain.flush()

    def clear_buffer(self) -> None:
        if self.next_in_chain:
            self.next_in_chain.clear_buffer()

    def on_playback_started(self, *, created_at: float) -> None:
        super().on_playback_started(created_at=created_at)
        if self._started_ns is None:
            self._started_ns = int(created_at * 1_000_000_000)

    def on_playback_finished(
        self,
        *,
        playback_position: float,
        interrupted: bool,
        synchronized_transcript: str | None = None,
    ) -> None:
        super().on_playback_finished(
            playback_position=playback_position,
            interrupted=interrupted,
            synchronized_transcript=synchronized_transcript,
        )
        frames, self._frames = self._frames, []
        started_ns, self._started_ns = self._started_ns, None
        try:
            if not frames:
                return
            rate, channels = frames[0].sample_rate, frames[0].num_channels
            pcm = b"".join(
                bytes(f.data) for f in frames if (f.sample_rate, f.num_channels) == (rate, channels)
            )
            played = round(playback_position * rate) * channels * 2
            if started_ns is None:
                started_ns = time.time_ns() - int(playback_position * 1_000_000_000)
            self._audio.add_played(started_ns, AudioClip(pcm[:played], rate, channels))
        except Exception:
            logger.exception("Failed to record LiveKit output audio")


def attach_audio_taps(session: "AgentSession[Any]", audio: SessionAudio) -> None:
    if session.input.audio is not None:
        session.input.audio = _TapAudioInput(session.input.audio, audio)
    if session.output.audio is not None:
        session.output.audio = _TapAudioOutput(session.output.audio, audio)

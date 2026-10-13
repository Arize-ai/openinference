from __future__ import annotations

import base64
import io
import threading
import wave
from collections import deque
from dataclasses import dataclass

_SAMPLE_WIDTH_BYTES = 2
# The input buffer keeps this much audio. A user turn longer than this loses its start.
_INPUT_RETAIN_NS = 120 * 1_000_000_000
# Frames arrive after VAD has already started the user turn.
_INPUT_LEAD_NS = 300_000_000


@dataclass(frozen=True)
class AudioClip:
    pcm: bytes
    sample_rate: int
    num_channels: int

    def wav_data_uri(self, max_length: int) -> str | None:
        """A playable WAV data URI, trimmed by whole frames to fit max_length characters."""
        prefix = "data:audio/wav;base64,"
        frame_bytes = _SAMPLE_WIDTH_BYTES * self.num_channels
        max_wav_bytes = (max_length - len(prefix)) // 4 * 3
        max_frames = (max_wav_bytes - 44) // frame_bytes
        if max_frames <= 0:
            return None
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(self.num_channels)
            wav.setsampwidth(_SAMPLE_WIDTH_BYTES)
            wav.setframerate(self.sample_rate)
            wav.writeframes(self.pcm[: max_frames * frame_bytes])
        return prefix + base64.b64encode(buffer.getvalue()).decode("ascii")


@dataclass(frozen=True)
class _Frame:
    arrived_ns: int
    pcm: bytes
    sample_rate: int
    num_channels: int


@dataclass(frozen=True)
class _Played:
    started_ns: int
    clip: AudioClip


class SessionAudio:
    """What one AgentSession heard and what it actually played, in wall-clock time."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._input: deque[_Frame] = deque()
        self._played: list[_Played] = []

    def add_input(self, arrived_ns: int, pcm: bytes, sample_rate: int, num_channels: int) -> None:
        with self._lock:
            self._input.append(_Frame(arrived_ns, pcm, sample_rate, num_channels))
            while self._input and self._input[0].arrived_ns < arrived_ns - _INPUT_RETAIN_NS:
                self._input.popleft()

    def add_played(self, started_ns: int, clip: AudioClip) -> None:
        if not clip.pcm:
            return
        with self._lock:
            self._played.append(_Played(started_ns, clip))

    def heard(self, start_ns: int, end_ns: int) -> AudioClip | None:
        with self._lock:
            frames = [
                frame
                for frame in self._input
                if start_ns - _INPUT_LEAD_NS <= frame.arrived_ns <= end_ns
            ]
        return _join([(frame.pcm, frame.sample_rate, frame.num_channels) for frame in frames])

    def spoken(self, start_ns: int, end_ns: int) -> AudioClip | None:
        """Join the segments that started playing in the window, and forget them."""
        with self._lock:
            segments = [seg for seg in self._played if start_ns <= seg.started_ns <= end_ns]
            self._played = [seg for seg in self._played if seg.started_ns > end_ns]
        return _join(
            [(seg.clip.pcm, seg.clip.sample_rate, seg.clip.num_channels) for seg in segments]
        )


def _join(chunks: list[tuple[bytes, int, int]]) -> AudioClip | None:
    if not chunks:
        return None
    _, sample_rate, num_channels = chunks[0]
    pcm = b"".join(
        data for data, rate, channels in chunks if (rate, channels) == (sample_rate, num_channels)
    )
    return AudioClip(pcm, sample_rate, num_channels) if pcm else None

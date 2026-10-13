import base64
import io
import wave

from openinference.instrumentation.livekit_agents._audio import AudioClip, SessionAudio

_SECOND = 1_000_000_000


def _frames(url: str) -> tuple[int, bytes]:
    with wave.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1], validate=True))) as wav:
        return wav.getframerate(), wav.readframes(wav.getnframes())


def test_trimmed_clip_is_still_a_valid_wav_within_the_limit() -> None:
    clip = AudioClip(b"\x01\x00" * 24_000, 24_000, 1)
    url = clip.wav_data_uri(10_000)
    assert url is not None and len(url) <= 10_000
    rate, pcm = _frames(url)
    assert rate == 24_000
    assert 0 < len(pcm) < len(clip.pcm) and pcm == clip.pcm[: len(pcm)]
    assert AudioClip(b"\x01\x00", 24_000, 1).wav_data_uri(40) is None


def test_heard_keeps_frames_that_arrived_in_the_user_window() -> None:
    audio = SessionAudio()
    for second, sample in ((1, b"\x01\x00"), (5, b"\x02\x00"), (9, b"\x03\x00")):
        audio.add_input(second * _SECOND, sample * 10, 16_000, 1)
    clip = audio.heard(4 * _SECOND, 6 * _SECOND)
    assert clip == AudioClip(b"\x02\x00" * 10, 16_000, 1)
    assert audio.heard(20 * _SECOND, 30 * _SECOND) is None


def test_spoken_joins_segments_in_the_turn_and_forgets_them() -> None:
    audio = SessionAudio()
    audio.add_played(2 * _SECOND, AudioClip(b"\x01\x00", 24_000, 1))
    audio.add_played(3 * _SECOND, AudioClip(b"\x02\x00", 24_000, 1))
    audio.add_played(9 * _SECOND, AudioClip(b"\x03\x00", 24_000, 1))
    assert audio.spoken(1 * _SECOND, 4 * _SECOND) == AudioClip(b"\x01\x00\x02\x00", 24_000, 1)
    assert audio.spoken(1 * _SECOND, 4 * _SECOND) is None
    assert audio.spoken(8 * _SECOND, 10 * _SECOND) == AudioClip(b"\x03\x00", 24_000, 1)

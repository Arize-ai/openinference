# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "openai>=1.60.0",
#     "openai-agents>=0.18.0,<0.19",
#     "openinference-instrumentation-openai-agents==1.6.1",
#     "opentelemetry-sdk>=1.42.0",
#     "opentelemetry-exporter-otlp-proto-http>=1.42.0",
# ]
# ///
"""Audio path: a real instrumented OpenAI Agents SDK realtime session with blob upload.

Drives an actual voice turn end to end: a spoken question (synthesized with
OpenAI TTS) is sent into a live Realtime API session through the openai-agents
SDK, instrumented by openinference-instrumentation-openai-agents. The
instrumentor produces the realtime span tree — an AUDIO ``conversation.turn``
root with a USER child (``input.audio.*``) and an LLM child (``output.audio.*``).

By default the demo patches the instrumentor's audio capture site
(``pcm16_to_wav_data_uri`` in ``_realtime.py``) the way techspec §2.3 proposes:
audio that exceeds the inline budget is handed to the ``BlobUploader`` and the
span records only the URI. Run with ``--inline`` to skip the patch and see
today's released behavior (data URIs truncated at 32,000 chars) instead.

Prerequisites: OPENAI_API_KEY (with Realtime API access); a local
``phoenix serve`` (http://localhost:6006).
Run:  uv run --script internal_docs/specs/blob_uploads/scripts/audio_blob_demo.py
"""

from __future__ import annotations

import asyncio
import base64
import os
import sys
from typing import Any

from agents.realtime import RealtimeAgent, RealtimeRunner
from agents.realtime.events import RealtimeAudio, RealtimeAudioEnd
from openai import OpenAI
from openinference.instrumentation.openai_agents import (
    OpenAIAgentsInstrumentor,
    _realtime,
)
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from common import Blob, LocalBlobStore

PROJECT_NAME = "blob-upload-audio-demo"
QUESTION = "What's the weather like in Paris right now?"
INLINE_BUDGET = 32_000  # OPENINFERENCE_BASE64_AUDIO_MAX_LENGTH default
SILENCE = b"\x00" * (
    24_000 * 2 * 2
)  # 2 s of 24 kHz PCM16 silence so server VAD ends the turn

store = LocalBlobStore()


def patch_capture_site() -> None:
    """Apply techspec §2.3 to the released instrumentor.

    ``_finalize_user`` / ``_finalize_response`` call ``pcm16_to_wav_data_uri``
    with the turn's complete audio and truncate the result over the budget.
    The patched version uploads over-budget audio instead and returns the blob
    URI; a ``None`` from the uploader falls through to today's truncation.
    In the real package this is a change at those two capture sites, not a patch.
    """
    original_encode = _realtime.pcm16_to_wav_data_uri

    def encode_or_upload(pcm_bytes: bytes, *args: Any, **kwargs: Any) -> str:
        data_uri = original_encode(pcm_bytes, *args, **kwargs)
        if len(data_uri) <= INLINE_BUDGET:  # fits inline — no offload
            return data_uri
        wav_bytes = base64.b64decode(data_uri.split(";base64,", 1)[1])
        uri = store.upload(Blob(data=wav_bytes, mime_type="audio/wav"))
        return uri if uri is not None else data_uri

    _realtime.pcm16_to_wav_data_uri = encode_or_upload


def tts_pcm(text: str) -> bytes:
    """Synthesize the user's spoken question: raw 24 kHz mono PCM16 —
    exactly the format a microphone feeds a realtime session."""
    response = OpenAI().audio.speech.create(
        model=os.environ.get("OPENAI_TTS_MODEL", "gpt-4o-mini-tts"),
        voice="alloy",
        input=text,
        response_format="pcm",
    )
    return response.read()


async def drive_voice_turn(question_pcm: bytes) -> None:
    """One real voice turn: send the question audio, wait for the spoken answer."""
    agent = RealtimeAgent(
        name="Assistant",
        instructions="You are a helpful voice assistant. Answer in one short sentence.",
    )
    runner = RealtimeRunner(
        agent,
        config={
            "model_settings": {
                "model_name": os.environ.get("OPENAI_REALTIME_MODEL", "gpt-realtime"),
                "voice": "alloy",
                "input_audio_transcription": {"model": "whisper-1"},
            }
        },
    )
    session = await runner.run()
    async with session:
        print("  connected — sending the question audio", flush=True)
        await session.send_audio(question_pcm + SILENCE)
        events = session.__aiter__()
        received = 0
        answer_done = False
        while True:
            try:
                event = await asyncio.wait_for(events.__anext__(), timeout=10.0)
            except (asyncio.TimeoutError, StopAsyncIteration):
                if answer_done or received:
                    break  # events went quiet after the answer — turn is over
                print("  waiting for the model to start speaking ...", flush=True)
                continue
            if isinstance(event, RealtimeAudio):
                if not received:
                    print("  assistant is speaking ...", flush=True)
                received += len(event.audio.data)
            elif isinstance(event, RealtimeAudioEnd):
                answer_done = True
                print(
                    f"  assistant answered with {received:,} bytes of audio", flush=True
                )
            elif type(event).__name__ == "RealtimeError":
                print(f"  [realtime error] {event}", file=sys.stderr)
                break


def print_span(span: Any) -> None:
    attributes = span.attributes or {}
    total = sum(len(k) + len(str(v)) for k, v in attributes.items())
    print(f"\n── {span.name}  ({len(attributes)} attrs, {total:,} B) ──")
    for key in sorted(attributes):
        text = str(attributes[key]).replace("\n", "\\n")
        if len(text) > 76:
            text = f"{text[:76]}… ({len(text):,} chars)"
        print(f"  {key} = {text}")


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit(
            "OPENAI_API_KEY is not set — this demo runs a live Realtime API session."
        )

    inline = "--inline" in sys.argv
    if inline:
        print("running with today's released behavior (inline truncated data URIs)")
    else:
        patch_capture_site()
        print("capture site patched: over-budget audio goes to the blob store")

    print(f"synthesizing the spoken question with TTS: {QUESTION!r}")
    question_pcm = tts_pcm(QUESTION)
    print(f"  {len(question_pcm):,} bytes of 24 kHz PCM16")

    phoenix = os.environ.get(
        "PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006"
    ).rstrip("/")
    provider = TracerProvider(
        resource=Resource.create({"openinference.project.name": PROJECT_NAME})
    )
    memory = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(memory))
    provider.add_span_processor(
        SimpleSpanProcessor(OTLPSpanExporter(f"{phoenix}/v1/traces"))
    )
    OpenAIAgentsInstrumentor().instrument(tracer_provider=provider)

    print("connecting to the Realtime API ...", flush=True)
    deadline_sec = float(os.environ.get("REALTIME_DEMO_TIMEOUT_SEC", "120"))
    try:
        asyncio.run(
            asyncio.wait_for(drive_voice_turn(question_pcm), timeout=deadline_sec)
        )
    except asyncio.TimeoutError:
        print(
            f"realtime session did not complete within {deadline_sec:.0f}s — "
            "exporting whatever spans were captured",
            file=sys.stderr,
        )

    OpenAIAgentsInstrumentor().uninstrument()
    provider.force_flush()

    for span in memory.get_finished_spans():
        print_span(span)

    print(f"\nPhoenix: {phoenix}  → project {PROJECT_NAME!r}")
    print("The user span's input.audio.url and the assistant span's output.audio.url")
    if inline:
        print("hold data URIs truncated at 32,000 chars — today's released behavior.")
    else:
        print(f"reference WAV files under {store.root_dir} (URI = repo-relative path).")
        print("Re-run with --inline to compare against today's truncated data URIs.")
    provider.shutdown()


if __name__ == "__main__":
    main()

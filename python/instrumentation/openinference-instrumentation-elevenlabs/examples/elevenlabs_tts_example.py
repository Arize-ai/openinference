"""
ElevenLabs Text-to-Speech Instrumentation Example

Demonstrates how to use OpenInference instrumentation with ElevenLabs TTS API,
sending traces to Arize.

Setup:
    pip install elevenlabs arize-otel openinference-instrumentation-elevenlabs python-dotenv

    Copy .env.example to .env and fill in your keys.
"""

import asyncio
import base64
import os

from arize.otel import register
from dotenv import load_dotenv
from elevenlabs import AsyncElevenLabs, ElevenLabs

from openinference.instrumentation.elevenlabs import ElevenLabsInstrumentor


async def async_tts_example(async_client: AsyncElevenLabs) -> None:
    audio = async_client.text_to_speech.convert(
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        text="This is an async text to speech call.",
        model_id="eleven_multilingual_v2",
    )
    audio_bytes = b"".join([chunk async for chunk in audio])
    print(f"Async generated {len(audio_bytes)} bytes of audio")


def main() -> None:
    load_dotenv()

    # --- Configure Arize OTel ---
    tracer_provider = register(
        space_id=os.environ["ARIZE_SPACE_ID"],
        api_key=os.environ["ARIZE_API_KEY"],
        project_name=os.environ.get("ARIZE_PROJECT", "elevenlabs-demo"),
    )

    # --- Instrument ElevenLabs ---
    ElevenLabsInstrumentor().instrument(tracer_provider=tracer_provider)

    client = ElevenLabs(api_key=os.environ["ELEVEN_API_KEY"])

    # --- Text-to-Speech: Convert (Sync) ---
    audio = client.text_to_speech.convert(
        voice_id="JBFqnCBsd6RMkjVDRZzb",  # George voice
        text="Hello! This is a test of ElevenLabs text to speech.",
        model_id="eleven_multilingual_v2",
    )
    audio_bytes = b"".join(audio)
    print(f"Generated {len(audio_bytes)} bytes of audio")

    # --- Text-to-Speech: Stream (Sync) ---
    audio_stream = client.text_to_speech.stream(
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        text="This demonstrates streaming audio generation.",
        model_id="eleven_multilingual_v2",
    )
    chunks = []
    for chunk in audio_stream:
        chunks.append(chunk)
    print(f"Received {len(chunks)} audio chunks")

    # --- Text-to-Speech: Async Convert ---
    async_client = AsyncElevenLabs(api_key=os.environ["ELEVEN_API_KEY"])
    asyncio.run(async_tts_example(async_client))

    # --- Text-to-Speech with Timestamps ---
    response = client.text_to_speech.convert_with_timestamps(
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        text="Hello world!",
        model_id="eleven_multilingual_v2",
    )
    if response.audio_base_64:
        audio_bytes = base64.b64decode(response.audio_base_64)
        print(f"Audio with timestamps: {len(audio_bytes)} bytes")
    if response.alignment:
        print(f"Characters: {response.alignment.characters}")
        print(f"Start times: {response.alignment.character_start_times_seconds}")
        print(f"End times: {response.alignment.character_end_times_seconds}")

    # --- Cleanup ---
    ElevenLabsInstrumentor().uninstrument()


if __name__ == "__main__":
    main()

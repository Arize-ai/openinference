# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "opentelemetry-api==1.42.1",
#     "opentelemetry-sdk==1.42.1",
#     "opentelemetry-exporter-otlp-proto-http==1.42.1",
#     "openinference-semantic-conventions==0.1.29",
# ]
# ///

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    AUDIO_URL,
    content_key,
    forbid_keys,
    media_part_from_url,
    pass_or_fail,
    read_content_url,
    require_keys,
    set_audio_content_block,
    set_text_content_block,
    setup_demo_tracing,
    shutdown,
    span_attrs_by_name,
    start_demo_span,
)
from openinference.semconv.trace import MessageAttributes, SpanAttributes

AUDIO_B64 = "UklGRiQAAABXQVZFZm10IBAAAAABAAEA"
DATA_URI = f"data:audio/wav;base64,{AUDIO_B64}"
PROMPT = "What is in this recording?"


def main() -> None:
    ctx = setup_demo_tracing("oi-audio-video-openai-chat")
    try:
        current: dict[str, object] = {}
        current[
            f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}"
        ] = "user"
        set_text_content_block(
            current, side="input", message_index=0, content_index=0, text=PROMPT
        )
        start_demo_span(ctx, "current", current, span_kind="LLM")

        future = dict(current)
        set_audio_content_block(
            future,
            side="input",
            message_index=0,
            content_index=1,
            url=DATA_URI,
        )
        start_demo_span(ctx, "future", future, span_kind="LLM")

        current_attrs = span_attrs_by_name(ctx, "current")
        future_attrs = span_attrs_by_name(ctx, "future")
        audio_url_key = content_key("input", 0, 1, f"message_content.audio.{AUDIO_URL}")
        audio_mime_key = content_key("input", 0, 1, "message_content.audio.audio.mime_type")
        forbid_keys(current_attrs, [audio_url_key])
        require_keys(future_attrs, [audio_url_key])
        forbid_keys(future_attrs, [audio_mime_key])
        url = read_content_url(
            future_attrs, side="input", message_index=0, content_index=1, media="audio"
        )
        part = media_part_from_url(url, "audio")
        if part != {
            "type": "blob",
            "modality": "audio",
            "mime_type": "audio/wav",
            "content": AUDIO_B64,
        }:
            raise AssertionError(f"unexpected GenAI part: {part}")
        pass_or_fail(True, "openai chat audio future keys and GenAI blob part")
    finally:
        shutdown(ctx)


if __name__ == "__main__":
    main()

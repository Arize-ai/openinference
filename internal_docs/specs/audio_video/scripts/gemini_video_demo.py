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

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    VIDEO_URL,
    content_key,
    forbid_keys,
    media_part_from_url,
    pass_or_fail,
    read_content_url,
    require_keys,
    set_text_content_block,
    set_video_content_block,
    setup_demo_tracing,
    shutdown,
    span_attrs_by_name,
    start_demo_span,
)
from openinference.semconv.trace import MessageAttributes, SpanAttributes

VIDEO_URI = "gs://cloud-samples-data/generative-ai/video/animals.mp4"
PROMPT = "Describe what happens in this video."


def main() -> None:
    ctx = setup_demo_tracing("oi-audio-video-gemini")
    try:
        current: dict[str, object] = {
            SpanAttributes.INPUT_VALUE: json.dumps(
                {
                    "parts": [
                        {"file_uri": VIDEO_URI, "mime_type": "video/mp4"},
                        {"text": PROMPT},
                    ]
                }
            ),
            SpanAttributes.INPUT_MIME_TYPE: "application/json",
        }
        current[
            f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}"
        ] = "user"
        set_text_content_block(
            current, side="input", message_index=0, content_index=0, text=PROMPT
        )
        start_demo_span(ctx, "current", current, span_kind="LLM")

        future = dict(current)
        set_video_content_block(
            future,
            side="input",
            message_index=0,
            content_index=1,
            url=VIDEO_URI,
        )
        start_demo_span(ctx, "future", future, span_kind="LLM")

        current_attrs = span_attrs_by_name(ctx, "current")
        future_attrs = span_attrs_by_name(ctx, "future")
        video_url_key = content_key("input", 0, 1, f"message_content.video.{VIDEO_URL}")
        forbid_keys(current_attrs, [video_url_key])
        require_keys(future_attrs, [video_url_key])
        url = read_content_url(
            future_attrs, side="input", message_index=0, content_index=1, media="video"
        )
        part = media_part_from_url(url, "video", mime_type="video/mp4")
        if part != {
            "type": "uri",
            "modality": "video",
            "uri": VIDEO_URI,
            "mime_type": "video/mp4",
        }:
            raise AssertionError(f"unexpected GenAI part: {part}")
        pass_or_fail(True, "gemini video future keys and GenAI uri part")
    finally:
        shutdown(ctx)


if __name__ == "__main__":
    main()

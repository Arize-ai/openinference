# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "opentelemetry-api==1.42.1",
#     "opentelemetry-sdk==1.42.1",
#     "opentelemetry-exporter-otlp-proto-http==1.42.1",
#     "openinference-semantic-conventions==0.1.29",
# ]
# ///
"""Publish openai-agents realtime span-root audio keys as future semconv.

Current and future use the same attribute names. The spec change is
publication, not a new shape. The future span must not grow llm.input_messages.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    INPUT_AUDIO_MIME_TYPE,
    INPUT_AUDIO_TRANSCRIPT,
    INPUT_AUDIO_URL,
    OUTPUT_AUDIO_MIME_TYPE,
    OUTPUT_AUDIO_TRANSCRIPT,
    OUTPUT_AUDIO_URL,
    forbid_keys,
    pass_or_fail,
    require_keys,
    set_span_root_audio,
    setup_demo_tracing,
    shutdown,
    span_attrs_by_name,
    start_demo_span,
)
from openinference.semconv.trace import SpanAttributes

AUDIO_B64 = "UklGRiQAAABXQVZFZm10IBAAAAABAAEA"
DATA_URI = f"data:audio/wav;base64,{AUDIO_B64}"


def main() -> None:
    ctx = setup_demo_tracing("oi-audio-video-openai-realtime")
    try:
        user_attrs: dict[str, object] = {
            SpanAttributes.INPUT_VALUE: "What's the weather in Tokyo?",
        }
        set_span_root_audio(
            user_attrs,
            side="input",
            url=DATA_URI,
            mime_type="audio/wav",
            transcript="What's the weather in Tokyo?",
        )
        start_demo_span(ctx, "current", user_attrs, span_kind="USER")
        start_demo_span(ctx, "future", user_attrs, span_kind="USER")

        llm_attrs: dict[str, object] = {}
        set_span_root_audio(
            llm_attrs,
            side="output",
            url=DATA_URI,
            mime_type="audio/wav",
            transcript="It's sunny in Tokyo.",
        )
        start_demo_span(ctx, "current-llm", llm_attrs, span_kind="LLM")
        start_demo_span(ctx, "future-llm", llm_attrs, span_kind="LLM")

        future_user = span_attrs_by_name(ctx, "future")
        future_llm = span_attrs_by_name(ctx, "future-llm")
        require_keys(
            future_user,
            [INPUT_AUDIO_URL, INPUT_AUDIO_MIME_TYPE, INPUT_AUDIO_TRANSCRIPT],
        )
        require_keys(
            future_llm,
            [OUTPUT_AUDIO_URL, OUTPUT_AUDIO_MIME_TYPE, OUTPUT_AUDIO_TRANSCRIPT],
        )
        llm_input_prefix = f"{SpanAttributes.LLM_INPUT_MESSAGES}."
        forbid_keys(
            future_user,
            [key for key in future_user if key.startswith(llm_input_prefix)],
        )
        pass_or_fail(True, "realtime span-root audio keys without input_messages")
    finally:
        shutdown(ctx)


if __name__ == "__main__":
    main()

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "opentelemetry-api==1.42.1",
#     "opentelemetry-sdk==1.42.1",
#     "opentelemetry-exporter-otlp-proto-http==1.42.1",
#     "openinference-semantic-conventions==0.1.30",
# ]
# ///
"""Audio path: openai-agents realtime turn, before (inline base64) vs after (blob upload).

Emits two realtime conversation turns shaped exactly like the spans
openinference-instrumentation-openai-agents produces for a voice turn
(PR #3173): an AUDIO ``conversation.turn`` root with a USER child carrying
``input.audio.*`` and an LLM child carrying ``output.audio.*``.

  before  ``input.audio.url`` / ``output.audio.url`` hold ``data:audio/wav;base64,...``
          truncated at OPENINFERENCE_BASE64_AUDIO_MAX_LENGTH (default 32,000 chars
          ≈ 0.5 s of 24 kHz PCM16) — the released behavior today.
  after   the finalized WAV bytes go to the proposed BlobUploader at the same
          capture sites (_finalize_user / _finalize_response) and the attributes
          record only the destination URI. Content survives in full.

The offload policy (techspec §3.4) is implemented faithfully, and the PASS
checks adapt so the policy matrix can be explored via env vars:

  OPENINFERENCE_HIDE_INPUT_AUDIO=1        privacy wins — no attribute, no upload
  OPENINFERENCE_HIDE_OUTPUT_AUDIO=1       same, for the assistant side
  OPENINFERENCE_BASE64_AUDIO_MAX_LENGTH=10000000
                                          fits the inline budget — stays inline,
                                          no upload even in the "after" variant

Run:  uv run --script internal_docs/specs/blob_uploads/scripts/audio_blob_demo.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Optional

from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from opentelemetry import trace as trace_api

from common import (
    DEFAULT_BASE64_AUDIO_MAX_LENGTH,
    Blob,
    Checker,
    LocalHttpBlobUploader,
    attributes_size_bytes,
    fetch_url,
    format_bytes,
    maybe_wait_for_browsing,
    pcm16_to_wav_bytes,
    pcm16_to_wav_data_uri,
    print_size_table,
    setup_tracing,
    synth_pcm16,
    truncate_audio_data_uri,
)

PROJECT_NAME = "blob-upload-audio-demo"

# Instrumentor-local constants from openai-agents _realtime.py (PR #3173).
INPUT_AUDIO_URL = "input.audio.url"
INPUT_AUDIO_MIME_TYPE = "input.audio.mime_type"
INPUT_AUDIO_TRANSCRIPT = "input.audio.transcript"
OUTPUT_AUDIO_URL = "output.audio.url"
OUTPUT_AUDIO_MIME_TYPE = "output.audio.mime_type"
OUTPUT_AUDIO_TRANSCRIPT = "output.audio.transcript"
AUDIO_KIND = "AUDIO"
USER_KIND = "USER"
END_REASON = "end_reason"
TIME_TO_FIRST_TOKEN_MS = "time_to_first_token_ms"

SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
LLM_KIND = OpenInferenceSpanKindValues.LLM.value

MODEL_NAME = "gpt-realtime"
SESSION_CONFIG = {
    "voice": "alloy",
    "modalities": ["audio", "text"],
    "input_audio_format": "pcm16",
    "output_audio_format": "pcm16",
    "input_audio_transcription": {"model": "whisper-1"},
}
USER_TRANSCRIPT = "What's the weather like in Paris right now?"
ASSISTANT_TRANSCRIPT = (
    "It's currently 15 degrees and partly cloudy in Paris this afternoon."
)

# A tone ladder (user) and a descending melody (assistant) stand in for mic /
# assistant PCM16. Realistic voice-turn durations: hundreds of KB per side.
USER_PCM = synth_pcm16([(392, 0.8), (440, 0.8), (494, 0.8), (523, 0.8)])  # 3.2 s
ASSISTANT_PCM = synth_pcm16([(659, 1.2), (587, 1.2), (523, 1.2), (494, 1.2)])  # 4.8 s

NS = 1_000_000_000


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def audio_max_length() -> int:
    raw = os.environ.get("OPENINFERENCE_BASE64_AUDIO_MAX_LENGTH")
    try:
        return int(raw) if raw is not None else DEFAULT_BASE64_AUDIO_MAX_LENGTH
    except ValueError:
        return DEFAULT_BASE64_AUDIO_MAX_LENGTH


def inline_audio_url(pcm: bytes) -> str:
    """The released behavior: data URI, truncated over the max length."""
    uri = pcm16_to_wav_data_uri(pcm)
    max_len = audio_max_length()
    if len(uri) > max_len:
        uri = truncate_audio_data_uri(uri, max_len)
    return uri


def blob_audio_url(
    uploader: LocalHttpBlobUploader,
    pcm: bytes,
    attribute_key: str,
    span: trace_api.Span,
) -> str:
    """The proposed behavior at the same capture site (techspec §3.3 + §3.4).

    Content that fits the inline budget stays inline; oversized content is
    uploaded and referenced by URI. A ``None`` return from ``upload``
    (backpressure/shutdown) falls back to today's truncation — enabling blob
    upload can never capture less than today.
    """
    inline = pcm16_to_wav_data_uri(pcm)
    if len(inline) <= audio_max_length():
        return inline
    span_ctx = span.get_span_context()
    uri = uploader.upload(
        Blob(
            data=pcm16_to_wav_bytes(pcm),
            mime_type="audio/wav",
            attribute_key=attribute_key,
            trace_id=f"{span_ctx.trace_id:032x}",
            span_id=f"{span_ctx.span_id:016x}",
        )
    )
    return (
        uri if uri is not None else truncate_audio_data_uri(inline, audio_max_length())
    )


def emit_turn(
    tracer: trace_api.Tracer,
    variant: str,
    uploader: Optional[LocalHttpBlobUploader],
) -> None:
    """One realtime voice turn in the PR #3173 wire form.

    ``uploader=None`` reproduces today's inline truncation; otherwise audio is
    offloaded at the two capture sites (_finalize_user / _finalize_response).
    Hide flags win over everything: no attribute, no upload (techspec §3.4).
    """
    hide_in = env_flag("OPENINFERENCE_HIDE_INPUT_AUDIO")
    hide_out = env_flag("OPENINFERENCE_HIDE_OUTPUT_AUDIO")
    base_ns = time.time_ns()
    turn_span = tracer.start_span(
        name=f"conversation.turn — {variant}",
        start_time=base_ns,
        attributes={
            SPAN_KIND: AUDIO_KIND,
            SpanAttributes.LLM_MODEL_NAME: MODEL_NAME,
            SpanAttributes.LLM_INVOCATION_PARAMETERS: json.dumps(SESSION_CONFIG),
        },
    )
    turn_ctx = trace_api.set_span_in_context(turn_span)

    # USER child — mic audio finalized when the transcript completes (_finalize_user).
    user_span = tracer.start_span(name="user", context=turn_ctx, start_time=base_ns)
    user_span.set_attribute(SPAN_KIND, USER_KIND)
    if not hide_in:
        if uploader is None:
            url = inline_audio_url(USER_PCM)
        else:
            url = blob_audio_url(uploader, USER_PCM, INPUT_AUDIO_URL, user_span)
        user_span.set_attribute(INPUT_AUDIO_URL, url)
        user_span.set_attribute(INPUT_AUDIO_MIME_TYPE, "audio/wav")
        user_span.set_attribute(INPUT_AUDIO_TRANSCRIPT, USER_TRANSCRIPT)
    user_span.end(end_time=base_ns + int(3.4 * NS))

    # LLM child — assistant audio finalized on response.done (_finalize_response).
    llm_span = tracer.start_span(
        name="assistant", context=turn_ctx, start_time=base_ns + int(3.6 * NS)
    )
    llm_span.set_attribute(SPAN_KIND, LLM_KIND)
    llm_span.set_attribute(SpanAttributes.LLM_SYSTEM, "openai")
    llm_span.set_attribute(SpanAttributes.LLM_MODEL_NAME, MODEL_NAME)
    llm_span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, 142)
    llm_span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, 96)
    llm_span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_TOTAL, 238)
    llm_span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_AUDIO, 118)
    llm_span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_AUDIO, 84)
    llm_span.set_attribute(TIME_TO_FIRST_TOKEN_MS, 412)
    if not hide_out:
        if uploader is None:
            out_url = inline_audio_url(ASSISTANT_PCM)
        else:
            out_url = blob_audio_url(
                uploader, ASSISTANT_PCM, OUTPUT_AUDIO_URL, llm_span
            )
        llm_span.set_attribute(OUTPUT_AUDIO_URL, out_url)
        llm_span.set_attribute(OUTPUT_AUDIO_MIME_TYPE, "audio/wav")
        llm_span.set_attribute(OUTPUT_AUDIO_TRANSCRIPT, ASSISTANT_TRANSCRIPT)
    llm_span.end(end_time=base_ns + int(8.8 * NS))

    # AUDIO parent aggregates transcripts as input.value / output.value,
    # gated like _set_turn_io_attributes.
    if not hide_in:
        turn_span.set_attribute(SpanAttributes.INPUT_VALUE, USER_TRANSCRIPT)
    if not hide_out:
        turn_span.set_attribute(SpanAttributes.OUTPUT_VALUE, ASSISTANT_TRANSCRIPT)
    turn_span.set_attribute(END_REASON, "complete")
    turn_span.end(end_time=base_ns + int(9.0 * NS))


def main() -> int:
    checker = Checker()
    ctx = setup_tracing(PROJECT_NAME)
    tracer = ctx.provider.get_tracer("blob-upload-audio-demo")
    uploader = LocalHttpBlobUploader()

    hide_in = env_flag("OPENINFERENCE_HIDE_INPUT_AUDIO")
    hide_out = env_flag("OPENINFERENCE_HIDE_OUTPUT_AUDIO")
    max_len = audio_max_length()
    user_wav = pcm16_to_wav_bytes(USER_PCM)
    assistant_wav = pcm16_to_wav_bytes(ASSISTANT_PCM)
    user_full_uri = pcm16_to_wav_data_uri(USER_PCM)
    assistant_full_uri = pcm16_to_wav_data_uri(ASSISTANT_PCM)
    prefix = "data:audio/wav;base64,"
    print(
        f"\nsynthesized audio: user {format_bytes(len(user_wav))} WAV, "
        f"assistant {format_bytes(len(assistant_wav))} WAV "
        f"(24 kHz mono PCM16, like the OpenAI Realtime API)"
    )

    emit_turn(tracer, "before (inline base64)", uploader=None)
    emit_turn(tracer, "after (blob upload)", uploader=uploader)

    checker.check(
        "blob uploads drained", uploader.force_flush(), "background queue empty"
    )
    ctx.provider.force_flush()

    spans = {s.name: s for s in ctx.finished_spans()}
    before_turn = spans["conversation.turn — before (inline base64)"]
    after_turn = spans["conversation.turn — after (blob upload)"]

    def child(turn: Any, name: str) -> Any:
        for s in ctx.finished_spans():
            if s.name == name and s.parent and s.parent.span_id == turn.context.span_id:
                return s
        raise RuntimeError(f"missing child span {name!r}")

    before_user, before_llm = (
        child(before_turn, "user"),
        child(before_turn, "assistant"),
    )
    after_user, after_llm = child(after_turn, "user"), child(after_turn, "assistant")

    # --- input audio (USER span): hide > inline-budget > offload ---
    if hide_in:
        checker.check(
            "hidden input audio: no attribute on either variant, no upload",
            INPUT_AUDIO_URL not in before_user.attributes
            and INPUT_AUDIO_URL not in after_user.attributes
            and INPUT_AUDIO_TRANSCRIPT not in after_user.attributes,
            "privacy wins over upload (techspec §3.4)",
        )
    elif len(user_full_uri) <= max_len:
        checker.check(
            "input audio fits the inline budget: stays inline on both variants",
            str(before_user.attributes[INPUT_AUDIO_URL]) == user_full_uri
            and str(after_user.attributes[INPUT_AUDIO_URL]) == user_full_uri,
            f"{len(user_full_uri)} chars <= budget {max_len} — no offload",
        )
    else:
        before_in_url = str(before_user.attributes[INPUT_AUDIO_URL])
        checker.check(
            "before: input audio is a truncated data URI",
            before_in_url.startswith(prefix)
            and len(before_in_url) == len(prefix) + max_len,
            f"payload capped at {max_len} chars",
        )
        full_b64_len = len(user_full_uri) - len(prefix)
        checker.check(
            "before: most of the user audio is lost",
            max_len < full_b64_len,
            f"kept {max_len}/{full_b64_len} base64 chars "
            f"(~{max_len * 3 / 4 / 48_000:.2f}s of a {len(USER_PCM) / 48_000:.1f}s utterance)",
        )
        after_in_url = str(after_user.attributes[INPUT_AUDIO_URL])
        checker.check(
            "after: input audio attribute is an external URI",
            after_in_url.startswith("http://"),
            after_in_url,
        )
        checker.check(
            "after: user audio round-trips byte-for-byte",
            fetch_url(after_in_url) == user_wav,
            f"GET {after_in_url} == original WAV",
        )
        checker.check(
            "after: transcript + mime type preserved on the span",
            after_user.attributes.get(INPUT_AUDIO_TRANSCRIPT) == USER_TRANSCRIPT
            and after_user.attributes.get(INPUT_AUDIO_MIME_TYPE) == "audio/wav",
        )

    # --- output audio (LLM span): same policy ---
    if hide_out:
        checker.check(
            "hidden output audio: no attribute on either variant, no upload",
            OUTPUT_AUDIO_URL not in before_llm.attributes
            and OUTPUT_AUDIO_URL not in after_llm.attributes,
            "privacy wins over upload (techspec §3.4)",
        )
    elif len(assistant_full_uri) <= max_len:
        checker.check(
            "output audio fits the inline budget: stays inline on both variants",
            str(before_llm.attributes[OUTPUT_AUDIO_URL]) == assistant_full_uri
            and str(after_llm.attributes[OUTPUT_AUDIO_URL]) == assistant_full_uri,
            f"{len(assistant_full_uri)} chars <= budget {max_len} — no offload",
        )
    else:
        after_out_url = str(after_llm.attributes[OUTPUT_AUDIO_URL])
        checker.check(
            "after: assistant audio round-trips byte-for-byte",
            after_out_url.startswith("http://")
            and fetch_url(after_out_url) == assistant_wav,
            f"GET {after_out_url} == original WAV",
        )

    print_size_table(
        [
            (
                "user (before)",
                attributes_size_bytes(before_user.attributes),
                "inline path",
            ),
            (
                "user (after)",
                attributes_size_bytes(after_user.attributes),
                "blob-upload path",
            ),
            (
                "assistant (before)",
                attributes_size_bytes(before_llm.attributes),
                "inline path",
            ),
            (
                "assistant (after)",
                attributes_size_bytes(after_llm.attributes),
                "blob-upload path",
            ),
        ]
    )
    if not hide_in and not hide_out:
        untruncated = len(user_full_uri) + len(assistant_full_uri)
        print(
            f"\n  (untruncated inline capture would put {format_bytes(untruncated)} "
            f"of base64 on this one turn)"
        )

    print(f"\nPhoenix: {ctx.phoenix_base_url}  → project {PROJECT_NAME!r}")
    print("Compare the two conversation.turn traces: user/assistant spans carry")
    print("input.audio.url / output.audio.url — truncated base64 vs blob URI.")

    ctx.shutdown()
    code = checker.exit_code()
    maybe_wait_for_browsing(uploader)
    return code


if __name__ == "__main__":
    sys.exit(main())

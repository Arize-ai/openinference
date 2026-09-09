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

import os
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from openinference.semconv.trace import (
    MessageAttributes,
    MessageContentAttributes,
    SpanAttributes,
)
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Status, StatusCode

CONTENT_TYPE_TEXT = "text"
CONTENT_TYPE_AUDIO = "audio"
CONTENT_TYPE_VIDEO = "video"

MESSAGE_CONTENT_AUDIO = "message_content.audio"
MESSAGE_CONTENT_VIDEO = "message_content.video"
AUDIO_URL = "audio.url"
AUDIO_MIME_TYPE = "audio.mime_type"
AUDIO_TRANSCRIPT = "audio.transcript"
VIDEO_URL = "video.url"

INPUT_AUDIO_URL = f"input.{AUDIO_URL}"
INPUT_AUDIO_MIME_TYPE = f"input.{AUDIO_MIME_TYPE}"
INPUT_AUDIO_TRANSCRIPT = f"input.{AUDIO_TRANSCRIPT}"
OUTPUT_AUDIO_URL = f"output.{AUDIO_URL}"
OUTPUT_AUDIO_MIME_TYPE = f"output.{AUDIO_MIME_TYPE}"
OUTPUT_AUDIO_TRANSCRIPT = f"output.{AUDIO_TRANSCRIPT}"

_DATA_URL_PATTERN = re.compile(
    r"^data:(?P<mime>[^;,]+);base64,(?P<content>.*)$",
    re.DOTALL,
)

Side = Literal["input", "output"]


def output_msg_key(message_index: int, suffix: str) -> str:
    return f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.{message_index}.{suffix}"


def input_msg_key(message_index: int, suffix: str) -> str:
    return f"{SpanAttributes.LLM_INPUT_MESSAGES}.{message_index}.{suffix}"


def output_content_key(message_index: int, content_index: int, suffix: str) -> str:
    return (
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.{message_index}"
        f".{MessageAttributes.MESSAGE_CONTENTS}.{content_index}.{suffix}"
    )


def input_content_key(message_index: int, content_index: int, suffix: str) -> str:
    return (
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.{message_index}"
        f".{MessageAttributes.MESSAGE_CONTENTS}.{content_index}.{suffix}"
    )


def content_key(side: Side, message_index: int, content_index: int, suffix: str) -> str:
    if side == "input":
        return input_content_key(message_index, content_index, suffix)
    return output_content_key(message_index, content_index, suffix)


def set_text_content_block(
    attrs: dict[str, Any],
    *,
    side: Side,
    message_index: int,
    content_index: int,
    text: str,
) -> None:
    attrs[
        content_key(
            side,
            message_index,
            content_index,
            MessageContentAttributes.MESSAGE_CONTENT_TYPE,
        )
    ] = CONTENT_TYPE_TEXT
    attrs[
        content_key(
            side,
            message_index,
            content_index,
            MessageContentAttributes.MESSAGE_CONTENT_TEXT,
        )
    ] = text


def set_audio_content_block(
    attrs: dict[str, Any],
    *,
    side: Side,
    message_index: int,
    content_index: int,
    url: str,
    mime_type: str | None = None,
    transcript: str | None = None,
) -> None:
    attrs[
        content_key(
            side,
            message_index,
            content_index,
            MessageContentAttributes.MESSAGE_CONTENT_TYPE,
        )
    ] = CONTENT_TYPE_AUDIO
    attrs[
        content_key(
            side, message_index, content_index, f"{MESSAGE_CONTENT_AUDIO}.{AUDIO_URL}"
        )
    ] = url
    if mime_type is not None:
        attrs[
            content_key(
                side,
                message_index,
                content_index,
                f"{MESSAGE_CONTENT_AUDIO}.{AUDIO_MIME_TYPE}",
            )
        ] = mime_type
    if transcript is not None:
        attrs[
            content_key(
                side,
                message_index,
                content_index,
                f"{MESSAGE_CONTENT_AUDIO}.{AUDIO_TRANSCRIPT}",
            )
        ] = transcript


def set_video_content_block(
    attrs: dict[str, Any],
    *,
    side: Side,
    message_index: int,
    content_index: int,
    url: str,
) -> None:
    attrs[
        content_key(
            side,
            message_index,
            content_index,
            MessageContentAttributes.MESSAGE_CONTENT_TYPE,
        )
    ] = CONTENT_TYPE_VIDEO
    attrs[
        content_key(
            side, message_index, content_index, f"{MESSAGE_CONTENT_VIDEO}.{VIDEO_URL}"
        )
    ] = url


def set_span_root_audio(
    attrs: dict[str, Any],
    *,
    side: Side,
    url: str,
    mime_type: str | None = None,
    transcript: str | None = None,
) -> None:
    prefix = "input" if side == "input" else "output"
    attrs[f"{prefix}.{AUDIO_URL}"] = url
    if mime_type is not None:
        attrs[f"{prefix}.{AUDIO_MIME_TYPE}"] = mime_type
    if transcript is not None:
        attrs[f"{prefix}.{AUDIO_TRANSCRIPT}"] = transcript


def media_part_from_url(
    url: str,
    modality: Literal["audio", "video", "image"],
    *,
    mime_type: str | None = None,
) -> dict[str, Any]:
    if data_url_match := _DATA_URL_PATTERN.match(url):
        part: dict[str, Any] = {
            "type": "blob",
            "modality": modality,
            "mime_type": data_url_match.group("mime"),
            "content": data_url_match.group("content"),
        }
        return part
    part = {
        "type": "uri",
        "modality": modality,
        "uri": url,
    }
    if mime_type is not None:
        part["mime_type"] = mime_type
    return part


def read_content_url(
    attrs: Mapping[str, Any],
    *,
    side: Side,
    message_index: int,
    content_index: int,
    media: Literal["audio", "video"],
) -> str:
    prefix = MESSAGE_CONTENT_AUDIO if media == "audio" else MESSAGE_CONTENT_VIDEO
    leaf = AUDIO_URL if media == "audio" else VIDEO_URL
    key = content_key(side, message_index, content_index, f"{prefix}.{leaf}")
    value = attrs.get(key)
    if not isinstance(value, str) or not value:
        raise AssertionError(f"missing {key}")
    return value


@dataclass
class DemoTracingCtx:
    tracer: trace.Tracer
    provider: TracerProvider
    memory_exporter: InMemorySpanExporter
    phoenix_exporter: OTLPSpanExporter | None
    project_name: str


def setup_demo_tracing(project_name: str) -> DemoTracingCtx:
    resource = Resource.create({"openinference.project.name": project_name})
    provider = TracerProvider(resource=resource)
    memory_exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(memory_exporter))
    phoenix_exporter: OTLPSpanExporter | None = None
    if os.environ.get("AUDIO_VIDEO_DEMO_SKIP_PHOENIX") != "1":
        base_url = os.environ.get(
            "PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006"
        ).rstrip("/")
        phoenix_exporter = OTLPSpanExporter(endpoint=f"{base_url}/v1/traces")
        provider.add_span_processor(SimpleSpanProcessor(phoenix_exporter))
    return DemoTracingCtx(
        tracer=provider.get_tracer("audio-video-semconv-demo"),
        provider=provider,
        memory_exporter=memory_exporter,
        phoenix_exporter=phoenix_exporter,
        project_name=project_name,
    )


def start_demo_span(
    ctx: DemoTracingCtx,
    name: str,
    attributes: Mapping[str, Any],
    *,
    span_kind: str,
) -> None:
    with ctx.tracer.start_as_current_span(name) as span:
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, span_kind)
        span.set_attributes(dict(attributes))
        span.set_status(Status(StatusCode.OK))


def require_keys(attrs: Mapping[str, Any], keys: list[str]) -> None:
    missing = [key for key in keys if key not in attrs]
    if missing:
        raise AssertionError(f"missing keys: {missing}")


def forbid_keys(attrs: Mapping[str, Any], keys: list[str]) -> None:
    present = [key for key in keys if key in attrs]
    if present:
        raise AssertionError(f"unexpected keys: {present}")


def span_attrs_by_name(ctx: DemoTracingCtx, name: str) -> dict[str, Any]:
    for span in ctx.memory_exporter.get_finished_spans():
        if span.name == name:
            return dict(span.attributes or {})
    raise AssertionError(f"no finished span named {name}")


def shutdown(ctx: DemoTracingCtx) -> None:
    ctx.provider.force_flush()
    if ctx.phoenix_exporter is not None:
        ctx.phoenix_exporter.shutdown()


def pass_or_fail(ok: bool, message: str) -> None:
    if ok:
        print(f"PASS {message}")
        return
    print(f"FAIL {message}", file=sys.stderr)
    raise SystemExit(1)

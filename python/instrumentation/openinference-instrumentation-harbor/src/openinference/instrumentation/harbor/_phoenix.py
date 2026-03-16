"""
Phoenix import utility: POST OTLP protobuf to Phoenix's /v1/traces endpoint.

Uses only stdlib (urllib.request) — no extra HTTP dependency needed.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Sequence
from urllib.request import Request, urlopen

from opentelemetry.sdk.trace import ReadableSpan

logger = logging.getLogger(__name__)

_PROTO_CONTENT_TYPE = "application/x-protobuf"


def phoenix_import_spans(
    spans: Sequence[ReadableSpan],
    *,
    endpoint: str = "http://localhost:6006",
) -> None:
    """POST spans directly to Phoenix as OTLP protobuf."""
    from openinference.instrumentation.harbor._file_exporter import _spans_to_proto_bytes

    url = f"{endpoint.rstrip('/')}/v1/traces"
    data = _spans_to_proto_bytes(spans)
    req = Request(url, data=data, headers={"Content-Type": _PROTO_CONTENT_TYPE})
    with urlopen(req) as resp:
        logger.info(f"Imported {len(spans)} spans -> {url} (HTTP {resp.status})")


def phoenix_import(
    source: str | Path,
    *,
    endpoint: str = "http://localhost:6006",
) -> None:
    """
    Import OTLP trace files into Phoenix.

    Accepts either .pb (protobuf) or .json files. JSON files are re-encoded
    to protobuf before sending, since Phoenix requires application/x-protobuf.
    """
    from google.protobuf.json_format import ParseDict  # type: ignore[import-untyped]
    from opentelemetry.proto.collector.trace.v1 import (  # type: ignore[import-untyped]
        trace_service_pb2,
    )

    ExportTraceServiceRequest = trace_service_pb2.ExportTraceServiceRequest

    source = Path(source)
    url = f"{endpoint.rstrip('/')}/v1/traces"
    if source.is_dir():
        files = sorted(source.glob("*.json")) + sorted(source.glob("*.pb"))
    else:
        files = [source]

    for filepath in files:
        if filepath.suffix == ".pb":
            data = filepath.read_bytes()
        else:
            with open(filepath) as f:
                json_data = json.load(f)
            proto = ParseDict(json_data, ExportTraceServiceRequest())
            data = proto.SerializeToString()

        req = Request(url, data=data, headers={"Content-Type": _PROTO_CONTENT_TYPE})
        with urlopen(req) as resp:
            logger.info(f"Imported {filepath.name} -> {url} (HTTP {resp.status})")

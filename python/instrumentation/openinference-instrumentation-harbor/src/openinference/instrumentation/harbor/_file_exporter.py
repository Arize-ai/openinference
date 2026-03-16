"""
OTLPJsonFileExporter: a SpanExporter that writes OTLP JSON to disk.

The JSON files use the OTLP protobuf-to-JSON mapping, suitable for
inspection and re-import into Phoenix via phoenix_import().
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Sequence

from google.protobuf.json_format import MessageToDict  # type: ignore[import-untyped]
from opentelemetry.exporter.otlp.proto.common._internal.trace_encoder import (
    encode_spans,
)
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

logger = logging.getLogger(__name__)


def _spans_to_otlp_dict(spans: Sequence[ReadableSpan]) -> dict[str, Any]:
    return MessageToDict(encode_spans(spans), preserving_proto_field_name=True)


def _spans_to_proto_bytes(spans: Sequence[ReadableSpan]) -> bytes:
    return encode_spans(spans).SerializeToString()


class OTLPJsonFileExporter(SpanExporter):
    """SpanExporter that writes OTLP-formatted JSON files to disk."""

    def __init__(self, output_dir: str = ".", *, file_prefix: str = "harbor_trace") -> None:
        self._output_dir = Path(output_dir)
        self._file_prefix = file_prefix
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        if not spans:
            return SpanExportResult.SUCCESS
        try:
            filename = f"{self._file_prefix}_{int(time.time() * 1000)}_{os.getpid()}.json"
            filepath = self._output_dir / filename
            with open(filepath, "w") as f:
                json.dump(_spans_to_otlp_dict(spans), f, indent=2)
            logger.debug(f"Exported {len(spans)} spans to {filepath}")
            return SpanExportResult.SUCCESS
        except Exception:
            logger.exception("Failed to export spans to file")
            return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def export_spans_to_file(spans: Sequence[ReadableSpan], output_path: str | Path) -> Path:
    """Export spans to a single OTLP JSON file (human-readable)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(_spans_to_otlp_dict(spans), f, indent=2)
    return output_path

"""
Experimental support for externalizing large binary media (audio, files,
images) captured during instrumentation.

Instead of recording multi-megabyte base64 data URIs in span attributes —
which exceed OTLP payload limits and inflate backend storage — a
``BlobUploader`` uploads the decoded bytes to external storage and the span
attribute records only the destination URI. This mirrors the OTel GenAI
semantic conventions message model, where inline base64 is a ``blob`` part
and an external reference is a ``uri`` part.

OpenInference ships only the interface and the offload policy — no
transport. Implementations are provided by applications, vendor SDKs, or a
future upstream (OTel util-genai) byte uploader, either programmatically
(``TraceConfig(blob_uploader=...)``) or via the ``openinference_blob_uploader``
entry-point group selected with the ``OPENINFERENCE_BLOB_UPLOADER``
environment variable.

Implementations must never block the instrumented call path: return the
destination URI immediately (content-addressed naming makes it computable
before any I/O) and move the bytes on a background worker.
"""

import base64
import hashlib
import inspect
import re
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import Optional, Protocol, Tuple, runtime_checkable

from .logging import logger

__all__ = (
    "Blob",
    "BlobUploader",
    "load_blob_uploader",
    "parse_base64_data_uri",
)

BLOB_UPLOADER_ENTRY_POINT_GROUP = "openinference_blob_uploader"

_DATA_URI_PATTERN = re.compile(r"^data:(?P<mime>[^;,]+);base64,(?P<content>.+)$", re.DOTALL)

_MODALITY_BY_MIME_PREFIX = {
    "image/": "image",
    "audio/": "audio",
    "video/": "video",
}


def parse_base64_data_uri(url: str) -> Optional[Tuple[str, str]]:
    """
    Parses a ``data:<mime>;base64,<payload>`` URI of any media type.
    Returns ``(mime_type, base64_payload)``, or None if the value is not a
    base64 data URI.
    """
    if not isinstance(url, str):
        return None
    if match := _DATA_URI_PATTERN.match(url):
        return match.group("mime"), match.group("content")
    return None


def _modality_from_mime_type(mime_type: str) -> str:
    for prefix, modality in _MODALITY_BY_MIME_PREFIX.items():
        if mime_type.startswith(prefix):
            return modality
    return "document"


@dataclass(frozen=True)
class Blob:
    """
    A unit of binary content captured during instrumentation, e.g. the
    decoded payload of a base64 data URI found in a span attribute.
    """

    data: bytes
    mime_type: str
    modality: str = ""
    """One of "image", "audio", "video", or "document". Derived from the
    mime type when not provided."""
    attribute_key: Optional[str] = None
    """The span attribute key the content was captured from, if any."""
    content_sha256: str = field(default="")
    """Hex digest of the decoded bytes. Computed automatically."""

    def __post_init__(self) -> None:
        if not self.modality:
            object.__setattr__(self, "modality", _modality_from_mime_type(self.mime_type))
        if not self.content_sha256:
            object.__setattr__(self, "content_sha256", hashlib.sha256(self.data).hexdigest())


@runtime_checkable
class BlobUploader(Protocol):
    """
    Uploads blobs to external storage, returning a reference URI that is
    recorded in span attributes in place of inline base64 content.
    """

    def upload(self, blob: Blob) -> Optional[str]:
        """
        Returns the destination URI immediately and performs the actual
        write asynchronously, or returns None if the blob cannot be
        accepted (the caller then falls back to redaction). Must not block
        the instrumented call path.
        """
        ...

    def shutdown(self, timeout_sec: float = 10.0) -> None:
        """Flushes pending uploads and stops background workers."""
        ...


def load_blob_uploader(name: str) -> Optional[BlobUploader]:
    """
    Loads a ``BlobUploader`` registered under the
    ``openinference_blob_uploader`` entry-point group (mirroring OTel
    util-genai's ``opentelemetry_genai_completion_hook`` mechanics). The
    entry point may resolve to an uploader instance or to a zero-argument
    callable (e.g. a class) producing one. Returns None — and logs — when
    the name is unknown or the loaded object does not satisfy the protocol.
    """
    try:
        for entry_point in entry_points(group=BLOB_UPLOADER_ENTRY_POINT_GROUP):
            if entry_point.name != name:
                continue
            loaded = entry_point.load()
            # runtime_checkable protocols match class objects too (the class
            # itself has the methods as attributes), so check for classes and
            # factories explicitly before the protocol check.
            if inspect.isclass(loaded) or (
                callable(loaded) and not isinstance(loaded, BlobUploader)
            ):
                loaded = loaded()
            if isinstance(loaded, BlobUploader) and not inspect.isclass(loaded):
                return loaded
            logger.warning(
                f"Entry point '{name}' in group '{BLOB_UPLOADER_ENTRY_POINT_GROUP}' "
                "did not produce a BlobUploader; ignoring it."
            )
            return None
        logger.warning(
            f"No blob uploader entry point named '{name}' found in group "
            f"'{BLOB_UPLOADER_ENTRY_POINT_GROUP}'. Oversized media will be redacted."
        )
    except Exception:
        logger.exception(f"Failed to load blob uploader '{name}'.")
    return None


def decode_base64_data_uri_to_blob(url: str, attribute_key: Optional[str] = None) -> Optional[Blob]:
    """
    Decodes a base64 data URI into a ``Blob``, or returns None if the value
    is not a base64 data URI or the payload cannot be decoded.
    """
    parsed = parse_base64_data_uri(url)
    if parsed is None:
        return None
    mime_type, payload = parsed
    try:
        data = base64.b64decode(payload)
    except Exception:
        return None
    return Blob(data=data, mime_type=mime_type, attribute_key=attribute_key)

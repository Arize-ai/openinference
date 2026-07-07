"""
Experimental support for externalizing large binary media (audio, files,
images) captured during instrumentation.

Instead of recording multi-megabyte base64 data URIs in span attributes —
which exceed OTLP payload limits and inflate backend storage — a
``BlobUploader`` uploads the decoded bytes to external storage and the span
attribute records only the destination URI. This mirrors the OTel GenAI
semantic conventions message model, where inline base64 is a ``blob`` part
and an external reference is a ``uri`` part.

The upload never blocks the instrumented call path: the destination URI is
computed synchronously (content-addressed by SHA-256 of the decoded bytes)
and the write happens on a background worker thread.
"""

import base64
import hashlib
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Optional, Protocol, Set, Tuple, runtime_checkable
from urllib.parse import urlparse

from .logging import logger

__all__ = (
    "Blob",
    "BlobUploader",
    "FsspecBlobUploader",
    "parse_base64_data_uri",
)

DEFAULT_BLOB_UPLOAD_MAX_QUEUE_SIZE = 20

_DATA_URI_PATTERN = re.compile(r"^data:(?P<mime>[^;,]+);base64,(?P<content>.+)$", re.DOTALL)

_MODALITY_BY_MIME_PREFIX = {
    "image/": "image",
    "audio/": "audio",
    "video/": "video",
}

_EXTENSION_BY_MIME = {
    "application/json": ".json",
    "application/pdf": ".pdf",
    "audio/aac": ".aac",
    "audio/flac": ".flac",
    "audio/mp3": ".mp3",
    "audio/mpeg": ".mp3",
    "audio/ogg": ".ogg",
    "audio/wav": ".wav",
    "audio/webm": ".webm",
    "audio/x-wav": ".wav",
    "image/gif": ".gif",
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "text/plain": ".txt",
    "video/mp4": ".mp4",
    "video/webm": ".webm",
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


def _extension_from_mime_type(mime_type: str) -> str:
    if extension := _EXTENSION_BY_MIME.get(mime_type.lower()):
        return extension
    import mimetypes

    return mimetypes.guess_extension(mime_type) or ".bin"


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


class FsspecBlobUploader:
    """
    A ``BlobUploader`` that writes blobs to any fsspec-supported filesystem
    (``s3://``, ``gs://``, ``file://``, plain local paths, ...) from a
    single background worker thread.

    Destination URIs are content-addressed:
    ``{base_path}/{sha256(data)}{extension}``, so identical content
    deduplicates and the URI is known before the write completes.

    Remote schemes require ``fsspec`` (and the matching filesystem
    implementation, e.g. ``s3fs``); local paths work without fsspec.
    """

    def __init__(
        self,
        base_path: str,
        *,
        max_queue_size: int = DEFAULT_BLOB_UPLOAD_MAX_QUEUE_SIZE,
    ) -> None:
        self._base_path = base_path.rstrip("/")
        # Queue(maxsize=0) would mean unbounded; the queue must stay bounded
        # so a slow destination can never buffer unbounded memory.
        self._queue: "Queue[Optional[Tuple[str, Blob]]]" = Queue(maxsize=max(1, max_queue_size))
        self._seen_digests: Set[str] = set()
        self._seen_lock = threading.Lock()
        self._filesystem, self._filesystem_root = self._open_filesystem(self._base_path)
        self._worker = threading.Thread(
            target=self._work,
            name="openinference-blob-uploader",
            daemon=True,
        )
        self._stopped = False
        self._worker.start()

    @staticmethod
    def _open_filesystem(base_path: str) -> Tuple[Optional[object], str]:
        scheme = urlparse(base_path).scheme
        try:
            import fsspec  # type: ignore[import-untyped,import-not-found,unused-ignore]

            filesystem, root = fsspec.url_to_fs(base_path)
            return filesystem, str(root).rstrip("/")
        except ImportError:
            # Local destinations work without fsspec; remote schemes do not.
            if scheme in ("", "file"):
                root = base_path[len("file://") :] if scheme == "file" else base_path
                return None, root.rstrip("/")
            raise ImportError(
                f"fsspec is required to upload blobs to '{base_path}'. "
                "Install it with: pip install openinference-instrumentation[blob-upload]"
            ) from None

    def upload(self, blob: Blob) -> Optional[str]:
        if self._stopped:
            return None
        uri = self._destination_uri(blob)
        with self._seen_lock:
            if blob.content_sha256 in self._seen_digests:
                return uri
        try:
            self._queue.put_nowait((self._destination_path(blob), blob))
        except Full:
            logger.warning(
                "Blob upload queue is full; falling back to redaction for "
                f"attribute {blob.attribute_key!r}."
            )
            return None
        with self._seen_lock:
            self._seen_digests.add(blob.content_sha256)
        return uri

    def shutdown(self, timeout_sec: float = 10.0) -> None:
        self._stopped = True
        try:
            self._queue.put_nowait(None)
        except Full:
            pass
        self._worker.join(timeout=timeout_sec)

    def _destination_name(self, blob: Blob) -> str:
        return f"{blob.content_sha256}{_extension_from_mime_type(blob.mime_type)}"

    def _destination_uri(self, blob: Blob) -> str:
        return f"{self._base_path}/{self._destination_name(blob)}"

    def _destination_path(self, blob: Blob) -> str:
        return f"{self._filesystem_root}/{self._destination_name(blob)}"

    def _work(self) -> None:
        while True:
            try:
                item = self._queue.get(timeout=0.2)
            except Empty:
                if self._stopped:
                    return
                continue
            if item is None:
                return
            path, blob = item
            try:
                self._write(path, blob.data)
            except Exception:
                logger.exception(f"Failed to upload blob to {path!r}.")
            finally:
                self._queue.task_done()

    def _write(self, path: str, data: bytes) -> None:
        if self._filesystem is not None:
            self._filesystem.pipe_file(path, data)  # type: ignore[attr-defined]
        else:
            destination = Path(path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)


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

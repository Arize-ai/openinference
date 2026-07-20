"""The proposed blob-upload interface (../blob_uploads.md §2.1) plus a demo store.

``Blob`` and ``BlobUploader`` are the pieces that would move into the
``openinference-instrumentation`` package; the demos define them here because the
released packages don't ship them yet.

``LocalBlobStore`` is a deliberately simple mock backend: it writes
content-addressed files under ``scripts/blob_store/`` and returns the file's path
relative to the repo root as the URI. Rendering (or just displaying) the URI is the
backend's responsibility — Phoenix shows it as an ordinary string attribute. The
packaged ``FsspecBlobUploader`` described in the spec adds the real-world concerns
this mock skips: remote schemes, a bounded queue, and background writes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

_MODALITY_BY_MIME_PREFIX = {"image/": "image", "audio/": "audio", "video/": "video"}

_EXT_BY_MIME = {
    "audio/wav": ".wav",
    "audio/mpeg": ".mp3",
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "application/pdf": ".pdf",
}


@dataclass(frozen=True)
class Blob:
    """A finalized binary payload captured by an instrumentor."""

    data: bytes
    """Decoded bytes — never base64 text."""

    mime_type: str
    """IANA media type, e.g. ``audio/wav``, ``image/png``."""

    modality: str = ""
    """"image" | "audio" | "video" | "document"; derived from mime_type when omitted."""

    attribute_key: Optional[str] = None
    """Span attribute the reference will be written to, e.g. ``input.audio.url``."""

    content_sha256: str = ""
    """Hex digest of ``data``; computed automatically."""

    def __post_init__(self) -> None:
        if not self.modality:
            modality = next(
                (
                    m
                    for p, m in _MODALITY_BY_MIME_PREFIX.items()
                    if self.mime_type.startswith(p)
                ),
                "document",
            )
            object.__setattr__(self, "modality", modality)
        if not self.content_sha256:
            object.__setattr__(
                self, "content_sha256", hashlib.sha256(self.data).hexdigest()
            )


@runtime_checkable
class BlobUploader(Protocol):
    """Pluggable destination for large multimodal payloads.

    ``upload`` MUST return quickly: implementations compute the destination URI
    synchronously and transfer bytes on a background worker. ``None`` means the
    blob cannot be accepted (backpressure, shutdown) — the caller then records
    ``__REDACTED__``, the uniform fallback.
    """

    def upload(self, blob: Blob) -> Optional[str]: ...

    def shutdown(self, timeout_sec: float = 10.0) -> None: ...


class LocalBlobStore:
    """Mock ``BlobUploader``: content-addressed files in the local repo.

    Returns the stored file's repo-root-relative path as the URI, e.g.
    ``internal_docs/specs/blob_uploads/scripts/blob_store/3a7bd3….wav``.
    Writes synchronously — fine for a demo; the packaged uploader is async.
    """

    def __init__(self, root_dir: Optional[Path] = None) -> None:
        self.root_dir = root_dir or Path(__file__).parent / "blob_store"
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._repo_root = self._find_repo_root(self.root_dir)

    @staticmethod
    def _find_repo_root(start: Path) -> Optional[Path]:
        for parent in [start, *start.resolve().parents]:
            if (parent / ".git").exists():
                return parent
        return None

    def upload(self, blob: Blob) -> Optional[str]:
        name = blob.content_sha256[:20] + _EXT_BY_MIME.get(blob.mime_type, ".bin")
        path = self.root_dir / name
        if not path.exists():  # content-addressed dedup
            path.write_bytes(blob.data)
            print(f"[blob] stored {blob.modality} ({len(blob.data):,} B) → {path.name}")
        if self._repo_root is not None:
            return str(path.resolve().relative_to(self._repo_root))
        return path.resolve().as_posix()

    def shutdown(self, timeout_sec: float = 10.0) -> None:
        pass  # synchronous mock — nothing pending

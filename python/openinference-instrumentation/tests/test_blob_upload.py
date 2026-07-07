import base64
from pathlib import Path
from typing import Optional

import pytest

from openinference.instrumentation import Blob, BlobUploader, FsspecBlobUploader, TraceConfig
from openinference.instrumentation.config import (
    OPENINFERENCE_BASE64_MEDIA_MAX_LENGTH,
    OPENINFERENCE_BLOB_UPLOAD_BASE_PATH,
    REDACTED_VALUE,
)
from openinference.semconv.trace import (
    AudioAttributes,
    FileAttributes,
    ImageAttributes,
    MessageAttributes,
    MessageContentAttributes,
    SpanAttributes,
)

WAV_BYTES = b"RIFF" + bytes(range(256)) * 40  # ~10KB of fake audio
PDF_BYTES = b"%PDF-1.4" + bytes(range(256)) * 40
PNG_BYTES = b"\x89PNG\r\n" + bytes(range(256)) * 40

AUDIO_DATA_URI = "data:audio/wav;base64," + base64.b64encode(WAV_BYTES).decode()
PDF_DATA_URI = "data:application/pdf;base64," + base64.b64encode(PDF_BYTES).decode()
PNG_DATA_URI = "data:image/png;base64," + base64.b64encode(PNG_BYTES).decode()

INPUT_AUDIO_URL_KEY = (
    f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENTS}.0."
    f"{MessageContentAttributes.MESSAGE_CONTENT_AUDIO}.{AudioAttributes.AUDIO_URL}"
)
OUTPUT_AUDIO_URL_KEY = (
    f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENTS}.0."
    f"{MessageContentAttributes.MESSAGE_CONTENT_AUDIO}.{AudioAttributes.AUDIO_URL}"
)
INPUT_FILE_URL_KEY = (
    f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENTS}.1."
    f"{MessageContentAttributes.MESSAGE_CONTENT_FILE}.{FileAttributes.FILE_URL}"
)
INPUT_IMAGE_URL_KEY = (
    f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENTS}.2."
    f"{MessageContentAttributes.MESSAGE_CONTENT_IMAGE}.{ImageAttributes.IMAGE_URL}"
)


def test_blob_derives_modality_and_digest() -> None:
    audio_blob = Blob(data=WAV_BYTES, mime_type="audio/wav")
    assert audio_blob.modality == "audio"
    assert len(audio_blob.content_sha256) == 64
    assert Blob(data=PDF_BYTES, mime_type="application/pdf").modality == "document"
    assert Blob(data=PNG_BYTES, mime_type="image/png").modality == "image"
    assert Blob(data=b"", mime_type="video/mp4").modality == "video"


def test_fsspec_uploader_writes_content_addressed_file(tmp_path: Path) -> None:
    uploader = FsspecBlobUploader(base_path=str(tmp_path))
    blob = Blob(data=WAV_BYTES, mime_type="audio/wav")
    uri = uploader.upload(blob)
    assert uri == f"{tmp_path}/{blob.content_sha256}.wav"
    # Re-uploading identical content returns the same URI without re-enqueueing.
    assert uploader.upload(Blob(data=WAV_BYTES, mime_type="audio/wav")) == uri
    uploader.shutdown()
    assert Path(uri).read_bytes() == WAV_BYTES


def test_fsspec_uploader_rejects_after_shutdown(tmp_path: Path) -> None:
    uploader = FsspecBlobUploader(base_path=str(tmp_path))
    uploader.shutdown()
    assert uploader.upload(Blob(data=WAV_BYTES, mime_type="audio/wav")) is None


def test_uploader_satisfies_protocol(tmp_path: Path) -> None:
    assert isinstance(FsspecBlobUploader(base_path=str(tmp_path)), BlobUploader)


def test_mask_uploads_oversized_input_audio(tmp_path: Path) -> None:
    uploader = FsspecBlobUploader(base_path=str(tmp_path))
    config = TraceConfig(blob_uploader=uploader, base64_media_max_length=100)
    masked = config.mask(INPUT_AUDIO_URL_KEY, AUDIO_DATA_URI)
    assert isinstance(masked, str)
    assert masked.startswith(str(tmp_path))
    assert masked.endswith(".wav")
    uploader.shutdown()
    assert Path(masked).read_bytes() == WAV_BYTES


def test_mask_uploads_oversized_output_audio(tmp_path: Path) -> None:
    uploader = FsspecBlobUploader(base_path=str(tmp_path))
    config = TraceConfig(blob_uploader=uploader, base64_media_max_length=100)
    masked = config.mask(OUTPUT_AUDIO_URL_KEY, AUDIO_DATA_URI)
    assert isinstance(masked, str) and masked.endswith(".wav")
    uploader.shutdown()


def test_mask_uploads_oversized_input_file(tmp_path: Path) -> None:
    uploader = FsspecBlobUploader(base_path=str(tmp_path))
    config = TraceConfig(blob_uploader=uploader, base64_media_max_length=100)
    masked = config.mask(INPUT_FILE_URL_KEY, PDF_DATA_URI)
    assert isinstance(masked, str)
    assert masked.endswith(".pdf")
    uploader.shutdown()
    assert Path(masked).read_bytes() == PDF_BYTES


def test_mask_uploads_oversized_input_image(tmp_path: Path) -> None:
    uploader = FsspecBlobUploader(base_path=str(tmp_path))
    config = TraceConfig(blob_uploader=uploader, base64_image_max_length=100)
    masked = config.mask(INPUT_IMAGE_URL_KEY, PNG_DATA_URI)
    assert isinstance(masked, str)
    assert masked.endswith(".png")
    uploader.shutdown()
    assert Path(masked).read_bytes() == PNG_BYTES


def test_mask_redacts_oversized_audio_without_uploader() -> None:
    config = TraceConfig(base64_media_max_length=100)
    assert config.mask(INPUT_AUDIO_URL_KEY, AUDIO_DATA_URI) == REDACTED_VALUE


def test_mask_redacts_when_uploader_rejects(tmp_path: Path) -> None:
    uploader = FsspecBlobUploader(base_path=str(tmp_path))
    uploader.shutdown()  # upload() now returns None
    config = TraceConfig(blob_uploader=uploader, base64_media_max_length=100)
    assert config.mask(INPUT_AUDIO_URL_KEY, AUDIO_DATA_URI) == REDACTED_VALUE


def test_mask_keeps_small_audio_inline() -> None:
    config = TraceConfig(base64_media_max_length=len(AUDIO_DATA_URI) + 1)
    assert config.mask(INPUT_AUDIO_URL_KEY, AUDIO_DATA_URI) == AUDIO_DATA_URI


def test_mask_keeps_external_audio_url() -> None:
    config = TraceConfig(base64_media_max_length=10)
    url = "https://example.com/audio.mp3"
    assert config.mask(INPUT_AUDIO_URL_KEY, url) == url


@pytest.mark.parametrize(
    "param,key",
    [
        ("hide_input_audio", INPUT_AUDIO_URL_KEY),
        ("hide_output_audio", OUTPUT_AUDIO_URL_KEY),
        ("hide_input_files", INPUT_FILE_URL_KEY),
    ],
)
def test_hide_settings_drop_media(param: str, key: str) -> None:
    config = TraceConfig(**{param: True})  # type: ignore[arg-type]
    assert config.mask(key, AUDIO_DATA_URI) is None


def test_hide_takes_precedence_over_upload(tmp_path: Path) -> None:
    uploader = FsspecBlobUploader(base_path=str(tmp_path))
    config = TraceConfig(
        blob_uploader=uploader,
        hide_input_audio=True,
        base64_media_max_length=100,
    )
    assert config.mask(INPUT_AUDIO_URL_KEY, AUDIO_DATA_URI) is None
    uploader.shutdown()
    # Hidden content must never reach storage.
    assert not list(tmp_path.iterdir())


def test_blob_uploader_from_env_var(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(OPENINFERENCE_BLOB_UPLOAD_BASE_PATH, str(tmp_path))
    monkeypatch.setenv(OPENINFERENCE_BASE64_MEDIA_MAX_LENGTH, "100")
    config = TraceConfig()
    assert config.blob_uploader is not None
    masked = config.mask(INPUT_AUDIO_URL_KEY, AUDIO_DATA_URI)
    assert isinstance(masked, str)
    assert masked.startswith(str(tmp_path))
    config.blob_uploader.shutdown()


def test_custom_uploader_implementation() -> None:
    class StaticUploader:
        def __init__(self) -> None:
            self.blobs: "list[Blob]" = []

        def upload(self, blob: Blob) -> Optional[str]:
            self.blobs.append(blob)
            return f"memory://{blob.content_sha256}"

        def shutdown(self, timeout_sec: float = 10.0) -> None:
            pass

    uploader = StaticUploader()
    config = TraceConfig(blob_uploader=uploader, base64_media_max_length=100)
    masked = config.mask(INPUT_AUDIO_URL_KEY, AUDIO_DATA_URI)
    assert isinstance(masked, str) and masked.startswith("memory://")
    assert len(uploader.blobs) == 1
    assert uploader.blobs[0].mime_type == "audio/wav"
    assert uploader.blobs[0].modality == "audio"
    assert uploader.blobs[0].attribute_key == INPUT_AUDIO_URL_KEY

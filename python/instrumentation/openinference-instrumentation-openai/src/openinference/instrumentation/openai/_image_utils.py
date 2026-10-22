import base64
import copy
import mimetypes
import os
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple, Union, cast

from opentelemetry.util.types import AttributeValue

from openinference.instrumentation import (
    REDACTED_VALUE,
    TraceConfig,
    decode_base64_data_uri_to_blob,
    safe_json_dumps,
)
from openinference.instrumentation.config import is_base64_url
from openinference.semconv.trace import ImageAttributes, SpanAttributes


def get_attributes_from_image_files(
    files: Any,
) -> Iterator[Tuple[str, AttributeValue]]:
    """Extract span-level image attributes from OpenAI multipart request files."""
    index = 0
    for _, file in _iter_image_files(files):
        if data_url := _file_to_data_url(file):
            yield (
                f"{SpanAttributes.INPUT_IMAGES}.{index}.{ImageAttributes.IMAGE_URL}",
                data_url,
            )
            index += 1


def image_b64_to_data_url(b64_json: str, image_format: Any = None) -> str:
    """Convert an Images API ``b64_json`` value into an image data URL."""
    if b64_json.startswith("data:image/"):
        return b64_json
    media_type = _media_type_from_format(image_format) or "image/png"
    return f"data:{media_type};base64,{b64_json}"


def _iter_image_files(files: Any) -> Iterator[Tuple[str, Any]]:
    if isinstance(files, Mapping):
        entries: Iterable[Any] = files.items()
    elif isinstance(files, (list, tuple)):
        entries = files
    else:
        return
    for entry in entries:
        if not isinstance(entry, tuple) or len(entry) != 2:
            continue
        field_name, file = entry
        if field_name in ("image", "image[]", "mask"):
            yield field_name, file


def _file_to_data_url(file: Any) -> Optional[str]:
    filename: Optional[str] = None
    media_type: Optional[str] = None
    content = file
    if isinstance(file, os.PathLike):
        filename = os.fspath(file)
    elif isinstance(file, tuple):
        if len(file) < 2:
            return None
        if isinstance(file[0], (str, os.PathLike)):
            filename = os.fspath(file[0])
        content = file[1]
        if len(file) >= 3 and isinstance(file[2], str) and file[2].startswith("image/"):
            media_type = file[2]
    data = _read_file_bytes(content)
    if data is None:
        return None
    media_type = media_type or _sniff_image_media_type(data) or _media_type_from_filename(filename)
    return f"data:{media_type or 'image/png'};base64,{base64.b64encode(data).decode('ascii')}"


def _read_file_bytes(content: Any) -> Optional[bytes]:
    if isinstance(content, bytes):
        return content
    if isinstance(content, os.PathLike):
        try:
            with open(content, "rb") as file:
                return file.read()
        except OSError:
            return None
    if not hasattr(content, "read") or not callable(content.read):
        return None
    try:
        if hasattr(content, "seekable") and not content.seekable():
            return None
        position = content.tell()
        try:
            content.seek(0)
            data = content.read()
        finally:
            # Reading is observational: leave the stream exactly where the SDK
            # found it, even when read() itself raises after advancing it.
            content.seek(position)
    except Exception:
        return None
    return data if isinstance(data, bytes) else None


def _media_type_from_filename(filename: Optional[str]) -> Optional[str]:
    if not filename:
        return None
    media_type, _ = mimetypes.guess_type(filename)
    return media_type if media_type and media_type.startswith("image/") else None


def _media_type_from_format(image_format: Any) -> Optional[str]:
    if not isinstance(image_format, str):
        return None
    normalized = image_format.lower()
    if normalized == "jpg":
        normalized = "jpeg"
    if normalized in ("png", "jpeg", "webp", "gif"):
        return f"image/{normalized}"
    return None


def _sniff_image_media_type(data: bytes) -> Optional[str]:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if len(data) >= 12 and data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def serialize_request_input(
    request_parameters: Mapping[str, Any],
    config: TraceConfig,
) -> str:
    if config.hide_inputs:
        return REDACTED_VALUE

    request_copy = copy.deepcopy(dict(request_parameters))
    messages = request_copy.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict):
                _process_chat_content(message.get("content"), config)

    inputs = request_copy.get("input")
    if isinstance(inputs, list):
        for input_item in inputs:
            if isinstance(input_item, dict):
                _process_response_content(input_item.get("content"), config)

    return safe_json_dumps(request_copy)


def _process_chat_content(
    content: Union[str, List[Dict[str, Any]], None],
    config: TraceConfig,
) -> None:
    if not isinstance(content, list):
        return
    for content_item in content:
        if not isinstance(content_item, dict) or content_item.get("type") != "image_url":
            continue
        image_url = content_item.get("image_url")
        if not isinstance(image_url, dict):
            continue
        url = image_url.get("url")
        if isinstance(url, str):
            image_url["url"] = _replace_image_url(url, config)


def _process_response_content(
    content: Union[str, List[Dict[str, Any]], None],
    config: TraceConfig,
) -> None:
    if not isinstance(content, list):
        return
    for content_item in content:
        if not isinstance(content_item, dict) or content_item.get("type") != "input_image":
            continue
        url = content_item.get("image_url")
        if isinstance(url, str):
            content_item["image_url"] = _replace_image_url(url, config)


def _replace_image_url(url: str, config: TraceConfig) -> str:
    if config.hide_input_images:
        return REDACTED_VALUE
    max_length = cast(int, config.base64_image_max_length)
    if not is_base64_url(url) or len(url) <= max_length:
        return url
    if config.blob_uploader is None:
        return REDACTED_VALUE
    blob = decode_base64_data_uri_to_blob(url)
    if blob is None:
        return REDACTED_VALUE
    return config.externalize_blob(
        blob.data,
        mime_type=blob.mime_type,
        attribute_key=SpanAttributes.INPUT_VALUE,
    )

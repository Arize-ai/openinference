import base64
import copy
import mimetypes
import os
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple, Union

from opentelemetry.util.types import AttributeValue

from openinference.instrumentation import REDACTED_VALUE
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
        data = content.read()
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


def redact_images_from_request_parameters(
    request_parameters: Dict[str, Any],
    hide_input_images: bool,
    base64_image_max_length: int,
) -> Dict[str, Any]:
    """
    Create a copy of request parameters with image data redacted based on configuration.

    Args:
        request_parameters: The original request parameters
        hide_input_images: Whether to hide image data
        base64_image_max_length: Maximum length for base64 images before redaction

    Returns:
        Modified copy of request parameters with images redacted if configured
    """
    if not hide_input_images and base64_image_max_length <= 0:
        return request_parameters

    modified_params = copy.deepcopy(request_parameters)

    # Process messages if they exist (Chat Completions API)
    if "messages" in modified_params and isinstance(modified_params["messages"], list):
        for message in modified_params["messages"]:
            if isinstance(message, dict) and "content" in message:
                _redact_images_from_message_content(
                    message["content"], base64_image_max_length, hide_input_images
                )

    # Process input if they exist (Responses API)
    if "input" in modified_params and isinstance(modified_params["input"], list):
        for input_item in modified_params["input"]:
            if isinstance(input_item, dict) and "content" in input_item:
                _redact_images_from_input_content(
                    input_item["content"], base64_image_max_length, hide_input_images
                )

    return modified_params


def _redact_images_from_message_content(
    content: Union[str, List[Dict[str, Any]]], base64_image_max_length: int, hide_input_images: bool
) -> None:
    """
    Redact image data from message content in-place.

    Args:
        content: Message content (string or list of content items)
        base64_image_max_length: Maximum length for base64 images before redaction
        hide_input_images: Whether to hide all images
    """
    if not isinstance(content, list):
        return

    for content_item in content:
        if not isinstance(content_item, dict):
            continue

        content_type = content_item.get("type")

        if content_type == "image_url" and "image_url" in content_item:
            image_url_obj = content_item["image_url"]
            if isinstance(image_url_obj, dict) and "url" in image_url_obj:
                url = image_url_obj["url"]
                if isinstance(url, str):
                    should_redact = False

                    # When hide_input_images=True, redact ALL images regardless of type or length
                    if hide_input_images:
                        should_redact = True
                    # When hide_input_images=False, only redact b64 images that exceed length limit
                    elif is_base64_url(url) and len(url) > base64_image_max_length:
                        should_redact = True

                    if should_redact:
                        image_url_obj["url"] = REDACTED_VALUE


def _redact_images_from_input_content(
    content: Union[str, List[Dict[str, Any]]], base64_image_max_length: int, hide_input_images: bool
) -> None:
    """
    Redact image data from responses API input content in-place.

    Args:
        content: Input content (string or list of content items)
        base64_image_max_length: Maximum length for base64 images before redaction
        hide_input_images: Whether to hide all images
    """
    if not isinstance(content, list):
        return

    for content_item in content:
        if not isinstance(content_item, dict):
            continue

        content_type = content_item.get("type")

        if content_type == "input_image" and "image_url" in content_item:
            url = content_item["image_url"]
            if isinstance(url, str):
                should_redact = False

                if hide_input_images:
                    should_redact = True
                elif is_base64_url(url) and len(url) > base64_image_max_length:
                    should_redact = True

                if should_redact:
                    content_item["image_url"] = REDACTED_VALUE

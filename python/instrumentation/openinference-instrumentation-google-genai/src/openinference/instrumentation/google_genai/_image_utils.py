from typing import Any

from openinference.instrumentation import REDACTED_VALUE
from openinference.instrumentation.config import is_base64_url


def redact_images_from_request_parameters(
    request_parameters: dict[str, Any],
    hide_input_images: bool,
    base64_image_max_length: int,
) -> dict[str, Any]:
    """Return request parameters with configured image data redacted.

    The Google GenAI SDK's serialized request represents inline images as a
    mapping containing ``mime_type`` and bare base64 ``data`` fields. The
    structured message attributes convert that representation into a data URL,
    so the length check here includes the equivalent data URL prefix to match
    ``TraceConfig.mask`` semantics.
    """
    redacted = _redact_images(
        request_parameters,
        hide_input_images=hide_input_images,
        base64_image_max_length=base64_image_max_length,
    )
    assert isinstance(redacted, dict)
    return redacted


def _redact_images(
    value: Any,
    *,
    hide_input_images: bool,
    base64_image_max_length: int,
) -> Any:
    """Walks the request tree and finds dicts with mime_type starting with
    image/ and redacts
    """
    if isinstance(value, list):
        return [
            _redact_images(
                item,
                hide_input_images=hide_input_images,
                base64_image_max_length=base64_image_max_length,
            )
            for item in value
        ]
    if not isinstance(value, dict):
        return value

    result = {
        key: _redact_images(
            item,
            hide_input_images=hide_input_images,
            base64_image_max_length=base64_image_max_length,
        )
        for key, item in value.items()
    }
    mime_type = result.get("mime_type")
    if not isinstance(mime_type, str) or not mime_type.startswith("image/"):
        return result

    if hide_input_images:
        if "data" in result:
            result["data"] = REDACTED_VALUE
        if "file_uri" in result:
            result["file_uri"] = REDACTED_VALUE
        return result

    data = result.get("data")
    if (
        image_url_length := _get_image_url_length(data, mime_type)
    ) is not None and image_url_length > base64_image_max_length:
        result["data"] = REDACTED_VALUE
    return result


def _get_image_url_length(data: Any, mime_type: str) -> int | None:
    if isinstance(data, str):
        if is_base64_url(data):
            return len(data)
        encoded_data_length = len(data)
    elif isinstance(data, (bytes, bytearray, memoryview)):
        encoded_data_length = 4 * ((len(data) + 2) // 3)
    else:
        return None
    return len("data:") + len(mime_type) + len(";base64,") + encoded_data_length

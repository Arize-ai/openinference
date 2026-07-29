import base64
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
    """Walk the request tree and redact configured image values."""
    # if this is a Pydantic object, dump it as a walkable tree first
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        value = model_dump(mode="python")

    if isinstance(value, list):
        return [
            _redact_images(
                item,
                hide_input_images=hide_input_images,
                base64_image_max_length=base64_image_max_length,
            )
            for item in value
        ]
    if isinstance(value, (bytes, bytearray, memoryview)):
        return base64.b64encode(value).decode("ascii")
    if not isinstance(value, dict):
        return value

    values = value
    mime_type = value.get("mime_type")
    if isinstance(mime_type, str) and mime_type.startswith("image/"):
        values = dict(value)
        if hide_input_images:
            for key in ("data", "file_uri"):
                if key in values:
                    values[key] = REDACTED_VALUE
        elif (
            image_url_length := _get_image_url_length(values.get("data"), mime_type)
        ) is not None and image_url_length > base64_image_max_length:
            values["data"] = REDACTED_VALUE

    return {
        key: _redact_images(
            item,
            hide_input_images=hide_input_images,
            base64_image_max_length=base64_image_max_length,
        )
        for key, item in values.items()
    }


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

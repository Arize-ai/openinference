import base64
from typing import Any

from openinference.instrumentation import REDACTED_VALUE
from openinference.instrumentation.config import is_base64_url

_IMAGE_VALUE_KEYS = ("data", "uri", "file_uri")


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
    return _redact_images_with_change(
        value,
        hide_input_images=hide_input_images,
        base64_image_max_length=base64_image_max_length,
    )[0]


def _redact_images_with_change(
    value: Any,
    *,
    hide_input_images: bool,
    base64_image_max_length: int,
) -> tuple[Any, bool]:
    """Return the walked value and whether any image value was redacted."""
    original_model = None
    # Google GenAI request types are Pydantic models; convert them to a
    # Python tree so nested media fields can be inspected before JSON encoding.
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        original_model = value
        value = model_dump(mode="python")

    if isinstance(value, list):
        list_results = [
            _redact_images_with_change(
                item,
                hide_input_images=hide_input_images,
                base64_image_max_length=base64_image_max_length,
            )
            for item in value
        ]
        return [item for item, _ in list_results], any(changed for _, changed in list_results)
    if isinstance(value, tuple):
        tuple_results = [
            _redact_images_with_change(
                item,
                hide_input_images=hide_input_images,
                base64_image_max_length=base64_image_max_length,
            )
            for item in value
        ]
        return tuple(item for item, _ in tuple_results), any(
            changed for _, changed in tuple_results
        )
    if isinstance(value, (bytes, bytearray, memoryview)):
        return base64.urlsafe_b64encode(value).decode("ascii"), False
    if not isinstance(value, dict):
        return value, False

    values = value
    # If we change nothing, then return the original Pydantic model
    changed = False
    mime_type = value.get("mime_type")
    if _is_image_container(value):
        values = dict(value)
        if hide_input_images:
            for key in _IMAGE_VALUE_KEYS:
                if key in values and values[key] != REDACTED_VALUE:
                    values[key] = REDACTED_VALUE
                    changed = True
        else:
            image_mime_type = (
                mime_type
                if isinstance(mime_type, str) and mime_type.startswith("image/")
                else "image/png"
                if value.get("type") == "image" and mime_type is None
                else None
            )
            if (
                image_mime_type is not None
                and (image_url_length := _get_image_url_length(values.get("data"), image_mime_type))
                is not None
                and image_url_length > base64_image_max_length
                and values.get("data") != REDACTED_VALUE
            ):
                values["data"] = REDACTED_VALUE
                changed = True
            for key in ("uri", "file_uri"):
                image_url = values.get(key)
                if (
                    isinstance(image_url, str)
                    and is_base64_url(image_url)
                    and len(image_url) > base64_image_max_length
                ):
                    values[key] = REDACTED_VALUE
                    changed = True

    dict_results = {
        key: _redact_images_with_change(
            item,
            hide_input_images=hide_input_images,
            base64_image_max_length=base64_image_max_length,
        )
        for key, item in values.items()
    }
    changed = changed or any(item_changed for _, item_changed in dict_results.values())
    if original_model is not None and not changed:
        return original_model, False
    return {key: item for key, (item, _) in dict_results.items()}, changed


def _is_image_container(value: dict[str, Any]) -> bool:
    mime_type = value.get("mime_type")
    if isinstance(mime_type, str) and mime_type.startswith("image/"):
        return True
    return value.get("type") == "image"


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

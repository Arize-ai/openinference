import copy
from typing import Any, Dict, List, Union

from openinference.instrumentation import REDACTED_VALUE
from openinference.instrumentation.config import is_base64_url


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

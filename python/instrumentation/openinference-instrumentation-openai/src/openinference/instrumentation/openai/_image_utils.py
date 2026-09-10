import copy
from typing import Any, Dict, List, Mapping, Union, cast

from openinference.instrumentation import (
    REDACTED_VALUE,
    TraceConfig,
    decode_base64_data_uri_to_blob,
    safe_json_dumps,
)
from openinference.instrumentation.config import is_base64_url
from openinference.semconv.trace import SpanAttributes


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

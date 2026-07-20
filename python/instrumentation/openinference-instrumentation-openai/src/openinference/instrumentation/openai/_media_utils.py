"""
Redaction/externalization of audio and file payloads embedded in the raw
request parameters before they are serialized into ``input.value``.

This complements the central ``TraceConfig`` masking of the structured
``message_content.audio`` / ``message_content.file`` attributes: without
this pass, multi-megabyte base64 payloads would survive inside the
JSON-serialized ``input.value`` attribute.
"""

import copy
from typing import Any, Dict, List, Optional, Union

from openinference.instrumentation import REDACTED_VALUE
from openinference.instrumentation._blob_upload import (
    BlobUploader,
    decode_base64_data_uri_to_blob,
)

from ._media import get_audio_data_uri, get_file_data_uri


def redact_media_from_request_parameters(
    request_parameters: Dict[str, Any],
    hide_input_audio: bool,
    hide_input_files: bool,
    base64_media_max_length: int,
    blob_uploader: Optional[BlobUploader] = None,
) -> Dict[str, Any]:
    """
    Create a copy of request parameters with base64 audio/file payloads
    hidden, redacted, or externalized based on configuration.

    When a blob uploader is configured, oversized payloads are uploaded and
    replaced with the destination URI; otherwise they are replaced with
    ``__REDACTED__``. Hidden content is never uploaded.
    """
    if not hide_input_audio and not hide_input_files and base64_media_max_length <= 0:
        return request_parameters

    modified_params = copy.deepcopy(request_parameters)

    # Chat Completions API
    if isinstance(messages := modified_params.get("messages"), list):
        for message in messages:
            if isinstance(message, dict) and isinstance(message.get("content"), list):
                _process_content_items(
                    message["content"],
                    hide_input_audio,
                    hide_input_files,
                    base64_media_max_length,
                    blob_uploader,
                )

    # Responses API
    if isinstance(input_items := modified_params.get("input"), list):
        for input_item in input_items:
            if isinstance(input_item, dict) and isinstance(input_item.get("content"), list):
                _process_content_items(
                    input_item["content"],
                    hide_input_audio,
                    hide_input_files,
                    base64_media_max_length,
                    blob_uploader,
                )

    return modified_params


def _process_content_items(
    content: Union[str, List[Dict[str, Any]]],
    hide_input_audio: bool,
    hide_input_files: bool,
    base64_media_max_length: int,
    blob_uploader: Optional[BlobUploader],
) -> None:
    if not isinstance(content, list):
        return
    for content_item in content:
        if not isinstance(content_item, dict):
            continue
        content_type = content_item.get("type")
        if content_type == "input_audio":
            # Both Chat Completions and Responses API use {"input_audio": {"data", "format"}}
            input_audio = content_item.get("input_audio")
            if isinstance(input_audio, dict) and isinstance(input_audio.get("data"), str):
                _process_payload(
                    input_audio,
                    "data",
                    get_audio_data_uri(input_audio["data"], input_audio.get("format")),
                    hide_input_audio,
                    base64_media_max_length,
                    blob_uploader,
                )
        elif content_type == "file":
            # Chat Completions API: {"file": {"file_data", "file_id", "filename"}}
            file = content_item.get("file")
            if isinstance(file, dict) and isinstance(file.get("file_data"), str):
                _process_payload(
                    file,
                    "file_data",
                    get_file_data_uri(file["file_data"], file.get("filename")),
                    hide_input_files,
                    base64_media_max_length,
                    blob_uploader,
                )
        elif content_type == "input_file":
            # Responses API: {"file_data", "file_id", "file_url", "filename"}
            if isinstance(content_item.get("file_data"), str):
                _process_payload(
                    content_item,
                    "file_data",
                    get_file_data_uri(content_item["file_data"], content_item.get("filename")),
                    hide_input_files,
                    base64_media_max_length,
                    blob_uploader,
                )


def _process_payload(
    container: Dict[str, Any],
    field: str,
    data_uri: str,
    hide: bool,
    base64_media_max_length: int,
    blob_uploader: Optional[BlobUploader],
) -> None:
    if hide:
        container[field] = REDACTED_VALUE
        return
    if base64_media_max_length <= 0 or len(data_uri) <= base64_media_max_length:
        return
    if blob_uploader is not None:
        if (blob := decode_base64_data_uri_to_blob(data_uri)) is not None:
            if uri := blob_uploader.upload(blob):
                container[field] = uri
                return
    container[field] = REDACTED_VALUE

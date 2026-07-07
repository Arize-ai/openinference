import base64
from types import SimpleNamespace
from typing import Any, Dict, Optional

import openai

from openinference.instrumentation import REDACTED_VALUE, Blob
from openinference.instrumentation.openai._attributes._responses_api import _ResponsesApiAttributes
from openinference.instrumentation.openai._media_utils import (
    redact_media_from_request_parameters,
)
from openinference.instrumentation.openai._request_attributes_extractor import (
    _RequestAttributesExtractor,
)
from openinference.instrumentation.openai._response_attributes_extractor import (
    _ResponseAttributesExtractor,
)

WAV_B64 = base64.b64encode(b"RIFF" + bytes(range(256)) * 10).decode()
PDF_B64 = base64.b64encode(b"%PDF-1.4" + bytes(range(256)) * 10).decode()


class TestChatCompletionsRequestMedia:
    def test_input_audio_part(self) -> None:
        extractor = _RequestAttributesExtractor(openai)
        actual = dict(
            extractor._get_attributes_from_message_content(
                {"type": "input_audio", "input_audio": {"data": WAV_B64, "format": "wav"}}
            )
        )
        assert actual == {
            "message_content.type": "audio",
            "message_content.audio.audio.url": f"data:audio/wav;base64,{WAV_B64}",
            "message_content.audio.audio.mime_type": "audio/wav",
        }

    def test_input_audio_mp3_mime_type(self) -> None:
        extractor = _RequestAttributesExtractor(openai)
        actual = dict(
            extractor._get_attributes_from_message_content(
                {"type": "input_audio", "input_audio": {"data": WAV_B64, "format": "mp3"}}
            )
        )
        assert actual["message_content.audio.audio.mime_type"] == "audio/mpeg"

    def test_file_part_with_file_data(self) -> None:
        extractor = _RequestAttributesExtractor(openai)
        actual = dict(
            extractor._get_attributes_from_message_content(
                {
                    "type": "file",
                    "file": {"file_data": PDF_B64, "filename": "report.pdf"},
                }
            )
        )
        assert actual == {
            "message_content.type": "file",
            "message_content.file.file.name": "report.pdf",
            "message_content.file.file.url": f"data:application/pdf;base64,{PDF_B64}",
            "message_content.file.file.mime_type": "application/pdf",
        }

    def test_file_part_with_data_uri_is_kept_verbatim(self) -> None:
        extractor = _RequestAttributesExtractor(openai)
        data_uri = f"data:application/pdf;base64,{PDF_B64}"
        actual = dict(
            extractor._get_attributes_from_message_content(
                {"type": "file", "file": {"file_data": data_uri, "filename": "report.pdf"}}
            )
        )
        assert actual["message_content.file.file.url"] == data_uri

    def test_file_part_with_file_id_only(self) -> None:
        extractor = _RequestAttributesExtractor(openai)
        actual = dict(
            extractor._get_attributes_from_message_content(
                {"type": "file", "file": {"file_id": "file-abc123"}}
            )
        )
        assert actual == {
            "message_content.type": "file",
            "message_content.file.file.id": "file-abc123",
        }


class TestResponsesApiRequestMedia:
    def test_input_file_with_file_data(self) -> None:
        actual = dict(
            _ResponsesApiAttributes._get_attributes_from_response_input_file_param(
                {
                    "type": "input_file",
                    "file_data": PDF_B64,
                    "filename": "report.pdf",
                }
            )
        )
        assert actual == {
            "message_content.type": "file",
            "message_content.file.file.name": "report.pdf",
            "message_content.file.file.url": f"data:application/pdf;base64,{PDF_B64}",
            "message_content.file.file.mime_type": "application/pdf",
        }

    def test_input_file_with_file_url_and_id(self) -> None:
        actual = dict(
            _ResponsesApiAttributes._get_attributes_from_response_input_file_param(
                {
                    "type": "input_file",
                    "file_url": "https://example.com/report.pdf",
                    "file_id": "file-abc123",
                }
            )
        )
        assert actual == {
            "message_content.type": "file",
            "message_content.file.file.url": "https://example.com/report.pdf",
            "message_content.file.file.id": "file-abc123",
        }

    def test_input_audio(self) -> None:
        actual = dict(
            _ResponsesApiAttributes._get_attributes_from_response_input_audio_param(
                {"type": "input_audio", "input_audio": {"data": WAV_B64, "format": "wav"}}
            )
        )
        assert actual == {
            "message_content.type": "audio",
            "message_content.audio.audio.url": f"data:audio/wav;base64,{WAV_B64}",
            "message_content.audio.audio.mime_type": "audio/wav",
        }

    def test_message_content_list_dispatches_file_and_audio(self) -> None:
        actual = dict(
            _ResponsesApiAttributes._get_attributes_from_message_param_content_list(
                [
                    {"type": "input_text", "text": "Summarize this."},
                    {"type": "input_file", "file_id": "file-abc123"},
                    {"type": "input_audio", "input_audio": {"data": WAV_B64, "format": "wav"}},
                ]
            )
        )
        assert actual["message.contents.1.message_content.type"] == "file"
        assert actual["message.contents.1.message_content.file.file.id"] == "file-abc123"
        assert actual["message.contents.2.message_content.type"] == "audio"
        assert (
            actual["message.contents.2.message_content.audio.audio.url"]
            == f"data:audio/wav;base64,{WAV_B64}"
        )


class TestChatCompletionsResponseAudio:
    def test_output_audio(self) -> None:
        extractor = _ResponseAttributesExtractor(openai)
        message = SimpleNamespace(
            role="assistant",
            content=None,
            audio=SimpleNamespace(
                id="audio-abc123",
                data=WAV_B64,
                transcript="Hello there!",
                expires_at=1,
            ),
            function_call=None,
            tool_calls=None,
        )
        actual = dict(
            extractor._get_attributes_from_chat_completion_message(message, audio_format="wav")
        )
        assert actual == {
            "message.role": "assistant",
            "message.contents.0.message_content.type": "audio",
            "message.contents.0.message_content.id": "audio-abc123",
            "message.contents.0.message_content.audio.audio.url": (
                f"data:audio/wav;base64,{WAV_B64}"
            ),
            "message.contents.0.message_content.audio.audio.mime_type": "audio/wav",
            "message.contents.0.message_content.audio.audio.transcript": "Hello there!",
        }


class _StaticUploader:
    def __init__(self) -> None:
        self.blobs: "list[Blob]" = []

    def upload(self, blob: Blob) -> Optional[str]:
        self.blobs.append(blob)
        return f"memory://{blob.content_sha256}"

    def shutdown(self, timeout_sec: float = 10.0) -> None:
        pass


class TestInputValueMediaRedaction:
    @staticmethod
    def _chat_params() -> Dict[str, Any]:
        return {
            "model": "gpt-4o-audio-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is in this recording?"},
                        {
                            "type": "input_audio",
                            "input_audio": {"data": WAV_B64, "format": "wav"},
                        },
                        {
                            "type": "file",
                            "file": {"file_data": PDF_B64, "filename": "report.pdf"},
                        },
                    ],
                }
            ],
        }

    def test_hide_input_audio(self) -> None:
        result = redact_media_from_request_parameters(
            self._chat_params(),
            hide_input_audio=True,
            hide_input_files=False,
            base64_media_max_length=0,
        )
        content = result["messages"][0]["content"]
        assert content[1]["input_audio"]["data"] == REDACTED_VALUE
        assert content[2]["file"]["file_data"] == PDF_B64

    def test_oversized_media_redacted_without_uploader(self) -> None:
        result = redact_media_from_request_parameters(
            self._chat_params(),
            hide_input_audio=False,
            hide_input_files=False,
            base64_media_max_length=100,
        )
        content = result["messages"][0]["content"]
        assert content[1]["input_audio"]["data"] == REDACTED_VALUE
        assert content[2]["file"]["file_data"] == REDACTED_VALUE

    def test_oversized_media_externalized_with_uploader(self) -> None:
        uploader = _StaticUploader()
        result = redact_media_from_request_parameters(
            self._chat_params(),
            hide_input_audio=False,
            hide_input_files=False,
            base64_media_max_length=100,
            blob_uploader=uploader,
        )
        content = result["messages"][0]["content"]
        assert content[1]["input_audio"]["data"].startswith("memory://")
        assert content[2]["file"]["file_data"].startswith("memory://")
        assert {b.mime_type for b in uploader.blobs} == {"audio/wav", "application/pdf"}

    def test_small_media_kept_inline(self) -> None:
        result = redact_media_from_request_parameters(
            self._chat_params(),
            hide_input_audio=False,
            hide_input_files=False,
            base64_media_max_length=10_000_000,
        )
        content = result["messages"][0]["content"]
        assert content[1]["input_audio"]["data"] == WAV_B64

    def test_original_params_not_mutated(self) -> None:
        params = self._chat_params()
        redact_media_from_request_parameters(
            params,
            hide_input_audio=True,
            hide_input_files=True,
            base64_media_max_length=100,
        )
        assert params["messages"][0]["content"][1]["input_audio"]["data"] == WAV_B64

    def test_responses_api_input_file(self) -> None:
        params = {
            "model": "gpt-4o",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_file", "file_data": PDF_B64, "filename": "report.pdf"},
                    ],
                }
            ],
        }
        result = redact_media_from_request_parameters(
            params,
            hide_input_audio=False,
            hide_input_files=False,
            base64_media_max_length=100,
        )
        assert result["input"][0]["content"][0]["file_data"] == REDACTED_VALUE

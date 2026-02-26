import logging

import pytest

from openinference.instrumentation.google_genai._request_attributes_extractor import (
    _RequestAttributesExtractor,
)
from openinference.semconv.trace import MessageAttributes, SpanAttributes


def test_request_extractor_accepts_dict_content(caplog: pytest.LogCaptureFixture) -> None:
    extractor = _RequestAttributesExtractor()

    request_parameters = {
        "contents": [
            {"role": "user", "parts": [{"text": "hello"}]},
        ]
    }

    caplog.set_level(logging.DEBUG)
    attrs = dict(extractor.get_extra_attributes_from_request(request_parameters))

    assert (
        attrs[f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}"] == "user"
    )
    assert (
        attrs[f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}"]
        == "hello"
    )

    # No noisy "Unexpected input contents type" logs for valid dict payloads
    assert not any(
        "Unexpected input contents type" in record.getMessage() for record in caplog.records
    )


def test_request_extractor_accepts_dict_part(caplog: pytest.LogCaptureFixture) -> None:
    extractor = _RequestAttributesExtractor()

    request_parameters = {
        "contents": [
            {"text": "just a part dict"},
        ]
    }

    caplog.set_level(logging.DEBUG)
    attrs = dict(extractor.get_extra_attributes_from_request(request_parameters))

    assert (
        attrs[f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}"]
        == "just a part dict"
    )

    assert not any(
        "Unexpected input contents type" in record.getMessage() for record in caplog.records
    )

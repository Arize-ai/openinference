from typing import Any

from openinference.semconv.trace import OpenInferenceMimeTypeValues

from openinference.instrumentation.agno._workflow_wrapper import _extract_output

JSON = OpenInferenceMimeTypeValues.JSON.value
TEXT = OpenInferenceMimeTypeValues.TEXT.value


class _Response:
    def __init__(self, content: Any) -> None:
        self.content = content


class _PydanticLikeContent:
    def model_dump_json(self) -> str:
        return '{"answer":"ok"}'


class _BrokenPydanticLikeContent:
    def model_dump_json(self) -> str:
        raise ValueError("boom")

    def __str__(self) -> str:
        return "fallback"


def test_extract_output_serializes_pydantic_like_response_content() -> None:
    assert _extract_output(_Response(_PydanticLikeContent())) == ('{"answer":"ok"}', JSON)


def test_extract_output_falls_back_to_string_for_content_dump_errors() -> None:
    assert _extract_output(_Response(_BrokenPydanticLikeContent())) == ("fallback", TEXT)

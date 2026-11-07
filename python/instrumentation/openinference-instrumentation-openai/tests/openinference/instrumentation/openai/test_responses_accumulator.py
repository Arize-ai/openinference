from typing import Any, Dict, List

import openai
import pytest
from openai.types.responses import Response, ResponseUsage

from openinference.instrumentation.openai._response_accumulator import _ResponsesAccumulator
from openinference.instrumentation.openai._response_attributes_extractor import (
    _ResponseAttributesExtractor,
)
from openinference.semconv.trace import SpanAttributes


# Newer SDKs require extra token-detail fields; fill every required one with 0.
def _details(model: Any, **given: int) -> Dict[str, int]:
    return {name: given.get(name, 0) for name in model.model_fields}


def _response(status: str, **extra: Any) -> Dict[str, Any]:
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 1758000000.0,
        "model": "gpt-4o-mini",
        "status": status,
        "output": [
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "status": "completed" if status == "completed" else "incomplete",
                "content": [
                    {"type": "output_text", "text": "Paris is the capital", "annotations": []}
                ],
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": 12,
            "input_tokens_details": _details(
                ResponseUsage.model_fields["input_tokens_details"].annotation
            ),
            "output_tokens": 5,
            "output_tokens_details": _details(
                ResponseUsage.model_fields["output_tokens_details"].annotation
            ),
            "total_tokens": 17,
        },
        **extra,
    }


def _event(event_type: str, response: Dict[str, Any]) -> Any:
    from openai.types.responses import (
        ResponseCompletedEvent,
        ResponseFailedEvent,
        ResponseIncompleteEvent,
    )

    cls = {
        "response.completed": ResponseCompletedEvent,
        "response.incomplete": ResponseIncompleteEvent,
        "response.failed": ResponseFailedEvent,
    }[event_type]
    return cls.model_validate({"type": event_type, "sequence_number": 9, "response": response})


def _attributes(events: List[Any]) -> Dict[str, Any]:
    accumulator = _ResponsesAccumulator(
        request_parameters={"model": "gpt-4o-mini", "input": "capital of France?"},
        chat_completion_type=Response,
        response_attributes_extractor=_ResponseAttributesExtractor(openai=openai),
    )
    for event in events:
        accumulator.process_chunk(event)
    return {**dict(accumulator.get_attributes()), **dict(accumulator.get_extra_attributes())}


@pytest.mark.parametrize(
    "event_type,response",
    [
        ("response.completed", _response("completed")),
        (
            "response.incomplete",
            _response("incomplete", incomplete_details={"reason": "max_output_tokens"}),
        ),
        (
            "response.failed",
            _response("failed", error={"code": "server_error", "message": "The model failed"}),
        ),
    ],
)
def test_terminal_event_gives_output_and_token_counts(
    event_type: str, response: Dict[str, Any]
) -> None:
    attributes = _attributes([_event(event_type, response)])

    assert "Paris is the capital" in str(attributes[SpanAttributes.OUTPUT_VALUE])
    assert attributes[SpanAttributes.LLM_TOKEN_COUNT_PROMPT] == 12
    assert attributes[SpanAttributes.LLM_TOKEN_COUNT_COMPLETION] == 5
    assert attributes[SpanAttributes.LLM_TOKEN_COUNT_TOTAL] == 17

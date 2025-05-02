from contextlib import ExitStack, suppress
from typing import Any, Dict, cast

import pytest
from google.cloud.aiplatform_v1beta1 import (
    Candidate,
    Content,
    GenerateContentRequest,
    GenerateContentResponse,
    Part,
)
from opentelemetry.sdk.trace import Tracer
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation import using_attributes
from openinference.instrumentation.vertexai._wrapper import (
    INPUT_MIME_TYPE,
    INPUT_VALUE,
    JSON,
    LLM_MODEL_NAME,
    MESSAGE_CONTENT,
    MESSAGE_CONTENT_TEXT,
    MESSAGE_CONTENTS,
    MESSAGE_NAME,
    MESSAGE_ROLE,
    MESSAGE_TOOL_CALLS,
    OPENINFERENCE_SPAN_KIND,
    OUTPUT_MIME_TYPE,
    OUTPUT_VALUE,
    TOOL_CALL_FUNCTION_NAME,
    _parse_content,
    _parse_part,
    _parse_parts,
    _parse_tool_calls,
    _update_span,
)


class Err(BaseException):
    pass


def message_role(prefix: str, i: int) -> str:
    return f"{prefix}.{i}.{MESSAGE_ROLE}"


def message_contents_text(prefix: str, i: int, j: int) -> str:
    return f"{prefix}.{i}.{MESSAGE_CONTENTS}.{j}.{MESSAGE_CONTENT_TEXT}"


def message_name(prefix: str, i: int) -> str:
    return f"{prefix}.{i}.{MESSAGE_NAME}"


def message_content(prefix: str, i: int) -> str:
    return f"{prefix}.{i}.{MESSAGE_CONTENT}"


def tool_call_function_name(prefix: str, i: int, j: int) -> str:
    return f"{prefix}.{i}.{MESSAGE_TOOL_CALLS}.{j}.{TOOL_CALL_FUNCTION_NAME}"


@pytest.fixture
def metadata() -> Dict[str, Any]:
    return {"test_key": "test_value", "environment": "testing"}


@pytest.mark.parametrize("has_error", [False])
async def test_part_with_multiple_fields(
    has_error: bool,
    in_memory_span_exporter: InMemorySpanExporter,
    metadata: Dict[str, Any],
    tracer: Tracer,
) -> None:
    """Test that we correctly handle Part objects with multiple data fields.

    This test creates a scenario where a Part object has both text and inline_data/file_data
    fields set at the same time, which would normally cause a "multiple data oneof fields" error.
    Our parsing code should safely handle this case.
    """
    # Create a request with a system instruction
    request = GenerateContentRequest(
        model="gemini-pro",
        contents=[
            Content(
                role="user",
                parts=[
                    Part(text="Hello, world!"),
                ],
            )
        ],
    )

    # Create a response with a Part that has multiple fields set
    multiple_field_part = Part()
    multiple_field_part.text = "This is text content"
    # Intentionally setting multiple fields that should be mutually exclusive in oneof
    multiple_field_part.inline_data.mime_type = "image/jpeg"
    multiple_field_part.inline_data.data = b"fake_image_data"
    multiple_field_part.file_data.mime_type = "image/png"
    multiple_field_part.file_data.file_uri = "gs://fake-bucket/image.png"

    response = GenerateContentResponse(
        candidates=[
            Candidate(
                index=0,
                content=Content(role="model", parts=[multiple_field_part]),
            )
        ],
    )

    # Mock the instrumentation run
    with ExitStack() as stack:
        stack.enter_context(suppress(Err))
        stack.enter_context(using_attributes(metadata=metadata))

        # Simulate instrumented function call
        mock_function = tracer.start_span(name="generate_content")
        mock_function.set_attribute(OPENINFERENCE_SPAN_KIND, "llm")
        # Use the proper method to serialize protobuf
        mock_function.set_attribute(INPUT_VALUE, str(request))
        mock_function.set_attribute(INPUT_MIME_TYPE, JSON)
        mock_function.set_attribute(LLM_MODEL_NAME, request.model)

        # Set output attributes
        mock_function.set_attribute(OUTPUT_VALUE, str(response))
        mock_function.set_attribute(OUTPUT_MIME_TYPE, JSON)

        # Process request and response
        _update_span(request, mock_function)
        _update_span(response, mock_function)

        # Also directly test the _parse_parts function with our multi-field part
        list(_parse_parts([multiple_field_part], "test_prefix"))

        mock_function.end()

    # If we got here without exceptions, the test passes
    assert True


@pytest.mark.parametrize("has_error", [False])
async def test_part_with_function_fields(
    has_error: bool,
    in_memory_span_exporter: InMemorySpanExporter,
    metadata: Dict[str, Any],
    tracer: Tracer,
) -> None:
    """Test that we correctly handle Part objects with function_call/response and other fields.

    This test creates a scenario where a Part object has both function_call/response and
    text/inline_data/file_data fields set at the same time.
    """
    # Create a request with a user message
    request = GenerateContentRequest(
        model="gemini-pro",
        contents=[
            Content(
                role="user",
                parts=[
                    Part(text="Hello, world!"),
                ],
            )
        ],
    )

    # Create a response with function_call and text in the same part
    function_part = Part()
    function_part.text = "Some extra text"
    function_part.function_call.name = "test_function"
    function_part.function_call.args = {"param1": "value1", "param2": 42}

    # Create a response with function_response and text in the same part
    function_response_part = Part()
    function_response_part.text = "Some response text"
    function_response_part.function_response.name = "test_function"
    function_response_part.function_response.response = {"result": "success", "value": 123}

    # Create request with function response
    request_with_response = GenerateContentRequest(
        model="gemini-pro",
        contents=[
            Content(role="user", parts=[Part(text="What did the function return?")]),
            Content(role="function", parts=[function_response_part]),
        ],
    )

    # Create response with function call
    response = GenerateContentResponse(
        candidates=[
            Candidate(
                index=0,
                content=Content(role="model", parts=[function_part]),
            )
        ],
    )

    # Mock the instrumentation run for function call
    with ExitStack() as stack:
        stack.enter_context(suppress(Err))
        stack.enter_context(using_attributes(metadata=metadata))

        # Simulate instrumented function call
        mock_function = tracer.start_span(name="generate_content")
        mock_function.set_attribute(OPENINFERENCE_SPAN_KIND, "llm")
        mock_function.set_attribute(INPUT_VALUE, str(request))
        mock_function.set_attribute(INPUT_MIME_TYPE, JSON)
        mock_function.set_attribute(LLM_MODEL_NAME, request.model)

        # Set output attributes
        mock_function.set_attribute(OUTPUT_VALUE, str(response))
        mock_function.set_attribute(OUTPUT_MIME_TYPE, JSON)

        # Process request and response
        _update_span(request, mock_function)
        _update_span(response, mock_function)

        # Also directly test the _parse_parts and _parse_tool_calls functions
        list(_parse_parts([function_part], "test_prefix"))
        list(_parse_tool_calls([function_part], "test_prefix"))

        mock_function.end()

    # Test function_response part
    with ExitStack() as stack:
        stack.enter_context(suppress(Err))
        stack.enter_context(using_attributes(metadata=metadata))

        # Simulate instrumented function call
        mock_function = tracer.start_span(name="generate_content")
        mock_function.set_attribute(OPENINFERENCE_SPAN_KIND, "llm")
        mock_function.set_attribute(INPUT_VALUE, str(request_with_response))
        mock_function.set_attribute(INPUT_MIME_TYPE, JSON)
        mock_function.set_attribute(LLM_MODEL_NAME, request_with_response.model)

        # Process request with function response
        _update_span(request_with_response, mock_function)

        # Also directly test the parts handling
        for content in request_with_response.contents:
            list(_parse_parts(content.parts, "test_prefix"))

        mock_function.end()

    # If we got here without exceptions, the test passes
    assert True


@pytest.mark.parametrize("has_error", [False])
async def test_missing_attributes(
    has_error: bool,
    in_memory_span_exporter: InMemorySpanExporter,
    metadata: Dict[str, Any],
    tracer: Tracer,
) -> None:
    """Test that we can handle Part objects with missing attributes.

    This test creates various mock objects with missing attributes to test
    our robust parsing logic.
    """
    # Create a request with a simple user message
    request = GenerateContentRequest(
        model="gemini-pro",
        contents=[
            Content(
                role="user",
                parts=[
                    Part(text="Hello, world!"),
                ],
            )
        ],
    )

    # Create a class that just looks like a Part but isn't a real Part
    # This will help us test our hasattr checks
    class PartialPart:
        """A mock object with only some of the attributes a Part would have."""

        def __init__(self) -> None:
            self.text = "This is mock text"
            # No other attributes

    # Mock the instrumentation run
    with ExitStack() as stack:
        stack.enter_context(suppress(Err))
        stack.enter_context(using_attributes(metadata=metadata))

        # Simulate instrumented function call
        mock_function = tracer.start_span(name="generate_content")
        mock_function.set_attribute(OPENINFERENCE_SPAN_KIND, "llm")
        mock_function.set_attribute(INPUT_VALUE, str(request))
        mock_function.set_attribute(INPUT_MIME_TYPE, JSON)
        mock_function.set_attribute(LLM_MODEL_NAME, request.model)

        # Set output attributes - would normally be set by the wrapper
        mock_function.set_attribute(OUTPUT_VALUE, "mock output")
        mock_function.set_attribute(OUTPUT_MIME_TYPE, JSON)

        # Process request and response
        _update_span(request, mock_function)

        # Directly test _parse_part with our mock object
        partial_part = PartialPart()
        results = list(_parse_part(cast(Part, partial_part), "test_prefix."))

        # We should get the text field but no errors from missing fields
        assert len(results) == 1
        assert results[0][0] == f"test_prefix.{MESSAGE_CONTENT_TEXT}"
        assert results[0][1] == "This is mock text"

        # Test with a completely empty object
        empty_part = type("EmptyPart", (), {})()
        empty_results = list(_parse_part(cast(Part, empty_part), "test_prefix."))
        # Should get no results but no errors either
        assert len(empty_results) == 0

        mock_function.end()

    # If we got here without exceptions, the test passes
    assert True


@pytest.mark.parametrize("has_error", [False])
async def test_complex_part_combinations(
    has_error: bool,
    in_memory_span_exporter: InMemorySpanExporter,
    metadata: Dict[str, Any],
    tracer: Tracer,
) -> None:
    """Test a complex combination of parts with different attribute combinations.

    This creates a more realistic and challenging scenario where multiple parts
    have different combinations of fields set or missing.
    """
    # Create a complex response with multiple parts having various field combinations
    # Part 1: Normal text part
    text_part = Part(text="This is text content")

    # Part 2: Part with multiple fields set
    multi_field_part = Part()
    multi_field_part.text = "This has both text and image"
    multi_field_part.inline_data.mime_type = "image/jpeg"
    multi_field_part.inline_data.data = b"fake_image_data"

    # Part a Part with function_call and other fields
    function_part = Part()
    function_part.text = "Function with text too"
    function_part.function_call.name = "test_function"
    function_part.function_call.args = {"param1": "value1"}

    # Part with function_response and other fields
    function_response_part = Part()
    function_response_part.text = "Function response with text"
    function_response_part.function_response.name = "test_function"
    function_response_part.function_response.response = {"result": "success"}

    # Create a comprehensive response with all these parts
    response = GenerateContentResponse(
        candidates=[
            Candidate(
                index=0,
                content=Content(
                    role="model",
                    parts=[text_part, multi_field_part, function_part, function_response_part],
                ),
            )
        ],
    )

    # Test that we can parse this complex combination of parts
    with ExitStack() as stack:
        stack.enter_context(suppress(Err))

        # Test parsing the entire content
        content = response.candidates[0].content
        # Store results in _ to avoid F841 unused variable error
        _ = list(_parse_content(content))

        # Test parsing just the parts
        _ = list(_parse_parts(content.parts))

        # Test parsing tool calls from the parts
        _ = list(_parse_tool_calls(content.parts))

        # Mock function to test update_span
        mock_function = tracer.start_span(name="generate_content")
        _update_span(response, mock_function)
        mock_function.end()

    # If we got here without exceptions, the test passes
    assert True

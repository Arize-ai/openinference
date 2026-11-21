"""Input extraction must not break the call being instrumented."""

from types import SimpleNamespace

import pytest

from openinference.instrumentation.agno._workflow_wrapper import (
    _get_input_from_args,
)


@pytest.mark.parametrize(
    "arguments",
    [
        {"step_input": SimpleNamespace(input={"pdf_source": b"\xf2\xad"})},
        {"step_input": SimpleNamespace(get_last_step_content=lambda: {"pdf_source": b"\xf2\xad"})},
        {"execution_input": SimpleNamespace(input={"pdf_source": b"\xf2\xad"})},
    ],
)
def test_bytes_in_input_is_serialized_without_raising(arguments) -> None:
    result = _get_input_from_args(arguments)

    assert "pdf_source" in result
    assert "b'" in result

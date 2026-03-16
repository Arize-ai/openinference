import json
from pathlib import Path
from typing import Any

from openinference.instrumentation.harbor._converter import (
    convert_trajectory,
    convert_trajectory_file,
)
from openinference.semconv.trace import SpanAttributes

FIXTURES_DIR = Path(__file__).parent / "fixtures"

JSON = "application/json"
TEXT = "text/plain"


class TestConvertTrajectory:
    def test_basic_trajectory_span_count(self, sample_trajectory: dict[str, Any]) -> None:
        """Fixture has 2 non-empty agent steps + 1 tool call: 1 root + 2 LLM + 1 TOOL = 4."""
        spans = convert_trajectory(sample_trajectory)
        assert len(spans) == 4

    def test_root_span_attributes_exhaustive(self, sample_trajectory: dict[str, Any]) -> None:
        """Root AGENT span has exactly the expected attributes, nothing extra."""
        spans = convert_trajectory(sample_trajectory)
        attrs = dict(spans[0].attributes or {})

        assert attrs.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND) == "AGENT"
        assert attrs.pop(SpanAttributes.AGENT_NAME) == "research_agent"
        assert attrs.pop(SpanAttributes.SESSION_ID) == "test-session-abc123"
        assert attrs.pop(SpanAttributes.LLM_MODEL_NAME) == "gpt-4o"
        assert "climate change impacts" in str(attrs.pop(SpanAttributes.INPUT_VALUE))
        assert attrs.pop(SpanAttributes.INPUT_MIME_TYPE) == TEXT
        assert "rising sea levels" in str(attrs.pop(SpanAttributes.OUTPUT_VALUE))
        assert attrs.pop(SpanAttributes.OUTPUT_MIME_TYPE) == TEXT
        assert attrs.pop(SpanAttributes.LLM_TOKEN_COUNT_PROMPT) == 350
        assert attrs.pop(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION) == 75
        assert attrs.pop(SpanAttributes.LLM_TOKEN_COUNT_TOTAL) == 425
        assert not attrs

    def test_llm_span_attributes_exhaustive(self, sample_trajectory: dict[str, Any]) -> None:
        """First LLM span has exactly the expected attributes."""
        spans = convert_trajectory(sample_trajectory)
        attrs = dict(spans[1].attributes or {})

        assert spans[1].name == "step 1"
        assert attrs.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND) == "LLM"
        assert attrs.pop(SpanAttributes.LLM_MODEL_NAME) == "gpt-4o"
        assert attrs.pop(SpanAttributes.LLM_TOKEN_COUNT_PROMPT) == 150
        assert attrs.pop(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION) == 25
        assert attrs.pop(SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ) == 50

        # Input: system + user messages
        input_messages = json.loads(str(attrs.pop(SpanAttributes.LLM_INPUT_MESSAGES)))
        assert len(input_messages) == 2
        assert input_messages[0]["message"]["role"] == "system"
        assert input_messages[1]["message"]["role"] == "user"

        output_messages = json.loads(str(attrs.pop(SpanAttributes.LLM_OUTPUT_MESSAGES)))
        assert len(output_messages) == 1
        assert output_messages[0]["message"]["role"] == "assistant"

        assert "climate change" in str(attrs.pop(SpanAttributes.INPUT_VALUE))
        assert "search for information" in str(attrs.pop(SpanAttributes.OUTPUT_VALUE))
        assert attrs.pop(SpanAttributes.INPUT_MIME_TYPE) == TEXT
        assert attrs.pop(SpanAttributes.OUTPUT_MIME_TYPE) == TEXT
        assert not attrs

    def test_tool_span_attributes_exhaustive(self, sample_trajectory: dict[str, Any]) -> None:
        """TOOL span has exactly the expected attributes."""
        spans = convert_trajectory(sample_trajectory)
        attrs = dict(spans[2].attributes or {})

        assert spans[2].name == "web_search"
        assert attrs.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND) == "TOOL"
        assert attrs.pop(SpanAttributes.TOOL_NAME) == "web_search"

        params = json.loads(str(attrs.pop(SpanAttributes.TOOL_PARAMETERS)))
        assert params["query"] == "climate change impacts 2024"

        input_val = json.loads(str(attrs.pop(SpanAttributes.INPUT_VALUE)))
        assert input_val["query"] == "climate change impacts 2024"
        assert attrs.pop(SpanAttributes.INPUT_MIME_TYPE) == JSON

        assert "rising sea levels" in str(attrs.pop(SpanAttributes.OUTPUT_VALUE))
        assert attrs.pop(SpanAttributes.OUTPUT_MIME_TYPE) == TEXT
        assert not attrs

    def test_second_llm_span_has_conversation_context(
        self, sample_trajectory: dict[str, Any]
    ) -> None:
        """Second LLM span input includes prior agent output in context."""
        spans = convert_trajectory(sample_trajectory)
        attrs = dict(spans[3].attributes or {})
        input_val = str(attrs[SpanAttributes.INPUT_VALUE])
        assert "search for information" in input_val

    def test_deterministic_ids(self, sample_trajectory: dict[str, Any]) -> None:
        spans1 = convert_trajectory(sample_trajectory)
        spans2 = convert_trajectory(sample_trajectory)
        for s1, s2 in zip(spans1, spans2):
            assert s1.context.trace_id == s2.context.trace_id
            assert s1.context.span_id == s2.context.span_id

    def test_all_spans_share_trace_id(self, sample_trajectory: dict[str, Any]) -> None:
        spans = convert_trajectory(sample_trajectory)
        trace_ids = {s.context.trace_id for s in spans}
        assert len(trace_ids) == 1

    def test_span_parent_relationships(self, sample_trajectory: dict[str, Any]) -> None:
        spans = convert_trajectory(sample_trajectory)
        root, llm1, tool, llm2 = spans

        assert root.parent is None
        assert llm1.parent is not None
        assert llm1.parent.span_id == root.context.span_id
        assert tool.parent is not None
        assert tool.parent.span_id == llm1.context.span_id
        assert llm2.parent is not None
        assert llm2.parent.span_id == root.context.span_id

    def test_empty_agent_steps_skipped(self) -> None:
        trajectory: dict[str, Any] = {
            "session_id": "skip-test",
            "agent": {"name": "test_agent", "model_name": "gpt-4"},
            "steps": [
                {"role": "user", "message": "hello"},
                {"role": "assistant", "message": "", "metrics": {}},
                {"role": "assistant", "message": "hi there!", "metrics": {}},
            ],
            "final_metrics": {},
        }
        spans = convert_trajectory(trajectory)
        assert len(spans) == 2
        assert spans[1].name == "step 1"
        attrs = dict(spans[1].attributes or {})
        assert attrs[SpanAttributes.OUTPUT_VALUE] == "hi there!"

    def test_multiple_turns(self) -> None:
        trajectory: dict[str, Any] = {
            "session_id": "multi-turn-test",
            "agent": {"name": "test_agent", "model_name": "gpt-4"},
            "steps": [
                {"role": "user", "message": "hello"},
                {"role": "assistant", "message": "hi there", "metrics": {}},
                {"role": "user", "message": "how are you"},
                {"role": "assistant", "message": "I'm great", "metrics": {}},
            ],
            "final_metrics": {},
        }
        spans = convert_trajectory(trajectory)
        assert len(spans) == 3
        assert spans[1].name == "step 1"
        assert spans[2].name == "step 2"

        attrs1 = dict(spans[1].attributes or {})
        assert "hello" in str(attrs1[SpanAttributes.INPUT_VALUE])
        assert "hi there" in str(attrs1[SpanAttributes.OUTPUT_VALUE])

        attrs2 = dict(spans[2].attributes or {})
        assert "how are you" in str(attrs2[SpanAttributes.INPUT_VALUE])
        assert "I'm great" in str(attrs2[SpanAttributes.OUTPUT_VALUE])

    def test_per_step_model_override(self) -> None:
        trajectory: dict[str, Any] = {
            "session_id": "override-test",
            "agent": {"name": "test_agent", "model_name": "gpt-4"},
            "steps": [
                {"role": "user", "message": "hello"},
                {
                    "role": "assistant",
                    "message": "hi",
                    "model_name": "claude-3-opus",
                    "metrics": {},
                },
            ],
            "final_metrics": {},
        }
        spans = convert_trajectory(trajectory)
        attrs = dict(spans[1].attributes or {})
        assert attrs[SpanAttributes.LLM_MODEL_NAME] == "claude-3-opus"

    def test_missing_timestamps_handled(self) -> None:
        trajectory: dict[str, Any] = {
            "session_id": "no-ts-test",
            "agent": {"name": "test_agent"},
            "steps": [
                {"role": "user", "message": "hello"},
                {"role": "assistant", "message": "hi", "metrics": {}},
            ],
            "final_metrics": {},
        }
        spans = convert_trajectory(trajectory)
        assert len(spans) == 2
        for span in spans:
            assert span.start_time is not None and span.start_time > 0
            assert span.end_time is not None and span.end_time > span.start_time

    def test_empty_steps(self) -> None:
        trajectory: dict[str, Any] = {
            "session_id": "empty-test",
            "agent": {"name": "empty_agent"},
            "steps": [],
            "final_metrics": {},
        }
        spans = convert_trajectory(trajectory)
        assert len(spans) == 1

    def test_resource_attributes(self, sample_trajectory: dict[str, Any]) -> None:
        resource_attrs = {"service.name": "harbor-test", "deployment.environment": "test"}
        spans = convert_trajectory(sample_trajectory, resource_attributes=resource_attrs)
        for span in spans:
            res_attrs = dict(span.resource.attributes)
            assert res_attrs["service.name"] == "harbor-test"
            assert res_attrs["deployment.environment"] == "test"

    def test_atif_source_field(self) -> None:
        """Real ATIF uses 'source' instead of 'role'."""
        trajectory: dict[str, Any] = {
            "session_id": "atif-test",
            "agent": {"name": "test_agent", "model_name": "gpt-4"},
            "steps": [
                {"step_id": 1, "source": "user", "message": "hello"},
                {"step_id": 2, "source": "agent", "message": "hi", "metrics": {}},
            ],
            "final_metrics": {},
        }
        spans = convert_trajectory(trajectory)
        assert len(spans) == 2
        attrs = dict(spans[1].attributes or {})
        assert attrs[SpanAttributes.INPUT_VALUE] == "user: hello"
        assert attrs[SpanAttributes.OUTPUT_VALUE] == "hi"

    def test_synthetic_model_filtered(self) -> None:
        """<synthetic> model name is not set on spans."""
        trajectory: dict[str, Any] = {
            "session_id": "synth-test",
            "agent": {"name": "test_agent", "model_name": "<synthetic>"},
            "steps": [
                {"role": "user", "message": "hello"},
                {"role": "assistant", "message": "hi", "model_name": "<synthetic>", "metrics": {}},
            ],
            "final_metrics": {},
        }
        spans = convert_trajectory(trajectory)
        attrs = dict(spans[1].attributes or {})
        assert SpanAttributes.LLM_MODEL_NAME not in attrs


class TestConvertTrajectoryFile:
    def test_convert_file(self) -> None:
        spans = convert_trajectory_file(FIXTURES_DIR / "sample_trajectory.json")
        assert len(spans) == 4
        assert spans[0].name == "research_agent trajectory"

    def test_convert_file_string_path(self) -> None:
        path = str(FIXTURES_DIR / "sample_trajectory.json")
        spans = convert_trajectory_file(path)
        assert len(spans) == 4

import gc
import json
from importlib.metadata import entry_points
from typing import Any, Dict, List, Optional, Sequence

import pytest
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from qwen_agent.agents import Assistant, Router
from qwen_agent.llm import LLM_REGISTRY, get_chat_model
from qwen_agent.llm.schema import ASSISTANT, FUNCTION, FunctionCall, Message
from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.tools.retrieval import Retrieval

from openinference.instrumentation import (
    REDACTED_VALUE,
    OITracer,
    TraceConfig,
    suppress_tracing,
    using_attributes,
)
from openinference.instrumentation.qwen_agent import QwenAgentInstrumentor, _wrappers
from openinference.semconv.trace import (
    DocumentAttributes,
    ImageAttributes,
    MessageAttributes,
    MessageContentAttributes,
    SpanAttributes,
    ToolAttributes,
    ToolCallAttributes,
)

# Registered by conftest.py; fetched from qwen-agent's registry so this module
# needs no package-relative import (which would shadow the real `qwen_agent`).
FakeChatModel = LLM_REGISTRY["oi_fake"]


@register_tool("get_weather", allow_overwrite=True)
class WeatherTool(BaseTool):  # type: ignore[misc]
    description = "Get the current weather for a city."
    parameters = {
        "type": "object",
        "properties": {"city": {"type": "string", "description": "City name"}},
        "required": ["city"],
    }

    def call(self, params: Any, **kwargs: Any) -> str:
        args = self._verify_json_format_args(params)
        return json.dumps({"city": args["city"], "temperature_c": 21})


@register_tool("boom", allow_overwrite=True)
class ExplodingTool(BaseTool):  # type: ignore[misc]
    description = "Always raises."
    parameters = {"type": "object", "properties": {}, "required": []}

    def call(self, params: Any, **kwargs: Any) -> str:
        raise RuntimeError("tool exploded")


class FakeRetrieval(Retrieval):  # type: ignore[misc]
    """A Retrieval that returns canned chunks in qwen-agent's own shape.

    `Retrieval.__init__` constructs a DocParser and a search tool, which need
    the `[rag]` extras, so it is deliberately not called.
    """

    def __init__(self) -> None:
        self.cfg: Dict[str, Any] = {}

    def call(self, params: Any, **kwargs: Any) -> List[Dict[str, Any]]:
        return [
            {"url": "/tmp/a.pdf", "text": ["chunk one", "chunk two"]},
            {"url": "/tmp/b.pdf", "text": ["other doc"]},
        ]


def _agent(**kwargs: Any) -> Assistant:
    return Assistant(
        llm={"model": "qwen-fake", "model_type": "oi_fake"},
        name=kwargs.pop("name", "TestAgent"),
        description=kwargs.pop("description", "An agent under test"),
        system_message=kwargs.pop("system_message", "You are a helpful assistant."),
        **kwargs,
    )


def _text_turn(text: str, **extra: Any) -> List[List[Message]]:
    """A scripted turn that streams `text` in two cumulative chunks."""
    half = max(1, len(text) // 2)
    return [
        [Message(role=ASSISTANT, content=text[:half], **extra)],
        [Message(role=ASSISTANT, content=text, **extra)],
    ]


def _tool_call_turn(name: str, arguments: Dict[str, Any]) -> List[List[Message]]:
    return [
        [
            Message(role=ASSISTANT, content="Looking that up."),
            Message(
                role=ASSISTANT,
                content="",
                function_call=FunctionCall(name=name, arguments=json.dumps(arguments)),
                extra={"function_id": "call_1"},
            ),
        ]
    ]


def _spans_by_kind(exporter: InMemorySpanExporter, kind: str) -> List[ReadableSpan]:
    return [
        span
        for span in exporter.get_finished_spans()
        if (span.attributes or {}).get(SpanAttributes.OPENINFERENCE_SPAN_KIND) == kind
    ]


def _attrs(span: ReadableSpan) -> Dict[str, Any]:
    return dict(span.attributes or {})


def _run(agent: Assistant, content: str = "what is the weather in Paris?") -> List[Any]:
    return list(agent.run([{"role": "user", "content": content}]))


class TestInstrumentorSetup:
    def test_entrypoints_are_registered(self) -> None:
        for group in ("opentelemetry_instrumentor", "openinference_instrumentor"):
            (entrypoint,) = [
                candidate
                for candidate in entry_points(group=group)
                if candidate.name == "qwen_agent"
            ]
            assert isinstance(entrypoint.load()(), QwenAgentInstrumentor)

    def test_instrumentation_dependencies(self) -> None:
        assert tuple(QwenAgentInstrumentor().instrumentation_dependencies()) == (
            "qwen-agent >= 0.0.20, < 1",
        )

    def test_uses_oitracer(self) -> None:
        assert isinstance(QwenAgentInstrumentor().tracer, OITracer)

    def test_uninstrument_restores_originals(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
        tracer_provider: Any,
    ) -> None:
        import qwen_agent.agent
        import qwen_agent.llm.base
        import qwen_agent.utils.parallel_executor

        # The autouse fixture has already wrapped everything, so unwrap first to
        # capture the genuine originals.
        QwenAgentInstrumentor().uninstrument()
        original_run = qwen_agent.agent.Agent.run
        original_call_tool = qwen_agent.agent.Agent._call_tool
        original_chat = qwen_agent.llm.base.BaseChatModel.chat
        original_executor = qwen_agent.utils.parallel_executor.ThreadPoolExecutor

        QwenAgentInstrumentor().instrument(tracer_provider=tracer_provider, skip_dep_check=True)
        assert qwen_agent.agent.Agent.run is not original_run
        assert qwen_agent.utils.parallel_executor.ThreadPoolExecutor is not original_executor
        QwenAgentInstrumentor().uninstrument()

        assert qwen_agent.agent.Agent.run is original_run
        assert qwen_agent.agent.Agent._call_tool is original_call_tool
        assert qwen_agent.llm.base.BaseChatModel.chat is original_chat
        assert qwen_agent.utils.parallel_executor.ThreadPoolExecutor is original_executor

        FakeChatModel.configure([_text_turn("Hello!")])
        _run(_agent(), "hi")
        assert not in_memory_span_exporter.get_finished_spans()


class TestAgentSpans:
    def test_agent_run_records_input_and_output(
        self, in_memory_span_exporter: InMemorySpanExporter
    ) -> None:
        FakeChatModel.configure([_text_turn("It is sunny in Paris.")])
        _run(_agent())

        (span,) = _spans_by_kind(in_memory_span_exporter, "AGENT")
        assert span.name == "TestAgent.run"
        assert span.status.status_code == StatusCode.OK
        attributes = _attrs(span)
        assert attributes[SpanAttributes.AGENT_NAME] == "TestAgent"
        assert attributes["qwen_agent.agent.description"] == "An agent under test"
        assert attributes[SpanAttributes.LLM_MODEL_NAME] == "qwen-fake"
        assert attributes["qwen_agent.llm.model_type"] == "oi_fake"
        assert attributes[SpanAttributes.OUTPUT_VALUE] == "It is sunny in Paris."
        recorded_input = json.loads(attributes[SpanAttributes.INPUT_VALUE])
        assert recorded_input[-1]["content"] == "what is the weather in Paris?"

    def test_agent_span_carries_no_token_counts(
        self, in_memory_span_exporter: InMemorySpanExporter
    ) -> None:
        """Token counts belong on LLM spans only.

        Trace-level totals are summed across every span, so counting the same
        tokens on an AGENT span as well would double them.
        """
        usage = {"usage": {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18}}
        FakeChatModel.configure([_text_turn("Sunny.", extra={"model_service_info": usage})])
        _run(_agent())

        (agent_span,) = _spans_by_kind(in_memory_span_exporter, "AGENT")
        for attribute in (
            SpanAttributes.LLM_TOKEN_COUNT_PROMPT,
            SpanAttributes.LLM_TOKEN_COUNT_COMPLETION,
            SpanAttributes.LLM_TOKEN_COUNT_TOTAL,
        ):
            assert attribute not in _attrs(agent_span)

    def test_run_nonstream_produces_one_agent_span(
        self, in_memory_span_exporter: InMemorySpanExporter
    ) -> None:
        """run_nonstream calls run internally, so it must not double-count."""
        FakeChatModel.configure([_text_turn("Sunny.")])
        _agent().run_nonstream([{"role": "user", "content": "weather?"}])

        assert len(_spans_by_kind(in_memory_span_exporter, "AGENT")) == 1

    def test_memory_is_a_chain_span(self, in_memory_span_exporter: InMemorySpanExporter) -> None:
        """Assistant._run always runs its Memory agent to gather knowledge."""
        FakeChatModel.configure([_text_turn("Sunny.")])
        _run(_agent())

        (chain_span,) = _spans_by_kind(in_memory_span_exporter, "CHAIN")
        assert chain_span.name == "Memory.run"

    def test_abandoned_generator_still_ends_the_span(
        self, in_memory_span_exporter: InMemorySpanExporter
    ) -> None:
        FakeChatModel.configure([_text_turn("A long answer that is never drained.")])
        generator = _agent().run([{"role": "user", "content": "hi"}])
        next(generator)
        generator.close()

        agent_spans = _spans_by_kind(in_memory_span_exporter, "AGENT")
        assert len(agent_spans) == 1
        assert agent_spans[0].end_time is not None


class TestLLMSpans:
    def test_llm_span_attributes(self, in_memory_span_exporter: InMemorySpanExporter) -> None:
        FakeChatModel.configure([_text_turn("It is sunny.")])
        _run(_agent(function_list=[WeatherTool()]))

        (span,) = _spans_by_kind(in_memory_span_exporter, "LLM")
        assert span.name == "FakeChatModel.chat"
        attributes = _attrs(span)
        assert attributes[SpanAttributes.LLM_MODEL_NAME] == "qwen-fake"
        assert attributes[SpanAttributes.OUTPUT_VALUE] == "It is sunny."
        assert json.loads(attributes[SpanAttributes.LLM_INVOCATION_PARAMETERS])["lang"] == "en"

        roles = [
            value
            for key, value in attributes.items()
            if key.startswith(SpanAttributes.LLM_INPUT_MESSAGES)
            and key.endswith(MessageAttributes.MESSAGE_ROLE.split(".")[-1])
        ]
        assert "system" in roles and "user" in roles
        assert (
            attributes[
                f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}"
            ]
            == "It is sunny."
        )
        tool_schema = json.loads(
            attributes[f"{SpanAttributes.LLM_TOOLS}.0.{ToolAttributes.TOOL_JSON_SCHEMA}"]
        )
        assert tool_schema["name"] == "get_weather"

    def test_tool_call_is_recorded_on_the_output_message(
        self, in_memory_span_exporter: InMemorySpanExporter
    ) -> None:
        FakeChatModel.configure(
            [_tool_call_turn("get_weather", {"city": "Paris"}), _text_turn("It is 21C.")]
        )
        _run(_agent(function_list=[WeatherTool()]))

        first_llm_span = _spans_by_kind(in_memory_span_exporter, "LLM")[0]
        attributes = _attrs(first_llm_span)
        prefix = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.1.{MessageAttributes.MESSAGE_TOOL_CALLS}.0"
        assert attributes[f"{prefix}.{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}"] == "get_weather"
        assert json.loads(
            attributes[f"{prefix}.{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}"]
        ) == {"city": "Paris"}
        assert attributes[f"{prefix}.{ToolCallAttributes.TOOL_CALL_ID}"] == "call_1"

    def test_tool_result_message_is_mapped_to_the_tool_role(
        self, in_memory_span_exporter: InMemorySpanExporter
    ) -> None:
        """qwen-agent uses role="function"; OpenInference expects "tool"."""
        FakeChatModel.configure(
            [_tool_call_turn("get_weather", {"city": "Paris"}), _text_turn("It is 21C.")]
        )
        _run(_agent(function_list=[WeatherTool()]))

        second_llm_span = _spans_by_kind(in_memory_span_exporter, "LLM")[1]
        attributes = _attrs(second_llm_span)
        tool_roles = [
            index
            for index in range(10)
            if attributes.get(
                f"{SpanAttributes.LLM_INPUT_MESSAGES}.{index}.{MessageAttributes.MESSAGE_ROLE}"
            )
            == "tool"
        ]
        assert tool_roles, "no tool-role input message found"
        index = tool_roles[0]
        assert (
            attributes[
                f"{SpanAttributes.LLM_INPUT_MESSAGES}.{index}."
                f"{MessageAttributes.MESSAGE_TOOL_CALL_ID}"
            ]
            == "call_1"
        )

    def test_reasoning_content_is_recorded(
        self, in_memory_span_exporter: InMemorySpanExporter
    ) -> None:
        FakeChatModel.configure(
            [[[Message(role=ASSISTANT, content="42", reasoning_content="thinking hard")]]]
        )
        _run(_agent())

        (span,) = _spans_by_kind(in_memory_span_exporter, "LLM")
        attributes = _attrs(span)
        prefix = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENTS}.0"
        content_type = MessageContentAttributes.MESSAGE_CONTENT_TYPE
        assert attributes[f"{prefix}.{content_type}"] == "reasoning"

    def test_non_streaming_chat_is_traced(
        self, in_memory_span_exporter: InMemorySpanExporter
    ) -> None:
        """chat(stream=False) returns a list rather than an iterator."""
        FakeChatModel.configure([_text_turn("Sunny.")])
        model = get_chat_model({"model": "qwen-fake", "model_type": "oi_fake"})
        result = model.chat(messages=[Message(role="user", content="weather?")], stream=False)

        assert isinstance(result, list)
        (span,) = _spans_by_kind(in_memory_span_exporter, "LLM")
        assert _attrs(span)[SpanAttributes.OUTPUT_VALUE] == "Sunny."

    def test_chat_error_is_recorded(self, in_memory_span_exporter: InMemorySpanExporter) -> None:
        model = get_chat_model({"model": "qwen-fake", "model_type": "oi_fake"})
        with pytest.raises(ValueError):
            model.chat(messages=[])

        (span,) = _spans_by_kind(in_memory_span_exporter, "LLM")
        assert span.status.status_code == StatusCode.ERROR
        assert span.events


class TestTokenCounts:
    """Token counts are recorded only where qwen-agent exposes them.

    The DashScope backends stash the raw response on
    ``Message.extra["model_service_info"]``; the OpenAI-compatible backends drop
    it, and on those the nested OpenAI-SDK span carries the counts instead.
    """

    @pytest.mark.parametrize(
        "usage,expected",
        [
            pytest.param(
                {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18},
                (11, 7, 18),
                id="dashscope-keys",
            ),
            pytest.param(
                {"prompt_tokens": 3, "completion_tokens": 5},
                (3, 5, 8),
                id="openai-keys-total-derived",
            ),
        ],
    )
    def test_usage_is_recorded_when_present(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
        usage: Dict[str, int],
        expected: Sequence[int],
    ) -> None:
        FakeChatModel.configure(
            [_text_turn("Sunny.", extra={"model_service_info": {"usage": usage}})]
        )
        _run(_agent())

        (span,) = _spans_by_kind(in_memory_span_exporter, "LLM")
        attributes = _attrs(span)
        assert attributes[SpanAttributes.LLM_TOKEN_COUNT_PROMPT] == expected[0]
        assert attributes[SpanAttributes.LLM_TOKEN_COUNT_COMPLETION] == expected[1]
        assert attributes[SpanAttributes.LLM_TOKEN_COUNT_TOTAL] == expected[2]

    def test_no_usage_means_no_token_attributes(
        self, in_memory_span_exporter: InMemorySpanExporter
    ) -> None:
        FakeChatModel.configure([_text_turn("Sunny.")])
        _run(_agent())

        (span,) = _spans_by_kind(in_memory_span_exporter, "LLM")
        attributes = _attrs(span)
        assert SpanAttributes.LLM_TOKEN_COUNT_PROMPT not in attributes
        assert SpanAttributes.LLM_TOKEN_COUNT_COMPLETION not in attributes
        assert SpanAttributes.LLM_TOKEN_COUNT_TOTAL not in attributes

    def test_all_zero_usage_is_ignored(self, in_memory_span_exporter: InMemorySpanExporter) -> None:
        """`quick_chat_oai` fabricates an all-zero usage block."""
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        FakeChatModel.configure(
            [_text_turn("Sunny.", extra={"model_service_info": {"usage": usage}})]
        )
        _run(_agent())

        (span,) = _spans_by_kind(in_memory_span_exporter, "LLM")
        assert SpanAttributes.LLM_TOKEN_COUNT_TOTAL not in _attrs(span)


class TestToolSpans:
    def test_tool_span_attributes(self, in_memory_span_exporter: InMemorySpanExporter) -> None:
        FakeChatModel.configure(
            [_tool_call_turn("get_weather", {"city": "Paris"}), _text_turn("It is 21C.")]
        )
        _run(_agent(function_list=[WeatherTool()]))

        (span,) = _spans_by_kind(in_memory_span_exporter, "TOOL")
        assert span.name == "get_weather.call"
        attributes = _attrs(span)
        assert attributes[SpanAttributes.TOOL_NAME] == "get_weather"
        assert attributes[SpanAttributes.TOOL_DESCRIPTION] == WeatherTool.description
        assert json.loads(attributes[SpanAttributes.TOOL_PARAMETERS])["required"] == ["city"]
        assert json.loads(attributes[SpanAttributes.INPUT_VALUE]) == {"city": "Paris"}
        assert json.loads(attributes[SpanAttributes.OUTPUT_VALUE])["temperature_c"] == 21
        assert attributes[ToolCallAttributes.TOOL_CALL_ID] == "call_1"
        assert attributes[SpanAttributes.AGENT_NAME] == "TestAgent"

    def test_tool_spans_nest_under_the_agent_span(
        self, in_memory_span_exporter: InMemorySpanExporter
    ) -> None:
        FakeChatModel.configure(
            [_tool_call_turn("get_weather", {"city": "Paris"}), _text_turn("It is 21C.")]
        )
        _run(_agent(function_list=[WeatherTool()]))

        (agent_span,) = _spans_by_kind(in_memory_span_exporter, "AGENT")
        agent_span_id = agent_span.context.span_id if agent_span.context else None
        (tool_span,) = _spans_by_kind(in_memory_span_exporter, "TOOL")
        assert tool_span.parent is not None
        assert tool_span.parent.span_id == agent_span_id
        for llm_span in _spans_by_kind(in_memory_span_exporter, "LLM"):
            assert llm_span.parent is not None
            assert llm_span.parent.span_id == agent_span_id

    def test_failing_tool_is_reported_as_output_not_error(
        self, in_memory_span_exporter: InMemorySpanExporter
    ) -> None:
        """Agent._call_tool swallows tool exceptions and returns the error text."""
        FakeChatModel.configure([_tool_call_turn("boom", {}), _text_turn("Sorry.")])
        _run(_agent(function_list=[ExplodingTool()]))

        (span,) = _spans_by_kind(in_memory_span_exporter, "TOOL")
        assert span.status.status_code == StatusCode.OK
        assert "tool exploded" in _attrs(span)[SpanAttributes.OUTPUT_VALUE]

    @pytest.mark.parametrize(
        "target,span_kind",
        [
            pytest.param("_tool_call_attributes", "TOOL", id="tool"),
            pytest.param("_chat_attributes", "LLM", id="llm"),
            pytest.param("_agent_attributes", "AGENT", id="agent"),
        ],
    )
    def test_failed_attribute_extraction_does_not_break_the_run(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
        monkeypatch: pytest.MonkeyPatch,
        target: str,
        span_kind: str,
    ) -> None:
        """Instrumentation must never raise into user code.

        If attribute extraction fails the agent still runs to completion and the
        span is still recorded — it just carries fewer attributes.
        """

        def boom(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("attribute extraction failed")

        monkeypatch.setattr(_wrappers, target, boom)
        FakeChatModel.configure(
            [_tool_call_turn("get_weather", {"city": "Paris"}), _text_turn("It is 21C.")]
        )
        responses = _run(_agent(function_list=[WeatherTool()]))

        assert responses[-1][-1]["content"] == "It is 21C."
        spans = _spans_by_kind(in_memory_span_exporter, span_kind)
        assert spans, f"no {span_kind} span was recorded"
        assert all(span.status.status_code == StatusCode.OK for span in spans)


class TestRequiredFeatures:
    def test_suppress_tracing(self, in_memory_span_exporter: InMemorySpanExporter) -> None:
        FakeChatModel.configure(
            [_tool_call_turn("get_weather", {"city": "Paris"}), _text_turn("It is 21C.")]
        )
        with suppress_tracing():
            _run(_agent(function_list=[WeatherTool()]))

        assert not in_memory_span_exporter.get_finished_spans()

    def test_context_attributes_are_propagated(
        self, in_memory_span_exporter: InMemorySpanExporter
    ) -> None:
        FakeChatModel.configure(
            [_tool_call_turn("get_weather", {"city": "Paris"}), _text_turn("It is 21C.")]
        )
        with using_attributes(
            session_id="session-1",
            user_id="user-1",
            metadata={"env": "test"},
            tags=["tag-1"],
            prompt_template="answer {question}",
            prompt_template_version="v1.0",
            prompt_template_variables={"question": "weather"},
        ):
            _run(_agent(function_list=[WeatherTool()]))

        spans = in_memory_span_exporter.get_finished_spans()
        assert spans
        for span in spans:
            attributes = _attrs(span)
            assert attributes[SpanAttributes.SESSION_ID] == "session-1"
            assert attributes[SpanAttributes.USER_ID] == "user-1"
            assert json.loads(attributes[SpanAttributes.METADATA])["env"] == "test"
            assert attributes[SpanAttributes.TAG_TAGS] == ("tag-1",)
            assert attributes[SpanAttributes.LLM_PROMPT_TEMPLATE] == "answer {question}"
            assert attributes[SpanAttributes.LLM_PROMPT_TEMPLATE_VERSION] == "v1.0"
            assert json.loads(attributes[SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES]) == {
                "question": "weather"
            }

    @pytest.mark.parametrize("hide_inputs", [True, False])
    def test_trace_config_masks_inputs(
        self,
        instrument_with_config: Any,
        in_memory_span_exporter: InMemorySpanExporter,
        hide_inputs: bool,
    ) -> None:
        instrument_with_config(TraceConfig(hide_inputs=hide_inputs))
        FakeChatModel.configure([_text_turn("Sunny.")])
        _run(_agent())

        (agent_span,) = _spans_by_kind(in_memory_span_exporter, "AGENT")
        recorded = _attrs(agent_span)[SpanAttributes.INPUT_VALUE]
        if hide_inputs:
            assert recorded == REDACTED_VALUE
        else:
            assert "what is the weather in Paris?" in recorded

    def test_trace_config_hides_llm_invocation_parameters(
        self, instrument_with_config: Any, in_memory_span_exporter: InMemorySpanExporter
    ) -> None:
        instrument_with_config(TraceConfig(hide_llm_invocation_parameters=True))
        FakeChatModel.configure([_text_turn("Sunny.")])
        _run(_agent())

        (llm_span,) = _spans_by_kind(in_memory_span_exporter, "LLM")
        assert SpanAttributes.LLM_INVOCATION_PARAMETERS not in _attrs(llm_span)


class TestMessageConversion:
    def test_multimodal_content_items(self, in_memory_span_exporter: InMemorySpanExporter) -> None:
        from qwen_agent.llm.schema import ContentItem

        FakeChatModel.configure([_text_turn("A cat.")])
        model = get_chat_model({"model": "qwen-fake", "model_type": "oi_fake"})
        model.chat(
            messages=[
                Message(
                    role="user",
                    content=[
                        ContentItem(text="what is this?"),
                        ContentItem(image="https://example.com/cat.png"),
                    ],
                )
            ],
            stream=False,
        )

        (span,) = _spans_by_kind(in_memory_span_exporter, "LLM")
        attributes = _attrs(span)
        # DEFAULT_SYSTEM_MESSAGE is empty, so chat() prepends nothing and the
        # user message stays at index 0.
        contents_prefix = (
            f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENTS}"
        )
        content_type = MessageContentAttributes.MESSAGE_CONTENT_TYPE
        content_text = MessageContentAttributes.MESSAGE_CONTENT_TEXT
        content_image = MessageContentAttributes.MESSAGE_CONTENT_IMAGE
        assert attributes[f"{contents_prefix}.0.{content_type}"] == "text"
        assert attributes[f"{contents_prefix}.0.{content_text}"] == "what is this?"
        assert attributes[f"{contents_prefix}.1.{content_type}"] == "image"
        assert (
            attributes[f"{contents_prefix}.1.{content_image}.{ImageAttributes.IMAGE_URL}"]
            == "https://example.com/cat.png"
        )

    def test_function_message_without_extra_is_tolerated(
        self, in_memory_span_exporter: InMemorySpanExporter
    ) -> None:
        """qwen-agent normally sets extra["function_id"], but not always."""
        FakeChatModel.configure([_text_turn("ok")])
        model = get_chat_model({"model": "qwen-fake", "model_type": "oi_fake"})
        model.chat(
            messages=[
                Message(role="user", content="weather?"),
                Message(role=FUNCTION, name="get_weather", content='{"temperature_c": 21}'),
            ],
            stream=False,
        )

        (span,) = _spans_by_kind(in_memory_span_exporter, "LLM")
        attributes = _attrs(span)
        roles: List[Optional[str]] = [
            attributes.get(
                f"{SpanAttributes.LLM_INPUT_MESSAGES}.{index}.{MessageAttributes.MESSAGE_ROLE}"
            )
            for index in range(4)
        ]
        assert "tool" in roles


class TestSpanHierarchy:
    def test_all_spans_share_one_trace(self, in_memory_span_exporter: InMemorySpanExporter) -> None:
        FakeChatModel.configure(
            [_tool_call_turn("get_weather", {"city": "Paris"}), _text_turn("It is 21C.")]
        )
        _run(_agent(function_list=[WeatherTool()]))

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) > 1
        assert len({span.context.trace_id for span in spans if span.context}) == 1

    def test_router_nests_member_agent_spans(
        self, in_memory_span_exporter: InMemorySpanExporter
    ) -> None:
        """Router calls `run` on the selected agent, which must nest."""
        FakeChatModel.configure(
            [_text_turn("Call: poet\nReply: on it"), _text_turn("A short poem.")]
        )
        poet = Assistant(
            llm={"model": "qwen-fake", "model_type": "oi_fake"},
            name="poet",
            description="writes poems",
        )
        router = Router(llm={"model": "qwen-fake", "model_type": "oi_fake"}, agents=[poet])
        list(router.run([{"role": "user", "content": "write a poem"}]))

        agent_spans = {span.name: span for span in _spans_by_kind(in_memory_span_exporter, "AGENT")}
        assert set(agent_spans) == {"Router.run", "poet.run"}
        router_span, poet_span = agent_spans["Router.run"], agent_spans["poet.run"]
        assert poet_span.parent is not None
        assert router_span.context is not None
        assert poet_span.parent.span_id == router_span.context.span_id
        assert len({s.context.trace_id for s in in_memory_span_exporter.get_finished_spans()}) == 1

    def test_parallel_exec_keeps_spans_in_the_trace(
        self, tracer_provider: Any, in_memory_span_exporter: InMemorySpanExporter
    ) -> None:
        """`parallel_exec` fans out over threads without copying the context.

        The instrumentor swaps in a context-preserving ThreadPoolExecutor so
        agents run this way stay in the caller's trace instead of becoming
        orphaned roots. ParallelDocQA relies on this.
        """
        from qwen_agent.utils.parallel_executor import parallel_exec

        FakeChatModel.configure([_text_turn("one"), _text_turn("two")])
        agent = _agent(name="member")

        def run_member(message: str) -> None:
            list(agent.run([{"role": "user", "content": message}]))

        tracer = tracer_provider.get_tracer(__name__)
        with tracer.start_as_current_span("outer") as outer:
            outer_span_id = outer.get_span_context().span_id
            parallel_exec(run_member, [{"message": "a"}, {"message": "b"}])

        member_spans = [
            span
            for span in _spans_by_kind(in_memory_span_exporter, "AGENT")
            if span.name == "member.run"
        ]
        assert len(member_spans) == 2
        for span in member_spans:
            assert span.parent is not None, "member agent span became an orphaned root"
            assert span.parent.span_id == outer_span_id
        assert len({s.context.trace_id for s in in_memory_span_exporter.get_finished_spans()}) == 1


class TestStreamHandling:
    def test_delta_stream_fragments_are_accumulated(
        self, in_memory_span_exporter: InMemorySpanExporter
    ) -> None:
        """With `delta_stream=True` each chunk is a fragment, not the full text.

        Deprecated upstream and never used by the agents, but a direct
        `chat(delta_stream=True)` call must still record the whole message
        rather than only the final fragment.
        """
        FakeChatModel.configure(
            [
                [
                    [Message(role=ASSISTANT, content="Hello ")],
                    [Message(role=ASSISTANT, content="world")],
                ]
            ]
        )
        model = get_chat_model({"model": "qwen-fake", "model_type": "oi_fake"})
        chunks = list(
            model.chat(
                messages=[Message(role="user", content="hi")], stream=True, delta_stream=True
            )
        )

        assert ["".join(m.content for m in chunk) for chunk in chunks] == ["Hello ", "world"]
        (span,) = _spans_by_kind(in_memory_span_exporter, "LLM")
        assert _attrs(span)[SpanAttributes.OUTPUT_VALUE] == "Hello world"

    def test_unconsumed_stream_still_records_its_span(
        self, in_memory_span_exporter: InMemorySpanExporter
    ) -> None:
        """A stream created but never iterated must not leak its span."""
        FakeChatModel.configure([_text_turn("Sunny.")])
        model = get_chat_model({"model": "qwen-fake", "model_type": "oi_fake"})
        stream = model.chat(messages=[Message(role="user", content="weather?")], stream=True)
        del stream
        gc.collect()

        assert len(_spans_by_kind(in_memory_span_exporter, "LLM")) == 1

    def test_closing_a_partly_read_stream_ends_the_span(
        self, in_memory_span_exporter: InMemorySpanExporter
    ) -> None:
        FakeChatModel.configure([_text_turn("A longer answer.")])
        model = get_chat_model({"model": "qwen-fake", "model_type": "oi_fake"})
        stream = model.chat(messages=[Message(role="user", content="hi")], stream=True)
        next(iter(stream))
        stream.close()

        (span,) = _spans_by_kind(in_memory_span_exporter, "LLM")
        assert span.end_time is not None
        assert span.status.status_code == StatusCode.OK


class TestRetrieverSpans:
    """qwen-agent's document `retrieval` tool becomes a RETRIEVER span.

    `Retrieval.call` returns one entry per source document,
    `{"url": ..., "text": [chunk, ...]}`, which `Agent._call_tool` serialises to
    JSON before the wrapper sees it. Each chunk becomes one document.
    """

    def test_retrieval_produces_a_retriever_span_with_documents(
        self, in_memory_span_exporter: InMemorySpanExporter
    ) -> None:
        FakeChatModel.configure(
            [
                _tool_call_turn("retrieval", {"query": "toaster", "files": ["/tmp/a.pdf"]}),
                _text_turn("Found it."),
            ]
        )
        _run(_agent(function_list=[FakeRetrieval()]), "what does the manual say?")

        assert not _spans_by_kind(in_memory_span_exporter, "TOOL")
        (span,) = _spans_by_kind(in_memory_span_exporter, "RETRIEVER")
        assert span.name == "retrieval.call"
        attributes = _attrs(span)
        assert attributes[SpanAttributes.TOOL_NAME] == "retrieval"
        prefix = SpanAttributes.RETRIEVAL_DOCUMENTS
        assert attributes[f"{prefix}.0.{DocumentAttributes.DOCUMENT_CONTENT}"] == "chunk one"
        assert attributes[f"{prefix}.0.{DocumentAttributes.DOCUMENT_ID}"] == "/tmp/a.pdf"
        assert json.loads(attributes[f"{prefix}.0.{DocumentAttributes.DOCUMENT_METADATA}"]) == {
            "url": "/tmp/a.pdf",
            "chunk_index": 0,
        }
        assert attributes[f"{prefix}.1.{DocumentAttributes.DOCUMENT_CONTENT}"] == "chunk two"
        assert attributes[f"{prefix}.2.{DocumentAttributes.DOCUMENT_CONTENT}"] == "other doc"
        assert attributes[f"{prefix}.2.{DocumentAttributes.DOCUMENT_ID}"] == "/tmp/b.pdf"
        # qwen-agent discards relevance scores in get_topk, so none is recorded.
        assert f"{prefix}.0.{DocumentAttributes.DOCUMENT_SCORE}" not in attributes

    def test_retriever_span_nests_under_the_agent(
        self, in_memory_span_exporter: InMemorySpanExporter
    ) -> None:
        FakeChatModel.configure(
            [_tool_call_turn("retrieval", {"query": "x", "files": []}), _text_turn("done")]
        )
        _run(_agent(function_list=[FakeRetrieval()]))

        (agent_span,) = _spans_by_kind(in_memory_span_exporter, "AGENT")
        (retriever_span,) = _spans_by_kind(in_memory_span_exporter, "RETRIEVER")
        assert retriever_span.parent is not None
        assert agent_span.context is not None
        assert retriever_span.parent.span_id == agent_span.context.span_id

    def test_other_tools_are_unaffected(
        self, in_memory_span_exporter: InMemorySpanExporter
    ) -> None:
        FakeChatModel.configure(
            [_tool_call_turn("get_weather", {"city": "Paris"}), _text_turn("21C.")]
        )
        _run(_agent(function_list=[WeatherTool()]))

        assert not _spans_by_kind(in_memory_span_exporter, "RETRIEVER")
        assert len(_spans_by_kind(in_memory_span_exporter, "TOOL")) == 1

    @pytest.mark.parametrize(
        "payload,expected",
        [
            pytest.param('[{"url": "u", "text": ["a", "b"]}]', 2, id="two-chunks"),
            pytest.param('[{"url": "u", "text": []}]', 0, id="no-chunks"),
            pytest.param("[]", 0, id="nothing-retrieved"),
            pytest.param('[{"text": ["a"]}]', 1, id="no-url"),
            pytest.param('[{"url": "u", "text": ["a", "", "b"]}]', 2, id="empty-chunk-skipped"),
        ],
    )
    def test_document_extraction_shapes(self, payload: str, expected: int) -> None:
        documents = _wrappers._retrieved_documents(payload)
        assert documents is not None
        assert len(documents) == expected

    @pytest.mark.parametrize(
        "payload",
        [
            pytest.param("not json at all", id="not-json"),
            pytest.param('{"url": "u"}', id="mapping-not-list"),
            pytest.param('[{"url": "u", "text": "a string"}]', id="text-not-a-list"),
            pytest.param('["just a string"]', id="entry-not-a-mapping"),
        ],
    )
    def test_unexpected_shapes_fall_back_to_no_documents(self, payload: str) -> None:
        """An unrecognised payload must not be forced into documents."""
        assert _wrappers._retrieved_documents(payload) is None


class TestInterruptHandling:
    """A BaseException must still end the span, without marking the call failed.

    KeyboardInterrupt and SystemExit are not errors of the model call, but the
    span still has to be ended or it is never exported.
    """

    def test_keyboard_interrupt_from_chat_still_ends_the_span(
        self, in_memory_span_exporter: InMemorySpanExporter
    ) -> None:
        model = get_chat_model({"model": "qwen-fake", "model_type": "oi_interrupt"})
        with pytest.raises(KeyboardInterrupt):
            model.chat(messages=[Message(role="user", content="hi")], stream=False)

        (span,) = _spans_by_kind(in_memory_span_exporter, "LLM")
        assert span.end_time is not None
        assert span.status.status_code is not StatusCode.ERROR

    def test_keyboard_interrupt_mid_stream_still_ends_the_span(
        self, in_memory_span_exporter: InMemorySpanExporter
    ) -> None:
        model = get_chat_model({"model": "qwen-fake", "model_type": "oi_interrupt_stream"})
        stream = model.chat(messages=[Message(role="user", content="hi")], stream=True)
        with pytest.raises(KeyboardInterrupt):
            list(stream)

        (span,) = _spans_by_kind(in_memory_span_exporter, "LLM")
        assert span.end_time is not None
        assert span.status.status_code is not StatusCode.ERROR


class TestRetrievalDetection:
    def test_a_retrieval_subclass_is_detected(self) -> None:
        assert _wrappers._is_retrieval(FakeRetrieval()) is True

    def test_an_unrelated_tool_named_retrieval_is_not_detected(self) -> None:
        """Detection is by class, not by name.

        A tool registered under the name `retrieval` that is not a `Retrieval`
        would not return the document shape a retriever span needs, so it stays
        an ordinary TOOL span.
        """
        assert _wrappers._is_retrieval(WeatherTool()) is False
        assert _wrappers._is_retrieval(None) is False


class TestStreamProtocol:
    """`chat(stream=True)` returns a real generator, so every way of driving it
    must advance the span, not just `__next__`."""

    def test_throw_ends_the_span_immediately(
        self, in_memory_span_exporter: InMemorySpanExporter
    ) -> None:
        FakeChatModel.configure([_text_turn("A longer answer.")])
        model = get_chat_model({"model": "qwen-fake", "model_type": "oi_fake"})
        stream = model.chat(messages=[Message(role="user", content="hi")], stream=True)
        next(iter(stream))

        with pytest.raises(RuntimeError):
            stream.throw(RuntimeError("caller aborts"))

        # Ended right away, not left for the garbage collector.
        (span,) = _spans_by_kind(in_memory_span_exporter, "LLM")
        assert span.end_time is not None

    def test_send_advances_the_span(self, in_memory_span_exporter: InMemorySpanExporter) -> None:
        FakeChatModel.configure([_text_turn("Sunny.")])
        model = get_chat_model({"model": "qwen-fake", "model_type": "oi_fake"})
        stream = model.chat(messages=[Message(role="user", content="hi")], stream=True)
        next(iter(stream))
        while True:
            try:
                stream.send(None)
            except StopIteration:
                break

        (span,) = _spans_by_kind(in_memory_span_exporter, "LLM")
        assert span.end_time is not None
        assert _attrs(span)[SpanAttributes.OUTPUT_VALUE] == "Sunny."

    def test_close_that_raises_still_ends_the_span(
        self, in_memory_span_exporter: InMemorySpanExporter
    ) -> None:
        """A failing generator cleanup must not leave the span unended."""

        class FailingClose:
            def __init__(self) -> None:
                self._done = False

            def __iter__(self) -> Any:
                return self

            def __next__(self) -> List[Message]:
                if self._done:
                    raise StopIteration
                self._done = True
                return [Message(role=ASSISTANT, content="x")]

            def close(self) -> None:
                raise RuntimeError("cleanup failed")

        tracer = QwenAgentInstrumentor().tracer
        assert tracer is not None
        span = tracer.start_span("manual.chat")
        stream = _wrappers._ChatStream(FailingClose(), span, lambda response: None, False)
        next(iter(stream))

        with pytest.raises(RuntimeError):
            stream.close()

        (recorded,) = in_memory_span_exporter.get_finished_spans()
        assert recorded.end_time is not None


class TestMultipleToolCalls:
    def test_two_calls_to_the_same_tool_get_distinct_ids(
        self, in_memory_span_exporter: InMemorySpanExporter
    ) -> None:
        """One model response can contain two calls to the same tool.

        `FnCallAgent._run` appends each tool result to `messages` before making
        the next call, so the "first pending call for this name" lookup resolves
        them in order.
        """
        FakeChatModel.configure(
            [
                [
                    [
                        Message(
                            role=ASSISTANT,
                            content="",
                            function_call=FunctionCall(
                                name="get_weather", arguments='{"city": "Paris"}'
                            ),
                            extra={"function_id": "call_1"},
                        ),
                        Message(
                            role=ASSISTANT,
                            content="",
                            function_call=FunctionCall(
                                name="get_weather", arguments='{"city": "Berlin"}'
                            ),
                            extra={"function_id": "call_2"},
                        ),
                    ]
                ],
                _text_turn("Both done."),
            ]
        )
        _run(_agent(function_list=[WeatherTool()]), "weather in Paris and Berlin?")

        tool_spans = _spans_by_kind(in_memory_span_exporter, "TOOL")
        assert len(tool_spans) == 2
        by_city = {
            json.loads(_attrs(span)[SpanAttributes.INPUT_VALUE])["city"]: _attrs(span)[
                ToolCallAttributes.TOOL_CALL_ID
            ]
            for span in tool_spans
        }
        assert by_city == {"Paris": "call_1", "Berlin": "call_2"}

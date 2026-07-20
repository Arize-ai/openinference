from autogen import ConversableAgent
from opentelemetry.util._importlib_metadata import entry_points

from openinference.instrumentation.ag2 import AG2Instrumentor
from openinference.instrumentation.autogen import AutogenInstrumentor


class TestInstrumentor:
    def test_entrypoint_for_opentelemetry_instrument(self) -> None:
        (instrumentor_entrypoint,) = entry_points(
            group="opentelemetry_instrumentor", name="autogen"
        )
        instrumentor = instrumentor_entrypoint.load()()
        assert isinstance(instrumentor, AutogenInstrumentor)

    def test_legacy_distribution_delegates_to_ag2_instrumentor(self) -> None:
        original = ConversableAgent.generate_reply
        instrumentor = AutogenInstrumentor()

        assert tuple(instrumentor.instrumentation_dependencies()) == ("autogen >= 0.5.0",)
        instrumentor.instrument()
        try:
            agent = ConversableAgent(
                "assistant",
                llm_config=False,
                code_execution_config=False,
                human_input_mode="NEVER",
                default_auto_reply="done",
            )
            assert agent.generate_reply(messages=[{"role": "user", "content": "hello"}]) == "done"
            assert instrumentor.is_instrumented_by_opentelemetry
            assert AG2Instrumentor().is_instrumented_by_opentelemetry
            assert ConversableAgent.generate_reply is not original
        finally:
            instrumentor.uninstrument()

        assert ConversableAgent.generate_reply is original

import json
from typing import Any, Dict, Generator, List

import pytest
import vcr  # type: ignore
from crewai import Agent, Crew, Process, Task
from crewai_tools import SerperDevTool  # type: ignore
from openinference.instrumentation import OITracer, using_attributes
from openinference.instrumentation.crewai import CrewAIInstrumentor
from openinference.semconv.trace import SpanAttributes
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

test_vcr = vcr.VCR(
    serializer="yaml",
    cassette_library_dir="tests/openinference/instrumentation/crewai/fixtures/",
    record_mode="never",
    match_on=["uri", "method"],
)


@pytest.fixture()
def in_memory_span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture()
def tracer_provider(in_memory_span_exporter: InMemorySpanExporter) -> TracerProvider:
    resource = Resource(attributes={})
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))
    return tracer_provider


@pytest.fixture()
def setup_crewai_instrumentation(
    tracer_provider: TracerProvider,
) -> Generator[None, None, None]:
    CrewAIInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    CrewAIInstrumentor().uninstrument()


# Ensure we're using the common OITracer from common opeinference-instrumentation pkg
def test_oitracer(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_crewai_instrumentation: Any,
) -> None:
    in_memory_span_exporter.clear()
    assert isinstance(CrewAIInstrumentor()._tracer, OITracer)


def test_crewai_instrumentation(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_crewai_instrumentation: Any,
) -> None:
    with test_vcr.use_cassette("crew_session.yaml", filter_headers=["authorization"]):
        import os

        os.environ["OPENAI_API_KEY"] = "fake_key"
        os.environ["SERPER_API_KEY"] = "another_fake_key"
        search_tool = SerperDevTool()
        greeter = Agent(
            role="Senior Hello Sayer",
            goal="Greet everyone you meet",
            backstory="""You work at a greeting store.
            Your expertise is greeting people
            Your parents were greeters, your grand parents were greeters.
            You were born. Nay, destined to be a greeter""",
            verbose=True,
            allow_delegation=False,
            tools=[search_tool],
        )
        aristocrat = Agent(
            role="Aristocrat",
            goal="Be greeted",
            backstory="""You were born to be treated with a greeting all the time
          You transform greetings into pleasantries that you graciously
          give to greeters.""",
            verbose=True,
            allow_delegation=True,
        )
        # Create tasks for your agents
        task1 = Task(
            description="greet like you've never greeted before",
            expected_output="A greeting in bullet points",
            agent=greeter,
        )
        task2 = Task(
            description="Using the greeting, respond with the most satisfying pleasantry",
            expected_output="a bullet pointed pleasantry",
            agent=aristocrat,
        )
        crew = Crew(
            agents=[greeter, aristocrat],
            tasks=[task1, task2],
            verbose=True,
            process=Process.sequential,
        )
        crew.kickoff()
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 6
    checked_spans = 0
    for span in spans:
        attributes = dict(span.attributes or dict())
        if span.name == "Crew.kickoff":
            checked_spans += 1
            assert attributes.get("openinference.span.kind") == "CHAIN"
            assert attributes.get("output.value")
            assert attributes.get("llm.token_count.prompt") == 5751
            assert attributes.get("llm.token_count.completion") == 1793
            assert attributes.get("llm.token_count.total") == 7544
            assert span.status.is_ok
        elif span.name == "ToolUsage._use":
            checked_spans += 1
            assert attributes.get("openinference.span.kind") == "TOOL"
            assert attributes.get("tool.name") in (
                "Search the internet",
                "Ask question to coworker",
            )
            assert span.status.is_ok
        elif span.name == "Task._execute_core":
            checked_spans += 1
            assert attributes["openinference.span.kind"] == "AGENT"
            assert attributes.get("input.value")
            assert span.status.is_ok
    assert checked_spans == 6


def test_crewai_instrumentation_context_attributes(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_crewai_instrumentation: Any,
    session_id: str,
    user_id: str,
    metadata: Dict[str, Any],
    tags: List[str],
    prompt_template: str,
    prompt_template_version: str,
    prompt_template_variables: Dict[str, Any],
) -> None:
    with using_attributes(
            session_id=session_id,
            user_id=user_id,
            metadata=metadata,
            tags=tags,
            prompt_template=prompt_template,
            prompt_template_version=prompt_template_version,
            prompt_template_variables=prompt_template_variables,
        ):
        with test_vcr.use_cassette("crew_session_context_attributes.yaml", filter_headers=["authorization"]):
            import os

            os.environ["OPENAI_API_KEY"] = "fake_key"
            os.environ["SERPER_API_KEY"] = "another_fake_key"
            search_tool = SerperDevTool()
            greeter = Agent(
                role="Senior Hello Sayer",
                goal="Greet everyone you meet",
                backstory="""You work at a greeting store.
                Your expertise is greeting people
                Your parents were greeters, your grand parents were greeters.
                You were born. Nay, destined to be a greeter""",
                verbose=True,
                allow_delegation=False,
                tools=[search_tool],
            )
            aristocrat = Agent(
                role="Aristocrat",
                goal="Be greeted",
                backstory="""You were born to be treated with a greeting all the time
              You transform greetings into pleasantries that you graciously
              give to greeters.""",
                verbose=True,
                allow_delegation=True,
            )
            # Create tasks for your agents
            task1 = Task(
                description="greet like you've never greeted before",
                expected_output="A greeting in bullet points",
                agent=greeter,
            )
            task2 = Task(
                description="Using the greeting, respond with the most satisfying pleasantry",
                expected_output="a bullet pointed pleasantry",
                agent=aristocrat,
            )
            crew = Crew(
                agents=[greeter, aristocrat],
                tasks=[task1, task2],
                verbose=True,
                process=Process.sequential,
            )
            crew.kickoff()
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) >= 1
    span = spans[0]
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))
    _check_context_attributes(
        attributes,
        session_id,
        user_id,
        metadata,
        tags,
        prompt_template,
        prompt_template_version,
        prompt_template_variables,
    )


def _check_context_attributes(
    attributes: Dict[str, Any],
    session_id: str,
    user_id: str,
    metadata: Dict[str, Any],
    tags: List[str],
    prompt_template: str,
    prompt_template_version: str,
    prompt_template_variables: Dict[str, Any],
) -> None:
    assert attributes.pop(SpanAttributes.SESSION_ID, None) == session_id
    assert attributes.pop(SpanAttributes.USER_ID, None) == user_id
    attr_metadata = attributes.pop(SpanAttributes.METADATA, None)
    assert attr_metadata is not None
    assert isinstance(attr_metadata, str)  # must be json string
    metadata_dict = json.loads(attr_metadata)
    assert metadata_dict == metadata
    attr_tags = attributes.pop(SpanAttributes.TAG_TAGS, None)
    assert attr_tags is not None
    assert len(attr_tags) == len(tags)
    assert list(attr_tags) == tags
    assert attributes.pop(SpanAttributes.LLM_PROMPT_TEMPLATE, None) == prompt_template
    assert (
        attributes.pop(SpanAttributes.LLM_PROMPT_TEMPLATE_VERSION, None) == prompt_template_version
    )
    assert attributes.pop(SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES, None) == json.dumps(
        prompt_template_variables
    )


@pytest.fixture()
def session_id() -> str:
    return "my-test-session-id"


@pytest.fixture()
def user_id() -> str:
    return "my-test-user-id"


@pytest.fixture()
def metadata() -> Dict[str, Any]:
    return {
        "test-int": 1,
        "test-str": "string",
        "test-list": [1, 2, 3],
        "test-dict": {
            "key-1": "val-1",
            "key-2": "val-2",
        },
    }


@pytest.fixture()
def tags() -> List[str]:
    return ["tag-1", "tag-2"]


@pytest.fixture
def prompt_template() -> str:
    return (
        "This is a test prompt template with int {var_int}, "
        "string {var_string}, and list {var_list}"
    )


@pytest.fixture
def prompt_template_version() -> str:
    return "v1.0"


@pytest.fixture
def prompt_template_variables() -> Dict[str, Any]:
    return {
        "var_int": 1,
        "var_str": "2",
        "var_list": [1, 2, 3],
    }

import os
import tempfile
import zipfile
from io import BytesIO
from typing import Any, Dict, List, Optional, Set, Tuple, cast

import requests
from llama_index.core import SQLDatabase
from llama_index.core.agent import (
    AgentRunner,
    QueryPipelineAgentWorker,
    ReActChatFormatter,
    ReActOutputParser,
)
from llama_index.core.agent.react.types import ObservationReasoningStep, ResponseReasoningStep
from llama_index.core.base.agent.types import Task
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.base.query_pipeline.query import QueryComponent
from llama_index.core.chat_engine.types import AgentChatResponse
from llama_index.core.indices.struct_store import NLSQLTableQueryEngine
from llama_index.core.query_pipeline import (
    AgentFnComponent,
    AgentInputComponent,
    CustomAgentComponent,
    QueryPipeline,
    ToolRunnerComponent,
)
from llama_index.core.tools import BaseTool, QueryEngineTool
from llama_index.llms.openai import OpenAI
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from sqlalchemy import create_engine

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)

temp_dir = tempfile.mkdtemp()
url = "https://www.sqlitetutorial.net/wp-content/uploads/2018/03/chinook.zip"
with zipfile.ZipFile(BytesIO(requests.get(url).content), "r") as f:
    f.extractall(temp_dir)
engine = create_engine(f"sqlite:///{os.path.join(temp_dir, 'chinook.db')}")
sql_database = SQLDatabase(engine)
sql_query_engine = NLSQLTableQueryEngine(
    sql_database=sql_database,
    tables=["albums", "tracks", "artists"],
    verbose=True,
)
sql_tool = QueryEngineTool.from_defaults(
    query_engine=sql_query_engine,
    name="sql_tool",
    description=("Useful for translating a natural language query into a SQL query"),
)


def agent_input_fn(task: Task, state: Dict[str, Any]) -> Dict[str, Any]:
    """Agent input function.

    Returns:
        A Dictionary of output keys and values. If you are specifying
        src_key when defining links between this component and other
        components, make sure the src_key matches the specified output_key.

    """
    # initialize current_reasoning
    if "current_reasoning" not in state:
        state["current_reasoning"] = []
    reasoning_step = ObservationReasoningStep(observation=task.input)
    state["current_reasoning"].append(reasoning_step)
    return {"input": task.input}


agent_input_component = AgentInputComponent(fn=agent_input_fn)


def react_prompt_fn(
    task: Task, state: Dict[str, Any], input: str, tools: List[BaseTool]
) -> List[ChatMessage]:
    # Add input to reasoning
    chat_formatter = ReActChatFormatter()
    return chat_formatter.format(
        tools,
        chat_history=task.memory.get() + state["memory"].get_all(),
        current_reasoning=state["current_reasoning"],
    )


react_prompt_component = AgentFnComponent(fn=react_prompt_fn, partial_dict={"tools": [sql_tool]})


def finalize_fn(
    task: Task,
    state: Dict[str, Any],
    reasoning_step: Any,
    is_done: bool = False,
    tool_output: Optional[Any] = None,
) -> Tuple[AgentChatResponse, bool]:
    """Finalize function.

    Here we take the latest reasoning step, and a tool output (if provided),
    and return the agent output (and decide if agent is done).

    This function returns an `AgentChatResponse` and `is_done` tuple. and
    is the last component of the query pipeline. This is the expected
    return type for any query pipeline passed to `QueryPipelineAgentWorker`.

    """
    current_reasoning = state["current_reasoning"]
    current_reasoning.append(reasoning_step)
    # if tool_output is not None, add to current reasoning
    if tool_output is not None:
        observation_step = ObservationReasoningStep(observation=str(tool_output))
        current_reasoning.append(observation_step)
    if isinstance(current_reasoning[-1], ResponseReasoningStep):
        response_step = cast(ResponseReasoningStep, current_reasoning[-1])
        response_str = response_step.response
    else:
        response_str = current_reasoning[-1].get_content()

    # if is_done, add to memory
    # NOTE: memory is a reserved keyword in `state`, but you can add your own too
    if is_done:
        state["memory"].put(ChatMessage(content=task.input, role=MessageRole.USER))
        state["memory"].put(ChatMessage(content=response_str, role=MessageRole.ASSISTANT))

    return AgentChatResponse(response=response_str), is_done


class OutputAgentComponent(CustomAgentComponent):
    """Output agent component."""

    tool_runner_component: ToolRunnerComponent
    output_parser: ReActOutputParser

    def __init__(self, tools, **kwargs):
        tool_runner_component = ToolRunnerComponent(tools)
        super().__init__(
            tool_runner_component=tool_runner_component, output_parser=ReActOutputParser(), **kwargs
        )

    def _run_component(self, **kwargs: Any) -> Any:
        """Run component."""
        chat_response = kwargs["chat_response"]
        task = kwargs["task"]
        state = kwargs["state"]
        reasoning_step = self.output_parser.parse(chat_response.message.content)
        if reasoning_step.is_done:
            return {"output": finalize_fn(task, state, reasoning_step, is_done=True)}
        else:
            tool_output = self.tool_runner_component.run_component(
                tool_name=reasoning_step.action,
                tool_input=reasoning_step.action_input,
            )
            return {
                "output": finalize_fn(
                    task,
                    state,
                    reasoning_step,
                    is_done=False,
                    tool_output=tool_output,
                )
            }

    @property
    def _input_keys(self) -> Set[str]:
        return {"chat_response"}

    @property
    def _optional_input_keys(self) -> Set[str]:
        return {"is_done", "tool_output"}

    @property
    def _output_keys(self) -> Set[str]:
        return {"output"}

    @property
    def sub_query_components(self) -> List[QueryComponent]:
        return [self.tool_runner_component]


react_output_component = OutputAgentComponent([sql_tool])

qp = QueryPipeline(
    modules={
        "agent_input": agent_input_component,
        "react_prompt": react_prompt_component,
        "llm": OpenAI(model="gpt-4o"),
        "react_output": react_output_component,
    },
    verbose=True,
)
qp.add_chain(["agent_input", "react_prompt", "llm", "react_output"])

agent_worker = QueryPipelineAgentWorker(qp)
agent = AgentRunner(agent_worker)

if __name__ == "__main__":
    response = agent.chat("What was the year that The Notorious B.I.G was signed to Bad Boy?")
    print(str(response))

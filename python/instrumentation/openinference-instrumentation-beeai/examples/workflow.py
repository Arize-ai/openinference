import asyncio
import sys
import traceback
from typing import Literal, TypeAlias

from beeai_framework.errors import FrameworkError
from beeai_framework.workflows import Workflow, WorkflowReservedStepName
from pydantic import BaseModel

from examples.setup import setup_observability

setup_observability()

WorkflowStep: TypeAlias = Literal["pre_process", "add_loop", "post_process"]


async def main() -> None:
    # State
    class State(BaseModel):
        x: int
        y: int
        abs_repetitions: int | None = None
        result: int | None = None

    def pre_process(state: State) -> WorkflowStep:
        print("pre_process")
        state.abs_repetitions = abs(state.y)
        return "add_loop"

    def add_loop(state: State) -> WorkflowStep | WorkflowReservedStepName:
        if state.abs_repetitions and state.abs_repetitions > 0:
            result = (state.result if state.result is not None else 0) + state.x
            abs_repetitions = (
                state.abs_repetitions if state.abs_repetitions is not None else 0
            ) - 1
            print(f"add_loop: intermediate result {result}")
            state.abs_repetitions = abs_repetitions
            state.result = result
            return Workflow.SELF
        else:
            return "post_process"

    def post_process(state: State) -> WorkflowReservedStepName:
        print("post_process")
        if state.y < 0:
            result = -(state.result if state.result is not None else 0)
            state.result = result
        return Workflow.END

    multiplication_workflow = Workflow[State, WorkflowStep](
        name="MultiplicationWorkflow", schema=State
    )
    multiplication_workflow.add_step("pre_process", pre_process)
    multiplication_workflow.add_step("add_loop", add_loop)
    multiplication_workflow.add_step("post_process", post_process)

    response = await multiplication_workflow.run(State(x=8, y=5))
    print(f"result: {response.state.result}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

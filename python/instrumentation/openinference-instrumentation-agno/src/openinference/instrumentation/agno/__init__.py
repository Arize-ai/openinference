from typing import Any, Callable, Collection, List, Optional, Type

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore
from wrapt import wrap_function_wrapper

from openinference.instrumentation import (
    OITracer,
    TraceConfig,
)
from openinference.instrumentation.agno._model_wrapper import (
    _ModelWrapper,
)
from openinference.instrumentation.agno._runs_wrapper import _RunWrapper
from openinference.instrumentation.agno._tools_wrapper import _FunctionCallWrapper
from openinference.instrumentation.agno._workflow_wrapper import (
    _ParallelWrapper,
    _StepWrapper,
    _WorkflowWrapper,
)
from openinference.instrumentation.agno.version import __version__

_instruments = ("agno >= 1.5.2",)


# Find all model classes in agno.models that inherit from BaseModel
def find_model_subclasses() -> List[Type[Any]]:
    """Find all model classes in agno.models that inherit from BaseModel."""
    # Import necessary modules
    import importlib
    import inspect
    import pkgutil

    from agno.models.base import Model

    model_subclasses = set()

    # Import the agno.models package
    try:
        import agno.models as models_package

        # Walk through all modules in the package
        for _, module_name, _ in pkgutil.walk_packages(
            models_package.__path__, models_package.__name__ + "."
        ):
            try:
                # Import the module
                module = importlib.import_module(module_name)

                # Find all classes in the module that inherit from Model
                for _, obj in inspect.getmembers(module):
                    if inspect.isclass(obj) and issubclass(obj, Model) and obj is not Model:
                        model_subclasses.add(obj)
            except (ImportError, AttributeError):
                # Skip modules that can't be imported
                continue
    except ImportError:
        # If agno.models can't be imported, return empty list
        pass

    # Sort the model subclasses by their method resolution order (MRO) length.
    # This ensures that subclasses are listed before their superclasses and that the
    # most specific classes come first. If a base class's method was wrapped before
    # the method of a subclass that inherits it, the resolution for the subclass would
    # find the already-wrapped base method (see wrap_function_wrapper execution flow).
    # This would result in the wrapper being applied a second time, leading to incorrect
    # behavior such as duplicated spans or metrics. By sorting from the most specific
    # class to the most general, we ensure that any method is wrapped only once,
    # starting at the level of the most-derived class in the hierarchy.
    sorted_models = sorted(list(model_subclasses), key=lambda cls: len(cls.__mro__), reverse=True)
    return sorted_models


class AgnoInstrumentor(BaseInstrumentor):  # type: ignore
    __slots__ = (
        "_original_run_method",
        "_original_run_stream_method",
        "_original_arun_method",
        "_original_arun_stream_method",
        "_original_team_run_method",
        "_original_team_run_stream_method",
        "_original_team_arun_method",
        "_original_team_arun_stream_method",
        "_original_function_execute_method",
        "_original_function_aexecute_method",
        "_original_model_call_methods",
        "_original_workflow_methods",
        "_original_step_methods",
        "_original_parallel_methods",
        "_tracer",
    )

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        from agno.agent import Agent
        from agno.team import Team
        from agno.tools.function import FunctionCall

        if not (tracer_provider := kwargs.get("tracer_provider")):
            tracer_provider = trace_api.get_tracer_provider()
        if not (config := kwargs.get("config")):
            config = TraceConfig()
        else:
            assert isinstance(config, TraceConfig)
        self._tracer = OITracer(
            trace_api.get_tracer(__name__, __version__, tracer_provider),
            config=config,
        )

        run_wrapper = _RunWrapper(tracer=self._tracer)
        self._original_run_method = getattr(Agent, "_run", None)
        wrap_function_wrapper(
            module=Agent,
            name="_run",
            wrapper=run_wrapper.run,
        )
        self._original_run_stream_method = getattr(Agent, "_run_stream", None)
        wrap_function_wrapper(
            module=Agent,
            name="_run_stream",
            wrapper=run_wrapper.run_stream,
        )
        self._original_arun_method = getattr(Agent, "_arun", None)
        wrap_function_wrapper(
            module=Agent,
            name="_arun",
            wrapper=run_wrapper.arun,
        )
        self._original_arun_stream_method = getattr(Agent, "_arun_stream", None)
        wrap_function_wrapper(
            module=Agent,
            name="_arun_stream",
            wrapper=run_wrapper.arun_stream,
        )

        # Register wrapper for team
        self._original_team_run_method = getattr(Team, "_run", None)
        wrap_function_wrapper(
            module=Team,
            name="_run",
            wrapper=run_wrapper.run,
        )
        self._original_team_run_stream_method = getattr(Team, "_run_stream", None)
        wrap_function_wrapper(
            module=Team,
            name="_run_stream",
            wrapper=run_wrapper.run_stream,
        )
        self._original_team_arun_method = getattr(Team, "_arun", None)
        wrap_function_wrapper(
            module=Team,
            name="_arun",
            wrapper=run_wrapper.arun,
        )
        self._original_team_arun_stream_method = getattr(Team, "_arun_stream", None)
        wrap_function_wrapper(
            module=Team,
            name="_arun_stream",
            wrapper=run_wrapper.arun_stream,
        )

        self._original_model_call_methods: Optional[dict[type, dict[str, Callable[..., Any]]]] = {}

        # Get all model subclasses
        agno_model_subclasses = find_model_subclasses()
        # Instrument all model subclasses
        for model_subclass in agno_model_subclasses:
            model_wrapper = _ModelWrapper(tracer=self._tracer)
            self._original_model_call_methods[model_subclass] = {
                "invoke": model_subclass.invoke,
                "ainvoke": model_subclass.ainvoke,
                "invoke_stream": model_subclass.invoke_stream,
                "ainvoke_stream": model_subclass.ainvoke_stream,
            }

            # Only wrap if the class has a invoke method
            for method_name, method in self._original_model_call_methods[model_subclass].items():
                if method is not None:
                    if method_name == "invoke":
                        wrap_function_wrapper(
                            module=model_subclass,
                            name=method_name,
                            wrapper=model_wrapper.run,
                        )
                    elif method_name == "invoke_stream":
                        wrap_function_wrapper(
                            module=model_subclass,
                            name=method_name,
                            wrapper=model_wrapper.run_stream,
                        )
                    elif method_name == "ainvoke":
                        wrap_function_wrapper(
                            module=model_subclass,
                            name=method_name,
                            wrapper=model_wrapper.arun,
                        )
                    elif method_name == "ainvoke_stream":
                        wrap_function_wrapper(
                            module=model_subclass,
                            name=method_name,
                            wrapper=model_wrapper.arun_stream,
                        )

        function_call_wrapper = _FunctionCallWrapper(tracer=self._tracer)
        self._original_function_execute_method = getattr(FunctionCall, "execute", None)
        wrap_function_wrapper(
            module=FunctionCall,
            name="execute",
            wrapper=function_call_wrapper.run,
        )
        self._original_function_aexecute_method = getattr(FunctionCall, "aexecute", None)
        wrap_function_wrapper(
            module=FunctionCall,
            name="aexecute",
            wrapper=function_call_wrapper.arun,
        )

        # Instrument Workflow and Step
        try:
            from agno.workflow.step import Step
            from agno.workflow.workflow import Workflow

            workflow_wrapper = _WorkflowWrapper(tracer=self._tracer)
            step_wrapper = _StepWrapper(tracer=self._tracer)

            # Store original methods
            self._original_workflow_methods = {}
            self._original_step_methods = {}

            # Wrap Workflow.run (sync) - wraps both streaming and non-streaming
            if hasattr(Workflow, "run") and callable(getattr(Workflow, "run", None)):
                self._original_workflow_methods["run"] = Workflow.run
                wrap_function_wrapper(
                    module=Workflow,
                    name="run",
                    wrapper=workflow_wrapper.run,
                )

            # Wrap Workflow.arun (async) - wraps both streaming and non-streaming
            if hasattr(Workflow, "arun") and callable(getattr(Workflow, "arun", None)):
                self._original_workflow_methods["arun"] = Workflow.arun  # type: ignore[assignment]
                wrap_function_wrapper(
                    module=Workflow,
                    name="arun",
                    wrapper=workflow_wrapper.arun,
                )

            # Wrap Step.execute (sync)
            if hasattr(Step, "execute") and callable(getattr(Step, "execute", None)):
                self._original_step_methods["execute"] = Step.execute
                wrap_function_wrapper(
                    module=Step,
                    name="execute",
                    wrapper=step_wrapper.run,
                )

            # Wrap Step.execute_stream (sync streaming)
            if hasattr(Step, "execute_stream") and callable(getattr(Step, "execute_stream", None)):
                self._original_step_methods["execute_stream"] = Step.execute_stream  # type: ignore[assignment]
                wrap_function_wrapper(
                    module=Step,
                    name="execute_stream",
                    wrapper=step_wrapper.run,
                )

            # Wrap Step.aexecute (async)
            if hasattr(Step, "aexecute") and callable(getattr(Step, "aexecute", None)):
                self._original_step_methods["aexecute"] = Step.aexecute  # type: ignore[assignment]
                wrap_function_wrapper(
                    module=Step,
                    name="aexecute",
                    wrapper=step_wrapper.arun,
                )

            # Wrap Step.aexecute_stream (async streaming)
            if hasattr(Step, "aexecute_stream") and callable(
                getattr(Step, "aexecute_stream", None)
            ):
                self._original_step_methods["aexecute_stream"] = Step.aexecute_stream  # type: ignore[assignment]
                wrap_function_wrapper(
                    module=Step,
                    name="aexecute_stream",
                    wrapper=step_wrapper.arun,
                )

            # Instrument Parallel for context propagation to worker threads
            try:
                from agno.workflow.parallel import Parallel

                parallel_wrapper = _ParallelWrapper(tracer=self._tracer)
                self._original_parallel_methods = {}

                # Wrap Parallel.execute (sync non-streaming)
                if hasattr(Parallel, "execute") and callable(getattr(Parallel, "execute", None)):
                    self._original_parallel_methods["execute"] = Parallel.execute
                    wrap_function_wrapper(
                        module=Parallel,
                        name="execute",
                        wrapper=parallel_wrapper.execute,
                    )

                # Wrap Parallel.execute_stream (sync streaming)
                if hasattr(Parallel, "execute_stream") and callable(
                    getattr(Parallel, "execute_stream", None)
                ):
                    self._original_parallel_methods["execute_stream"] = Parallel.execute_stream  # type: ignore[assignment]
                    wrap_function_wrapper(
                        module=Parallel,
                        name="execute_stream",
                        wrapper=parallel_wrapper.execute,
                    )

                # Wrap Parallel.aexecute (async non-streaming)
                if hasattr(Parallel, "aexecute") and callable(getattr(Parallel, "aexecute", None)):
                    self._original_parallel_methods["aexecute"] = Parallel.aexecute  # type: ignore[assignment]
                    wrap_function_wrapper(
                        module=Parallel,
                        name="aexecute",
                        wrapper=parallel_wrapper.aexecute,
                    )

                # Wrap Parallel.aexecute_stream (async streaming)
                if hasattr(Parallel, "aexecute_stream") and callable(
                    getattr(Parallel, "aexecute_stream", None)
                ):
                    self._original_parallel_methods["aexecute_stream"] = Parallel.aexecute_stream  # type: ignore[assignment]
                    wrap_function_wrapper(
                        module=Parallel,
                        name="aexecute_stream",
                        wrapper=parallel_wrapper.aexecute,
                    )

            except (ImportError, AttributeError):
                # Parallel not available in this version of agno
                self._original_parallel_methods = None  # type: ignore[assignment]

        except (ImportError, AttributeError):
            # Workflow/Step not available in this version of agno
            self._original_workflow_methods = None  # type: ignore[assignment]
            self._original_step_methods = None  # type: ignore[assignment]
            self._original_parallel_methods = None  # type: ignore[assignment]

    def _uninstrument(self, **kwargs: Any) -> None:
        from agno.agent import Agent
        from agno.team import Team
        from agno.tools.function import FunctionCall

        if self._original_run_method is not None:
            Agent._run = self._original_run_method  # type: ignore[method-assign]
            self._original_run_method = None
        if self._original_run_stream_method is not None:
            Agent._run_stream = self._original_run_stream_method  # type: ignore[method-assign]
            self._original_run_stream_method = None

        if self._original_arun_method is not None:
            Agent._arun = self._original_arun_method  # type: ignore[method-assign]
            self._original_arun_method = None
        if self._original_arun_stream_method is not None:
            Agent._arun_stream = self._original_arun_stream_method  # type: ignore[method-assign]
            self._original_arun_stream_method = None

        if self._original_team_run_method is not None:
            Team._run = self._original_team_run_method  # type: ignore[method-assign]
            self._original_team_run_method = None
        if self._original_team_run_stream_method is not None:
            Team._run_stream = self._original_team_run_stream_method  # type: ignore[method-assign]
            self._original_team_run_stream_method = None

        if self._original_team_arun_method is not None:
            Team._arun = self._original_team_arun_method  # type: ignore[method-assign]
            self._original_team_arun_method = None
        if self._original_team_arun_stream_method is not None:
            Team._arun_stream = self._original_team_arun_stream_method  # type: ignore[method-assign]
            self._original_team_arun_stream_method = None

        if self._original_model_call_methods is not None:
            for (
                model_subclass,
                original_model_call_methods,
            ) in self._original_model_call_methods.items():
                for method_name, method in original_model_call_methods.items():
                    setattr(model_subclass, method_name, method)
            self._original_model_call_methods = None

        if self._original_function_execute_method is not None:
            FunctionCall.execute = self._original_function_execute_method  # type: ignore[method-assign]
            self._original_function_execute_method = None

        if self._original_function_aexecute_method is not None:
            FunctionCall.aexecute = self._original_function_aexecute_method  # type: ignore[method-assign]
            self._original_function_aexecute_method = None

        # Uninstrument Workflow and Step
        if self._original_workflow_methods is not None:
            try:
                from agno.workflow.workflow import Workflow

                for method_name, original in self._original_workflow_methods.items():
                    if original is not None:
                        setattr(Workflow, method_name, original)
            except ImportError:
                pass
            self._original_workflow_methods = None  # type: ignore[assignment]

        if self._original_step_methods is not None:
            try:
                from agno.workflow.step import Step

                for method_name, original in self._original_step_methods.items():  # type: ignore[assignment]
                    if original is not None:
                        setattr(Step, method_name, original)
            except ImportError:
                pass
            self._original_step_methods = None  # type: ignore[assignment]

        # Uninstrument Parallel
        if self._original_parallel_methods is not None:
            try:
                from agno.workflow.parallel import Parallel

                for method_name, original in self._original_parallel_methods.items():  # type: ignore[assignment]
                    if original is not None:
                        setattr(Parallel, method_name, original)
            except ImportError:
                pass
            self._original_parallel_methods = None  # type: ignore[assignment]

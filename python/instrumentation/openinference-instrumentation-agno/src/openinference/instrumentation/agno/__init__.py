from typing import Any, Callable, Collection, List, Optional, Type

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore
from wrapt import wrap_function_wrapper

from openinference.instrumentation import (
    OITracer,
    TraceConfig,
)
from openinference.instrumentation.agno._wrappers import (
    _FunctionCallWrapper,
    _ModelWrapper,
    _RunWrapper,
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

    return list(model_subclasses)


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

    def _uninstrument(self, **kwargs: Any) -> None:
        from agno.agent import Agent
        from agno.team import Team
        from agno.tools.function import FunctionCall

        if self._original_run_method is not None:
            Agent.run = self._original_run_method  # type: ignore[method-assign]
            self._original_run_method = None

        if self._original_arun_method is not None:
            Agent.arun = self._original_arun_method  # type: ignore[method-assign]
            self._original_arun_method = None

        if self._original_team_run_method is not None:
            Team.run = self._original_team_run_method  # type: ignore[method-assign]
            self._original_team_run_method = None

        if self._original_team_arun_method is not None:
            Team.arun = self._original_team_arun_method  # type: ignore[method-assign]
            self._original_team_arun_method = None

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

"""Input and output for the span that represents a whole agent run.

The SDK's tracing callbacks only ever see ``SpanData`` objects, and the three that
describe an agent operation -- ``AgentSpanData``, ``TaskSpanData`` and ``TurnSpanData`` --
carry no input or output fields at all. A ``TracingProcessor`` therefore cannot observe
the boundary values of the run it is tracing, and deriving them from child LLM spans
would be a guess rather than an observation: input guardrails run concurrently with the
model turn, so "the most recent response wins" is a race, and a run whose agent declares
an ``output_type`` finishes with a structured object that no raw LLM text ever equals.

Both values are available from public API, just at two different moments:

* the **input** is ``AgentRunner.run``'s own argument, known before the trace exists
* the **output** is handed to ``RunHooks.on_agent_end`` while the spans are still open

So this module wraps the runner to seed the input and to compose the caller's hooks with
one that records the output, and the processor reads both off a holder kept in a
``ContextVar``.

The holder is mutable and shared by reference on purpose. ``on_agent_end`` is awaited
inside an ``asyncio.gather``, which copies the current context, so a value *set* on a
``ContextVar`` inside the hook would not propagate back out to the run loop -- but a
mutation of an object the enclosing context already points at is visible to everyone.
That same property makes nesting correct for free: an agent exposed as a tool, or a
guardrail that runs an agent of its own, gets its own holder in its own context and so
cannot overwrite the enclosing run's output.

Only the trace root records these values. Agent, task and turn spans are left alone: the
SDK gives no way to tell which of several same-named agent spans a value belongs to, and
guessing is the thing this module exists to avoid.
"""

from __future__ import annotations

import importlib
import inspect
import logging
from contextvars import ContextVar
from typing import Any, Callable, Mapping, Optional

from opentelemetry import context as context_api
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.util.types import AttributeValue

from openinference.instrumentation import get_input_attributes, get_output_attributes

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Distinguishes "never recorded" from a legitimately empty value such as "".
_UNSET: Any = object()

# The entry points to wrap. ``Runner.run`` and friends are thin classmethod facades that
# delegate to the module-global ``DEFAULT_AGENT_RUNNER``, so patching the runner covers
# both those callers and anyone holding a runner directly, and ``hooks`` always arrives
# as a keyword at this boundary.
RUNNER_METHODS: tuple[str, ...] = ("run", "run_sync", "run_streamed")


class RunIO:
    """The boundary values of one agent run, filled in as they become available."""

    __slots__ = ("input", "output")

    def __init__(self, input: Any = _UNSET, output: Any = _UNSET) -> None:
        self.input = input
        self.output = output

    @property
    def has_input(self) -> bool:
        return self.input is not _UNSET

    @property
    def has_output(self) -> bool:
        return self.output is not _UNSET


_run_io: ContextVar[Optional[RunIO]] = ContextVar(
    "openinference_openai_agents_run_io", default=None
)


def current_run_io() -> Optional[RunIO]:
    """The holder for the run in progress, if this code is running inside one."""
    return _run_io.get()


def input_attributes(value: Any) -> Optional[Mapping[str, AttributeValue]]:
    """A run's input as span attributes, or ``None`` when there is nothing to record.

    A resumed run is handed a ``RunState`` in place of the user's input. That is the
    SDK's own bookkeeping rather than a boundary value, so it is deliberately skipped:
    anything other than a string or a list of input items is not this run's input.
    Serializing whatever passes that gate is left to the shared helper.
    """
    if not isinstance(value, (str, list, tuple)):
        return None
    return get_input_attributes(value)


def output_attributes(value: Any) -> Optional[Mapping[str, AttributeValue]]:
    """A run's final output as span attributes, or ``None`` when there is nothing to record.

    Unlike the input there is no gate on the type: an agent's final output is whatever it
    declared, and the shared helper already handles the shapes that arrive here --
    notably the Pydantic model produced by an ``output_type``.
    """
    if value is None:
        return None
    return get_output_attributes(value)


def _output_from_call(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> Any:
    """``on_agent_end``'s output argument: third positional in every SDK version so far.

    Only its position is relied on, never its type, which is what makes one wrapper work
    across versions -- the first argument changed from ``RunContextWrapper`` to
    ``AgentHookContext`` and is never read here.
    """
    if "output" in kwargs:
        return kwargs["output"]
    if len(args) >= 3:
        return args[2]
    return _UNSET


def _input_from_call(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> Any:
    """The runner's ``input`` argument: second positional after the bound instance."""
    if "input" in kwargs:
        return kwargs["input"]
    if len(args) >= 2:
        return args[1]
    return _UNSET


def _record_output(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> None:
    if (holder := _run_io.get()) is None:
        return
    if (output := _output_from_call(args, kwargs)) is _UNSET:
        return
    holder.output = output


def _delegating_method(name: str) -> Callable[..., Any]:
    async def method(self: Any, *args: Any, **kwargs: Any) -> Any:
        return await getattr(self._inner, name)(*args, **kwargs)

    method.__name__ = name
    return method


def _recording_method(name: str) -> Callable[..., Any]:
    async def method(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            _record_output(args, kwargs)
        except Exception:
            # A hook that raises would fail the user's run, and the only thing at stake
            # here is one attribute.
            logger.debug("could not record run output", exc_info=True)
        return await getattr(self._inner, name)(*args, **kwargs)

    method.__name__ = name
    return method


_hooks_class: Optional[type] = None


def run_hooks_class() -> Optional[type]:
    """A ``RunHooks`` subclass that records the run's output and delegates everything.

    Built by enumerating the base class rather than by naming its methods, because the
    SDK has grown hooks over time -- 0.2.x has five, 0.18 has seven -- and a caller's
    callbacks must never be swallowed on a version this was not written against.
    """
    global _hooks_class
    if _hooks_class is not None:
        return _hooks_class
    if (base := _run_hooks_base()) is None:
        return None

    def __init__(self: Any, inner: Any = None) -> None:
        self._inner = inner if inner is not None else base()

    namespace: dict[str, Any] = {
        "__init__": __init__,
        "__doc__": "Records the run's output, then delegates to the caller's hooks.",
    }
    for name, _ in inspect.getmembers(base, inspect.iscoroutinefunction):
        if not name.startswith("on_"):
            continue
        namespace[name] = (
            _recording_method(name) if name == "on_agent_end" else _delegating_method(name)
        )
    if "on_agent_end" not in namespace:
        logger.debug("RunHooks has no on_agent_end -- run output will not be recorded")
        return None
    _hooks_class = type("_OpenInferenceRunHooks", (base,), namespace)
    return _hooks_class


def _run_hooks_base() -> Optional[type]:
    try:
        from agents.lifecycle import RunHooksBase

        return RunHooksBase
    except Exception:
        logger.debug("agents.lifecycle.RunHooksBase not importable", exc_info=True)
        return None


def _wrappable_hooks(hooks: Any) -> bool:
    """Whether the caller's ``hooks`` argument can be composed with ours.

    Anything else is passed through untouched. The SDK validates this argument and
    raises on a bad one; wrapping it would hide that error behind an instance that does
    pass validation and then fails mid-run.
    """
    if hooks is None:
        return True
    base = _run_hooks_base()
    return base is not None and isinstance(hooks, base)


def _prepare(
    args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> tuple[Optional[RunIO], Mapping[str, Any]]:
    """A holder seeded with the run's input, and the kwargs with our hooks composed in."""
    try:
        holder = RunIO(input=_input_from_call(args, kwargs))
        if (hooks_class := run_hooks_class()) is not None and _wrappable_hooks(kwargs.get("hooks")):
            return holder, {**kwargs, "hooks": hooks_class(kwargs.get("hooks"))}
        # Input is still worth recording even when the output half cannot be set up.
        return holder, kwargs
    except Exception:
        logger.debug("could not set up run I/O capture", exc_info=True)
        return None, kwargs


def make_run_wrapper() -> Callable[..., Any]:
    """Wrapper for the awaitable runner entry point."""

    async def wrapper(
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        # Suppressed means inert: the caller's own ``hooks`` argument must reach the SDK
        # untouched, not wrapped in ours.
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return await wrapped(*args, **kwargs)
        holder, kwargs = _prepare(args, kwargs)
        if holder is None:
            return await wrapped(*args, **kwargs)
        token = _run_io.set(holder)
        try:
            return await wrapped(*args, **kwargs)
        finally:
            _run_io.reset(token)

    return wrapper


def make_sync_run_wrapper() -> Callable[..., Any]:
    """Wrapper for the runner entry points that are ordinary methods.

    ``run_streamed`` returns before the run has finished: it starts a task, and that task
    copies the context as it is created, so resetting the ``ContextVar`` here leaves the
    holder the background run is already pointing at untouched. ``run_sync`` drives the
    loop to completion first, and the context it copies for the coroutine likewise points
    at the same holder.
    """

    def wrapper(
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)
        holder, kwargs = _prepare(args, kwargs)
        if holder is None:
            return wrapped(*args, **kwargs)
        token = _run_io.set(holder)
        try:
            return wrapped(*args, **kwargs)
        finally:
            _run_io.reset(token)

    return wrapper


def find_agent_runner_bindings() -> list[tuple[Any, str]]:
    """The runner entry points to patch, as ``(owner, attribute)``.

    Unlike the function-tool execution step, these are methods on a class, so every
    module that re-exports the class shares the one object and patching it once is
    enough. ``Runner``'s classmethods are deliberately left alone: they look the runner
    up at call time, so they pick the patch up for free, and patching both would nest
    one capture inside the other for no gain.
    """
    try:
        run_module = importlib.import_module("agents.run")
    except Exception:
        logger.debug("agents.run not importable", exc_info=True)
        return []
    if (runner := getattr(run_module, "AgentRunner", None)) is None:
        logger.debug("agents.run.AgentRunner not present")
        return []
    return [(runner, name) for name in RUNNER_METHODS if name in runner.__dict__]

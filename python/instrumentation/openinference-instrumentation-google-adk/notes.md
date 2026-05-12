# Notes — Google ADK Instrumentor

Reference notes covering the integration surface between this instrumentor and `google-adk`: the span families ADK emits, how each is hooked, why the abstractions exist in the shape they do, and where coverage has known gaps.

Scope: ADK ≥ 1.2.1 (the package's minimum). Differences between ADK versions are called out where they shape the architecture. Permalinks pin to the [`v1.32.0` tag](https://github.com/google/adk-python/tree/v1.32.0) of `google/adk-python` (commit [`5e49cfa6`](https://github.com/google/adk-python/commit/5e49cfa6567a09e06409b0f380434f12f85a17c9)); in-repo references use relative paths so they render in IDEs and on GitHub.

---

## ADK telemetry surface and the consolidation in 1.32

### The pre-1.32 model: one tracer per span family

Up to and including ADK 1.31, each span family was created from its own module-level `tracer` attribute, which the instrumentor can monkey-patch independently:

| span family                                 | module                              | attribute                |
|---------------------------------------------|-------------------------------------|--------------------------|
| `invoke_agent {name}`                       | `agents/base_agent.py`              | `base_agent.tracer`      |
| `execute_tool {name}` and `(merged)`        | `flows/llm_flows/functions.py`      | `functions.tracer`       |
| `call_llm`, `send_data`                     | `flows/llm_flows/base_llm_flow.py`  | `base_llm_flow.tracer`   |
| `generate_content {model}` (experimental)   | `telemetry/tracing.py`              | `tracing.tracer`         |
| inner `invocation`                          | `runners.py`                        | `runners.tracer`         |

Each attribute drove exactly one span family, so per-family decisions (emit as OI span vs. suppress vs. enrich) reduced to picking the right replacement object for each attribute.

### What ADK 1.32 changed

Commit [`6942aac5`](https://github.com/google/adk-python/commit/6942aac5d7b1f465c20febe2a48bac90da32c4eb) ("feat: add native OpenTelemetry agentic metrics") consolidated tool and agent telemetry into [`telemetry/_instrumentation.py`](https://github.com/google/adk-python/blob/v1.32.0/src/google/adk/telemetry/_instrumentation.py). Two named async context managers, [`record_agent_invocation`](https://github.com/google/adk-python/blob/v1.32.0/src/google/adk/telemetry/_instrumentation.py#L74-L107) and [`record_tool_execution`](https://github.com/google/adk-python/blob/v1.32.0/src/google/adk/telemetry/_instrumentation.py#L110-L164), now drive both `invoke_agent {name}` and `execute_tool {name}`. Both open spans via the same [`tracing.tracer.start_as_current_span(...)`](https://github.com/google/adk-python/blob/v1.32.0/src/google/adk/telemetry/_instrumentation.py#L83) call; `record_tool_execution` additionally calls [`tracing.trace_tool_call(...)`](https://github.com/google/adk-python/blob/v1.32.0/src/google/adk/telemetry/_instrumentation.py#L136-L141) internally.

Other 1.32 moves:

- `tracer` removed from `agents/base_agent.py` (the `invoke_agent` span path no longer has a per-module attribute).
- `trace_tool_call` moved from `flows/llm_flows/functions.py` to [`telemetry/tracing.py`](https://github.com/google/adk-python/blob/v1.32.0/src/google/adk/telemetry/tracing.py#L168).

### Implication: one shared tracer, three different desired behaviors

On ADK ≥ 1.32, `tracing.tracer` drives three span families that the OI instrumentor wants to handle differently:

| span family                  | desired behavior                                                         |
|------------------------------|--------------------------------------------------------------------------|
| `invoke_agent {name}`        | suppress — `_BaseAgentRunAsync` already produces `agent_run [name]`      |
| `generate_content {model}`   | suppress — `_TraceCallLlm` already produces `call_llm`                   |
| `execute_tool {name}`        | emit as OI span — `_TraceToolCall` enriches it with TOOL attributes      |

Because all three families flow through one tracer, the integration primitive must dispatch by span name. Two simpler primitives don't work:

- A blanket [`_PassthroughTracer`](src/openinference/instrumentation/google_adk/__init__.py#L238-L254) suppresses all three. The `execute_tool {name}` span is not created, and [`_TraceToolCall`](src/openinference/instrumentation/google_adk/_wrappers.py#L290-L345) (which writes attributes via `get_current_span()`) ends up attaching TOOL attributes to whatever the active span happens to be — typically the parent `call_llm` span. End-to-end tests that assert the existence of an `execute_tool {tool.name}` span will fail.
- An `OITracer` swap emits all three as OI spans. That produces a duplicate `invoke_agent` span alongside `agent_run`, and a duplicate `generate_content` span alongside `call_llm`.

### The selective-tracer primitive

[`_SelectiveExecuteToolTracer`](src/openinference/instrumentation/google_adk/__init__.py#L257-L320) is a `wrapt.ObjectProxy` holding both the original ADK tracer (the wrapped object) and the OI tracer (`_self_oi_tracer`). It overrides `start_as_current_span(name, ...)` to dispatch by name prefix:

- `name.startswith("execute_tool")` → forward to the OI tracer (real OI span).
- everything else → yield the current span (suppress).

Effectively, the proxy re-creates at runtime the per-family granularity that pre-1.32 had via separate module attributes. Both `tracing.tracer` and `functions.tracer` are wrapped by it on ADK ≥ 1.32 in [`_disable_existing_tracers`](src/openinference/instrumentation/google_adk/__init__.py#L162-L192).

### Why `functions.tracer` is patched in addition to `tracing.tracer`

[`flows/llm_flows/functions.py:47`](https://github.com/google/adk-python/blob/v1.32.0/src/google/adk/flows/llm_flows/functions.py#L47) captures the tracer in a local binding at import time:

```python
from ...telemetry.tracing import tracer  # captured at IMPORT time
```

That local name is unaffected by later reassignments of `tracing.tracer`. It is still used for parallel-call `execute_tool (merged)` spans at [`functions.py:429`](https://github.com/google/adk-python/blob/v1.32.0/src/google/adk/flows/llm_flows/functions.py#L429) and [`functions.py:658`](https://github.com/google/adk-python/blob/v1.32.0/src/google/adk/flows/llm_flows/functions.py#L658):

```python
with tracer.start_as_current_span('execute_tool (merged)'):
    trace_merged_tool_calls(...)
```

If only `tracing.tracer` is wrapped, those merged-call spans escape as raw native ADK spans. Wrapping `functions.tracer` independently with the same proxy keeps the parallel-call path consistent with the per-call path. [`test_google_adk_instrumentor_parallel_tool_calls`](tests/test_instrumentor.py#L1507) exercises this branch.

---

## Architecture options if support for ADK < 1.32 is dropped

### Inventory of telemetry call sites in ADK 1.32

| ADK span                  | Source                                                                                                                                                                                                                                                                                                                                | Today's strategy                                                |
|---------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------|
| `invocation`              | [`runners.py:545`](https://github.com/google/adk-python/blob/v1.32.0/src/google/adk/runners.py#L545) via `runners.tracer`                                                                                                                                                                                                              | Outer-wrap `Runner.run_async` + `_PassthroughTracer` on inner   |
| `invoke_agent {name}`     | [`base_agent.py:288`](https://github.com/google/adk-python/blob/v1.32.0/src/google/adk/agents/base_agent.py#L288) / [`:320`](https://github.com/google/adk-python/blob/v1.32.0/src/google/adk/agents/base_agent.py#L320) via `record_agent_invocation` → `tracing.tracer`                                                              | Outer-wrap `BaseAgent.run_async` + selective passthrough        |
| `call_llm`                | [`base_llm_flow.py:1172`](https://github.com/google/adk-python/blob/v1.32.0/src/google/adk/flows/llm_flows/base_llm_flow.py#L1172) via `base_llm_flow.tracer`                                                                                                                                                                          | `base_llm_flow.tracer = OITracer` + wrap `trace_call_llm`       |
| `send_data`               | [`base_llm_flow.py:531`](https://github.com/google/adk-python/blob/v1.32.0/src/google/adk/flows/llm_flows/base_llm_flow.py#L531) via `base_llm_flow.tracer`                                                                                                                                                                            | **Accidentally an OI span, never enriched** — silent gap        |
| `execute_tool {name}`     | [`functions.py:595`](https://github.com/google/adk-python/blob/v1.32.0/src/google/adk/flows/llm_flows/functions.py#L595) / [`:830`](https://github.com/google/adk-python/blob/v1.32.0/src/google/adk/flows/llm_flows/functions.py#L830) via `record_tool_execution` → `tracing.tracer`                                                | `_SelectiveExecuteToolTracer` + wrap `trace_tool_call`          |
| `execute_tool (merged)`   | [`functions.py:429`](https://github.com/google/adk-python/blob/v1.32.0/src/google/adk/flows/llm_flows/functions.py#L429) / [`:658`](https://github.com/google/adk-python/blob/v1.32.0/src/google/adk/flows/llm_flows/functions.py#L658) via `functions.tracer`                                                                         | `_SelectiveExecuteToolTracer` (no enricher → untyped OI span)   |
| `generate_content {model}`| [`tracing.py:720`](https://github.com/google/adk-python/blob/v1.32.0/src/google/adk/telemetry/tracing.py#L720) / [`:764`](https://github.com/google/adk-python/blob/v1.32.0/src/google/adk/telemetry/tracing.py#L764) via `tracing.tracer` (experimental)                                                                              | Suppressed via selective passthrough                            |

### Span-by-span analysis

1. **`invocation` (runners.py)** — outer-wrap of `Runner.run_async` is the only option that has access to typed kwargs (`user_id`, `session_id`, `new_message`). The inner ADK `invocation` span has no need to be converted; suppressing it via `runners.tracer = _PassthroughTracer` is the right shape regardless of ADK version.
2. **`invoke_agent`** — outer-wrap of `BaseAgent.run_async` carries an advantage that wrapping `record_agent_invocation` does not: async-generator iteration gives access to per-event hooks (e.g. `is_final_response()`) for capturing OUTPUT_VALUE. The dedicated `invoke_agent` span is then a duplicate, suppressed via the selective-tracer's passthrough branch.
3. **`call_llm`** — no equivalent of `record_*` exists for the LLM path. ADK opens the `call_llm` span before any of its `trace_*` enrichers fire, so the only way to get OI semantics is to coerce the existing span by patching `base_llm_flow.tracer = OITracer` and wrapping `trace_call_llm` to enrich it via `get_current_span()`.
4. **`send_data`** — currently coerced to an OI span as a side effect of the `base_llm_flow.tracer = OITracer` swap (necessary for `call_llm`), but no enricher is registered for it, so the resulting OI span has no `openinference.span.kind` and no input/output attributes. Live/audio users would observe these as untyped OI spans. Two viable handlings: write a `_TraceSendData` enricher, or wrap `start_as_current_span("send_data")` into a passthrough.
5. **`execute_tool {name}`** — the strongest case for switching primitives. `_instrumentation.record_tool_execution` exposes typed kwargs (`tool`, `agent`, `invocation_context`, `function_args`) and yields a `TelemetryContext` whose `function_response_event` carries the tool output. Replacing this function directly removes the need for `_SelectiveExecuteToolTracer`, the `tracing.trace_tool_call` wrap, `_TraceToolCall`'s `bind_args_kwargs` reflection, and the `functions.tracer` second-binding patch.
6. **`execute_tool (merged)`** — ADK's own comment calls this a debug aid for parallel calls. Today it becomes an untyped OI span. Three handlings: suppress it (per-call OI tool spans already cover the user-facing story), write a `_TraceMergedToolCalls` enricher, or accept it as a native ADK span.
7. **`generate_content`** — experimental-semconv-only path. Suppression continues to be appropriate; `_TraceCallLlm` covers the LLM path.

### Shape of a 1.32-only redesign

A 1.32-only architecture would collapse to three primitives, each justified by the span family it serves:

- **Outer-method wrap** (`Runner.run_async`, `BaseAgent.run_async`) — for the span families that need typed kwargs and per-event iteration.
- **OITracer swap + `trace_*` enricher** (`base_llm_flow.tracer` + `trace_call_llm`) — for `call_llm`, where ADK opens the span before any hook fires.
- **Named-function replacement** (`_instrumentation.record_tool_execution`) — for spans where ADK exposes a typed context-manager hook.

Items that would no longer be needed:

- `_SelectiveExecuteToolTracer`.
- The `_TraceToolCall` decorator and the `bind_args_kwargs` reflection it relies on.
- The `tracing.trace_tool_call` wrap.
- The `functions.tracer` patch (and the import-time-binding consideration that motivates it).
- Version gating throughout `_disable_existing_tracers` and adjacent methods.

Items that would need an explicit decision: the `send_data` and `execute_tool (merged)` paths, which today fall through to "untyped OI span" by accident.

The runner/agent/LLM paths would not change shape; the saving is concentrated in the tool-execution path.

### Tradeoffs

1. While support for ADK 1.2.1+ is maintained, a 1.32-only redesign would require either bifurcating the package by version or running both architectures behind a version gate during a deprecation window — both add complexity rather than reducing it.
2. The current architecture covers the span families in the inventory above on both the pinned and `-latest` tox envs.
3. The structural simplification a 1.32-only design unlocks is concentrated in the tool-execution path; the runner / agent / LLM paths would be roughly the same shape under either approach.

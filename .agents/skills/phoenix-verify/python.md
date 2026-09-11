# Phoenix Verify: Python

Read this file only when the instrumentor under test is Python. `$SCRATCH` is your session
scratchpad; `<pkg>` is the short name (`openai`, `ag2`), `<mod>` the module under
`openinference.instrumentation`.

## Where examples live

`python/instrumentation/openinference-instrumentation-<pkg>/examples/<scenario>.py`, with
`requirements.txt` and usually a `README.md` beside them. Eight packages ship `examples/` with
no `requirements.txt` (`groq`, `mcp`, `instructor`, `google-adk`, `agent-framework`,
`agentspec`, `pipecat`, `promptflow`; tracked in #3724).

## Setup, proof, run

The package needs a venv on its `requires-python` (usually >= 3.10). Tox venvs install
non-editably and do not see working-tree edits, so make a fresh one:

```bash
uv venv --python 3.12 "$SCRATCH/venv-<pkg>"
uv pip install --python "$SCRATCH/venv-<pkg>/bin/python" \
  -e python/instrumentation/openinference-instrumentation-<pkg> \
  -r python/instrumentation/openinference-instrumentation-<pkg>/examples/requirements.txt
"$SCRATCH/venv-<pkg>/bin/python" -c 'import openinference.instrumentation.<mod> as m; print(m.__file__)'
# proof: a path under the repo's src/ (a PyPI "before" venv prints site-packages instead)
cd python/instrumentation/openinference-instrumentation-<pkg>/examples && "$SCRATCH/venv-<pkg>/bin/python" <scenario>.py
```

Install `-e` and the requirements in one command: `requirements.txt` lists the package from
PyPI and would shadow the editable install if installed afterwards. A missing `-r` file aborts
the whole command, including the editable install; for the packages without one, drop `-r` and
add `opentelemetry-sdk opentelemetry-exporter-otlp-proto-http` plus the provider SDK the example
imports.

## Example template

Plain OpenTelemetry SDK only, the shape of
`openinference-instrumentation-ag2/examples/no_llm_multi_agent.py`:

```python
"""
<One paragraph: what this traces and what the span tree should look like.>

1. Run a local OTLP collector such as Phoenix: `pip install arize-phoenix && phoenix serve`
2. Install dependencies: `pip install -r requirements.txt`
3. Run this example: `python <scenario>.py`
4. View the traces at http://localhost:6006 under the `<pkg>-<scenario>` project.
"""

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.<mod> import <Pkg>Instrumentor
from openinference.semconv.resource import ResourceAttributes

endpoint = "http://localhost:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider(
    resource=Resource({ResourceAttributes.PROJECT_NAME: "<pkg>-<scenario>"})
)
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
<Pkg>Instrumentor().instrument(tracer_provider=tracer_provider)


def main() -> None:
    ...  # the minimal call that exercises the changed code path


if __name__ == "__main__":
    main()
```

### Scratchpad copy for before/after runs

Copy the example to `$SCRATCH` and take the project name from the command line, so both runs
use identical code and the committed example is untouched:

```python
import sys

project_name = sys.argv[1] if len(sys.argv) > 1 else "<pkg>-<scenario>"
tracer_provider = trace_sdk.TracerProvider(
    resource=Resource({ResourceAttributes.PROJECT_NAME: project_name})
)
```

### Context attributes and suppression

Both helpers come from `openinference.instrumentation`. Make one traced call and one suppressed
call in the same run so "zero spans" is provable as "exactly one span":

```python
from openinference.instrumentation import suppress_tracing, using_attributes

with using_attributes(
    session_id="verify-session-1",
    user_id="verify-user",
    metadata={"run": "after"},
    tags=["verify"],
):
    ...  # the instrumented call -> one span with context attributes

with suppress_tracing():
    ...  # the same call -> no span
```

Check (metadata is flattened to `metadata.<key>`):

```bash
jq length "$F"   # expect 1
jq '.[] | .attributes | with_entries(select(.key | startswith("session.") or startswith("user.") or startswith("metadata.") or startswith("tag.")))' "$F"
```

## Getting a "before" build

Use two venvs so the runs cannot contaminate each other, and print `<module>.__file__` in each:
the "after" venv must show the repo's `src/`, a PyPI "before" venv shows `site-packages`. Both
venvs follow the one-command install rule above.

| Situation | "Before" |
| --- | --- |
| Fix not yet applied | Run, apply the change, run again in the same editable venv |
| Fix already in the working tree or branch | `git worktree add "$SCRATCH/wt-main" origin/main`, `uv pip install -e "$SCRATCH/wt-main/python/instrumentation/openinference-instrumentation-<pkg>"` into `venv-before` |
| Parity with the last release | `uv pip install openinference-instrumentation-<pkg>` (PyPI) into `venv-before`; read the installed version with `uv pip show --python <venv>/bin/python openinference-instrumentation-<pkg>`. The tag is `python-openinference-instrumentation-<pkg>-vX.Y.Z`; `git diff --stat <tag> -- python/instrumentation/openinference-instrumentation-<pkg>/src` shows whether any difference is even possible |

```bash
"$SCRATCH/venv-before/bin/python" "$SCRATCH/<scenario>.py" <pkg>-<scenario>-before
"$SCRATCH/venv-after/bin/python"  "$SCRATCH/<scenario>.py" <pkg>-<scenario>-after
```

## Wiring a new example into the repo

- Add a row to `examples/README.md` (create it if missing; the ag2 package shows the layout):
  `| [`<scenario>.py`](<scenario>.py) | `<pkg>-<scenario>` | Yes/No | <what it traces, one clause> |`.
  Or a `## Examples` section in the package README if you prefer prose.
- A package gaining its first `examples/` directory also needs a row in the root `README.md`
  "Examples" table (`python/DEVELOPMENT.md`, "Update the root README").
- New direct imports go in `examples/requirements.txt`. Existing examples rely on the
  instrumentor's transitive dependencies for `opentelemetry-sdk` and
  `openinference-instrumentation`; list only genuinely new ones.
- Lint as CI does, from the package directory with the pinned ruff: `tox run -e ruff-<pkg>` or
  `uvx ruff@0.9.2 format --diff . && uvx ruff@0.9.2 check --no-fix .` (an unpinned ruff reports
  false import-order errors).

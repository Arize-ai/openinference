# Phoenix Verify: Reference

Templates and recipes for the workflow in [SKILL.md](SKILL.md). Copy, rename the project,
and trim to the path under test. `$SCRATCH` is your session scratchpad directory.

## Example templates

Plain OpenTelemetry SDK only: a `TracerProvider` with the project pinned on the resource, an
OTLP HTTP exporter, and a `SimpleSpanProcessor`. This is the shape of the committed examples
(for instance `python/instrumentation/openinference-instrumentation-ag2/examples/no_llm_multi_agent.py`),
so a committed example stays a plain, self-contained script. To redirect a run to another
project or host, copy the example to `$SCRATCH` and edit the two constants in the copy.

### Python

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

`examples/requirements.txt` in this repo usually lists the instrumentor itself from PyPI. Install
it together with `-e` in one `uv pip install` command so the editable install wins, then print
`<module>.__file__` to prove the working tree is loaded.

### JS / TypeScript

Check the package's `@opentelemetry/resources` major first. Most instrumentation packages pin
1.x and use `new Resource({...})` (as in
`js/packages/openinference-instrumentation-openai/examples/manual-instrumentation.ts`);
`openinference-core` and `openinference-genai` are on 2.x and use `resourceFromAttributes`.

```ts
import { SEMRESATTRS_PROJECT_NAME } from "@arizeai/openinference-semantic-conventions";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";
import { Resource } from "@opentelemetry/resources"; // 2.x: import { resourceFromAttributes }
import { NodeTracerProvider, SimpleSpanProcessor } from "@opentelemetry/sdk-trace-node";

import { <Pkg>Instrumentation } from "../src";

const provider = new NodeTracerProvider({
  resource: new Resource({ [SEMRESATTRS_PROJECT_NAME]: "<pkg>-<scenario>" }), // 2.x: resourceFromAttributes({...})
  spanProcessors: [
    new SimpleSpanProcessor(new OTLPTraceExporter({ url: "http://localhost:6006/v1/traces" })),
  ],
});
provider.register();
new <Pkg>Instrumentation().manuallyInstrument(<module>); // or registerInstrumentations

async function main() {
  ...
  await provider.forceFlush(); // a pending export keeps Node alive, but process.exit() would drop it
}
main();
```

Packages whose examples share an `instrumentation.ts` bootstrap: copy the bootstrap and the
example to `$SCRATCH` together and change the project name in the copy.

Setup and run, from `js/`:

```bash
pnpm install --frozen-lockfile -r
pnpm --filter "@arizeai/openinference-instrumentation-<pkg>..." run build   # package + workspace deps
cd packages/openinference-instrumentation-<pkg> && pnpm exec tsx examples/<scenario>.ts
```

### Scratchpad copy for before/after runs

One copy, project name from the command line, so both runs use identical code:

```python
import sys

project_name = sys.argv[1] if len(sys.argv) > 1 else "<pkg>-<scenario>"
tracer_provider = trace_sdk.TracerProvider(
    resource=Resource({ResourceAttributes.PROJECT_NAME: project_name})
)
```

### Context attributes and suppression

Both helpers come from the same module. Make one traced call and one suppressed call in the
same run so "zero spans" is provable as "exactly one span".

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
the "after" venv must show the repo's `src/`, a PyPI "before" venv shows `site-packages`.
Install `-r examples/requirements.txt` in the same command in both venvs.

| Situation | "Before" |
| --- | --- |
| Fix not yet applied | Run, apply the change, run again in the same editable venv |
| Fix already in the working tree or branch | `git worktree add "$SCRATCH/wt-main" origin/main`, `uv pip install -e "$SCRATCH/wt-main/python/instrumentation/<pkg>"` into `venv-before` |
| Parity with the last release | `uv pip install openinference-instrumentation-<pkg>` (PyPI) into `venv-before`; read the installed version with `uv pip show --python <venv>/bin/python openinference-instrumentation-<pkg>`. The tag is `python-openinference-instrumentation-<pkg>-vX.Y.Z`; `git diff --stat <tag> -- python/instrumentation/openinference-instrumentation-<pkg>/src` shows whether any difference is even possible |

Run the same scratchpad copy into both projects:

```bash
"$SCRATCH/venv-before/bin/python" "$SCRATCH/<scenario>.py" <pkg>-<scenario>-before
"$SCRATCH/venv-after/bin/python"  "$SCRATCH/<scenario>.py" <pkg>-<scenario>-after
```

## README wiring

Create `examples/README.md` if the package has none; the ag2 package shows the full layout.
Table row:

```markdown
| [`<scenario>.py`](<scenario>.py) | `<pkg>-<scenario>` | Yes/No | <what it traces, one clause> |
```

Package `README.md` section, if you prefer prose:

```markdown
## Examples

Runnable examples that send traces to a local Phoenix at `http://localhost:6006` live in
[`examples/`](examples). `<scenario>.py` exercises <path>; it needs `<PROVIDER>_API_KEY`.
```

JS packages have no `requirements.txt`; new example dependencies go in `devDependencies`.
Existing example files rely on the instrumentor's transitive dependencies for
`opentelemetry-sdk` and `openinference-instrumentation`; list only genuinely new direct imports.

## jq recipes

Save per project first, then filter. A jq `Invalid numeric literal` error means px wrote an
error message, not JSON: check the project name and that Phoenix is up.

```bash
F="$SCRATCH/<project>.spans.json"
px span list --project <project> --format raw --no-progress --limit 500 > "$F" && jq length "$F"
```

| Goal | Command |
| --- | --- |
| Span tree | `span_tree.sh <project>` |
| Attribute keys per span | `span_tree.sh <project> keys` |
| Attribute values, volatile keys dropped | `span_tree.sh <project> values` |
| ERROR spans with messages | `span_tree.sh <project> errors` |
| Name, kind, status per span | `jq -r '.[] \| "\(.name)\t\(.span_kind)\t\(.status_code)"' "$F"` |
| Parent name per span | `jq -r '. as $s \| ($s \| map({key: .context.span_id, value: .name}) \| from_entries) as $n \| $s[] \| "\(.name) <- \($n[.parent_id // ""] // "<root>")"' "$F"` |
| One span's full attributes | `jq '.[] \| select(.name == "<name>") \| .attributes' "$F"` |
| Spans missing an attribute | `jq '.[] \| select(.attributes["<key>"] == null) \| .name' "$F"` |
| Spans of one kind | `px span list --project <p> --span-kind LLM ...` or `jq 'map(select(.span_kind == "LLM"))' "$F"` |
| Input/output messages | `jq '.[] \| select(.span_kind == "LLM") \| .attributes \| with_entries(select(.key \| startswith("llm.input_messages") or startswith("llm.output_messages")))' "$F"` |
| Token counts | `jq '.[] \| select(.span_kind == "LLM") \| .attributes \| with_entries(select(.key \| startswith("llm.token_count")))' "$F"` |
| Tool calls | `jq '.[] \| .attributes \| with_entries(select(.key \| contains("tool_call")))' "$F"` |
| Distinct traces | `jq '[.[].context.trace_id] \| unique \| length' "$F"` |
| Exactly one span (suppression proof) | `jq length "$F"` prints `1`; a 404 from px means the project never received a span |

### Before / after diff

```bash
S=.agents/skills/phoenix-verify/scripts/span_tree.sh
for m in tree keys values; do
  diff <($S <pkg>-<scenario>-before $m 2>/dev/null) <($S <pkg>-<scenario>-after $m 2>/dev/null) && echo "$m: identical"
done
```

Check `jq length` on both projects first; `2>/dev/null` hides the script's span count and its
"no spans" error, so an empty side would show up as a one-sided diff. `diff` exits 0 when the
two projects match. For a fix, the diff should show exactly the intended change and nothing
else. For a parity or regression check, "identical" on all three modes is the answer.
`values` mode drops keys that legitimately vary between runs (`output.value`,
`llm.output_messages.*`, `llm.token_count.*`, `llm.finish_reason`); if the claim is about one
of those, use a targeted jq filter instead. If a `values` diff is confined to another
model-dependent key, confirm against the raw responses in the two run logs and say so in the
report rather than calling it an instrumentor change.
Paste the diff into the PR body under the exact commands that produced it.

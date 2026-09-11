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

## MCP path

When the Phoenix MCP server is connected (tool `mcp__plugin_arize-phoenix_phoenix__execute`,
code mode with `call_tool`), you do not need `px` or `jq`. `getSpans` returns the same span
objects as `px span list --format raw` (`name`, `span_kind`, `status_code`, `status_message`,
`parent_id`, `context.span_id`, `start_time`, flattened `attributes`). A project that never
received a span raises `HTTP error 404`. `getSpans` also takes `span_kind`, `name`,
`status_code`, `trace_id`, and `attribute: ["key:value"]` filters.

Connectivity: `await call_tool("getProjects", {})`. Project exists: `await call_tool("getProject", {"project_identifier": "<project>"})`.

Read back one project, or diff two, with this block (edit the three constants):

```python
PROJECT = "<pkg>-<scenario>"
BEFORE = None   # e.g. "<pkg>-<scenario>-before"; leave None for a single project
AFTER = None    # e.g. "<pkg>-<scenario>-after"
import re
VOLATILE = re.compile(r"^(output\.value|llm\.output_messages\..*|llm\.token_count\..*|llm\.finish_reason)$")

async def fetch(project):
    spans, cursor = [], None
    while True:
        r = await call_tool("getSpans", {"project_identifier": project, "limit": 100, **({"cursor": cursor} if cursor else {})})
        spans += r["data"]; cursor = r.get("next_cursor")
        if not cursor: break
    return sorted(spans, key=lambda s: s["start_time"])

def tree(spans):
    ids = {s["context"]["span_id"] for s in spans}; kids = {}
    for s in spans: kids.setdefault(s["parent_id"], []).append(s)
    out = []
    def walk(s, d):
        msg = f"  -- {s['status_message']}" if s.get("status_message") else ""
        out.append(f"{'  '*d}{s['name']} [{s['span_kind']}] {s['status_code']}{msg}")
        for k in kids.get(s["context"]["span_id"], []): walk(k, d + 1)
    for s in spans:
        if s["parent_id"] is None or s["parent_id"] not in ids: walk(s, 0)
    return out

def keys(spans):
    return [l for s in spans for l in ([f"{s['name']} [{s['span_kind']}]"] + [f"    {k}" for k in sorted(s["attributes"])])]

def values(spans):
    return [l for s in spans for l in ([f"{s['name']} [{s['span_kind']}] {s['status_code']}"]
            + [f"    {k}={str(v)[:200]}" for k, v in sorted(s["attributes"].items()) if not VOLATILE.match(k)])]

def errors(spans):
    e = [s for s in spans if s["status_code"] == "ERROR"]
    return [f"no ERROR spans ({len(spans)} checked)"] if not e else [
        f"{s['name']} [{s['span_kind']}]  status_message: {s.get('status_message', '')}  exception.message: {s['attributes'].get('exception.message', '')}" for s in e]

result = {}
spans = await fetch(PROJECT)
result[PROJECT] = {"count": len(spans), "tree": tree(spans), "keys": keys(spans), "errors": errors(spans)}
if BEFORE and AFTER:
    b, a = await fetch(BEFORE), await fetch(AFTER)
    diff = {}
    for mode, fn in (("tree", tree), ("keys", keys), ("values", values)):
        lb, la = fn(b), fn(a)
        diff[mode] = "identical" if lb == la else {"before_only": [x for x in lb if x not in la], "after_only": [x for x in la if x not in lb]}
    result["diff"] = {"before_count": len(b), "after_count": len(a), **diff}
return result
```

For a targeted check, filter server-side and return only what the claim needs:

```python
r = await call_tool("getSpans", {"project_identifier": "<project>", "span_kind": ["LLM"], "limit": 100})
return [{k: v for k, v in s["attributes"].items() if k.startswith("llm.token_count")} for s in r["data"]]
```

`executeSql` (read-only SQLite over `spans`, `traces`, `projects`) is a one-line alternative for
name, kind, and status:

```sql
SELECT s.name, s.span_kind, s.status_code FROM spans s
JOIN traces t ON s.trace_rowid = t.id JOIN projects p ON t.project_rowid = p.id
WHERE p.name = '<project>' ORDER BY s.start_time
```

To reuse `span_tree.sh` on MCP output, return `spans` from the block and write it to
`$SCRATCH/<project>.spans.json`; the script accepts that file in place of a project name.

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

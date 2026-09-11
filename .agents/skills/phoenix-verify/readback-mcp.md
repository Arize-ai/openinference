# Phoenix Verify: reading spans back through the Phoenix MCP server

When the Phoenix MCP server is connected (tool `mcp__plugin_arize-phoenix_phoenix__execute`,
code mode with `call_tool`), you do not need `px` or `jq`. `getSpans` returns the same span
objects as `px span list --format raw` (`name`, `span_kind`, `status_code`, `status_message`,
`parent_id`, `context.span_id`, `start_time`, flattened `attributes`). A project that never
received a span raises `HTTP error 404`. `getSpans` also takes `span_kind`, `name`,
`status_code`, `trace_id`, and `attribute: ["key:value"]` filters. The sandbox has a restricted
standard library: `json` and `re` import, `collections` does not.

Connectivity: `await call_tool("getProjects", {})`. Project exists: `await call_tool("getProject", {"project_identifier": "<project>"})`.

## Tree, keys, values, errors, and before/after diff

Set `PROJECT` for a single run, or `BEFORE` and `AFTER` for a diff (then `PROJECT` is not
fetched). The renderers print the same lines as `span_tree.sh` (JSON-encoded values,
`?`/`UNSET` fallbacks) so a diff made here is comparable to one made with px:

```python
PROJECT = None  # e.g. "<pkg>-<scenario>"
BEFORE = None   # e.g. "<pkg>-<scenario>-before"
AFTER = None    # e.g. "<pkg>-<scenario>-after"
import json, re
VOLATILE = re.compile(r"^(output\.value|llm\.output_messages\..*|llm\.token_count\..*|llm\.finish_reason)$")

async def fetch(project):
    spans, cursor = [], None
    while True:
        r = await call_tool("getSpans", {"project_identifier": project, "limit": 100, **({"cursor": cursor} if cursor else {})})
        spans += r["data"]; cursor = r.get("next_cursor")
        if not cursor: break
    return sorted(spans, key=lambda s: s["start_time"])

def head(s, status=False):
    h = f"{s['name']} [{s.get('span_kind') or '?'}]"
    return f"{h} {s.get('status_code') or 'UNSET'}" if status else h

def tree(spans):
    ids = {s["context"]["span_id"] for s in spans}; kids = {}
    for s in spans: kids.setdefault(s.get("parent_id"), []).append(s)
    out = []
    def walk(s, d):
        msg = f"  -- {s['status_message']}" if s.get("status_message") else ""
        out.append(f"{'  '*d}{head(s, status=True)}{msg}")
        for k in kids.get(s["context"]["span_id"], []): walk(k, d + 1)
    for s in spans:
        if s.get("parent_id") is None or s["parent_id"] not in ids: walk(s, 0)
    return out

def keys(spans):
    return [l for s in spans for l in ([head(s)] + [f"    {k}" for k in sorted(s.get("attributes") or {})])]

def values(spans):
    return [l for s in spans for l in ([head(s, status=True)]
            + [f"    {k}={json.dumps(v, separators=(',', ':'))[:200]}" for k, v in sorted((s.get("attributes") or {}).items()) if not VOLATILE.match(k)])]

def errors(spans):
    e = [s for s in spans if s.get("status_code") == "ERROR"]
    return [f"no ERROR spans ({len(spans)} checked)"] if not e else [l for s in e for l in (
        head(s), f"    status_message: {s.get('status_message') or ''}", f"    exception.message: {(s.get('attributes') or {}).get('exception.message', '')}")]

def multiset_minus(x, y):  # lines of x not in y, counting repeats; the sandbox has no collections module
    n = {}
    for l in y: n[l] = n.get(l, 0) + 1
    out = []
    for l in x:
        if n.get(l, 0): n[l] -= 1
        else: out.append(l)
    return out

result = {}
if PROJECT:
    spans = await fetch(PROJECT)
    result[PROJECT] = {"count": len(spans), "tree": tree(spans), "keys": keys(spans), "values": values(spans), "errors": errors(spans)}
if BEFORE and AFTER:
    b, a = await fetch(BEFORE), await fetch(AFTER)
    diff = {}
    for mode, fn in (("tree", tree), ("keys", keys), ("values", values)):
        lb, la = fn(b), fn(a)
        diff[mode] = "identical" if lb == la else {"before_only": multiset_minus(lb, la), "after_only": multiset_minus(la, lb)}
    result["diff"] = {"before_count": len(b), "after_count": len(a), **diff}
return result
```

## Targeted checks

Filter server-side and return only what the claim needs:

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
`$SCRATCH/<project>.spans.json`; the script reads any first argument ending in `.json` as a
saved span array instead of a project name.

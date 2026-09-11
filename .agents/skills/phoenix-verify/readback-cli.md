# Phoenix Verify: reading spans back with px and jq

Fetch once per project to a file, check the count, then render every view from the file so all
views see the same spans. A jq `Invalid numeric literal` error means px wrote an error message,
not JSON: check the project name and that Phoenix is up.

```bash
S="$(git rev-parse --show-toplevel)/.agents/skills/phoenix-verify/scripts/span_tree.sh"
F="$SCRATCH/<project>.spans.json"
px span list --project <project> --format raw --no-progress --limit 500 > "$F" && jq length "$F"
$S "$F"            # name [KIND] status, nested
$S "$F" keys       # attribute keys per span
$S "$F" values     # attribute values minus output.value, llm.output_messages.*, llm.token_count.*, llm.finish_reason
$S "$F" errors     # ERROR spans + status_message + exception.message
$S <project> errors -- --last-n-minutes 10 --limit 500    # live fetch instead; any px span list flags after --
```

Keep the `.json` suffix on saved files; that is how the script tells a file from a project
name. The span count goes to stderr so it never pollutes a diff.

## jq recipes

| Goal | Command |
| --- | --- |
| Name, kind, status per span | `jq -r '.[] \| "\(.name)\t\(.span_kind)\t\(.status_code)"' "$F"` |
| Parent name per span | `jq -r '. as $s \| ($s \| map({key: .context.span_id, value: .name}) \| from_entries) as $n \| $s[] \| "\(.name) <- \($n[.parent_id // ""] // "<root>")"' "$F"` |
| One span's full attributes | `jq '.[] \| select(.name == "<name>") \| .attributes' "$F"` |
| Spans missing an attribute | `jq '.[] \| select(.attributes["<key>"] == null) \| .name' "$F"` |
| Spans of one kind | `px span list --project <p> --span-kind LLM ...` or `jq 'map(select(.span_kind == "LLM"))' "$F"` |
| Input/output messages | `jq '.[] \| select(.span_kind == "LLM") \| .attributes \| with_entries(select(.key \| startswith("llm.input_messages") or startswith("llm.output_messages")))' "$F"` |
| Token counts | `jq '.[] \| select(.span_kind == "LLM") \| .attributes \| with_entries(select(.key \| startswith("llm.token_count")))' "$F"` |
| Tool calls | `jq '.[] \| .attributes \| with_entries(select(.key \| contains("tool_call")))' "$F"` |
| Context attributes | `jq '.[] \| .attributes \| with_entries(select(.key \| startswith("session.") or startswith("user.") or startswith("metadata.") or startswith("tag.")))' "$F"` |
| Distinct traces | `jq '[.[].context.trace_id] \| unique \| length' "$F"` |
| Exactly one span (suppression proof) | `jq length "$F"` prints `1` (a 404 is the zero-span case, see Gotchas in SKILL.md) |

## Before / after diff

Fetch each project once, prove both files hold spans, then diff the rendered views. Two failed
or empty sides would otherwise diff as "identical", which is the parity answer.

```bash
B="$SCRATCH/<project>-before.spans.json"; A="$SCRATCH/<project>-after.spans.json"
px span list --project <project>-before --format raw --no-progress --limit 500 > "$B"
px span list --project <project>-after  --format raw --no-progress --limit 500 > "$A"
jq -e 'length > 0' "$B" >/dev/null && jq -e 'length > 0' "$A" >/dev/null || { echo "a side is empty or not JSON"; exit 1; }
for m in tree keys values; do
  diff <($S "$B" $m) <($S "$A" $m) && echo "$m: identical"
done
```

`diff` exits 0 when the two projects match. For a fix, the diff should show exactly the
intended change and nothing else. For a parity or regression check, "identical" on all three
modes is the answer. `values` mode drops keys that legitimately vary between runs
(`output.value`, `llm.output_messages.*`, `llm.token_count.*`, `llm.finish_reason`); if the
claim is about one of those, use a targeted jq filter instead. If a `values` diff is confined to
another model-dependent key, confirm against the raw responses in the two run logs and say so
in the report rather than calling it an instrumentor change. Paste the diff into the PR body
under the exact commands that produced it.

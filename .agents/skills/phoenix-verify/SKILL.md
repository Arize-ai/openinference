---
name: phoenix-verify
description: Prove an OpenInference instrumentation change end to end by running a package example against a local Phoenix (http://localhost:6006) in a dedicated project and inspecting the resulting spans with the px CLI or the Phoenix MCP server. Use when an instrumentation fix or feature in any language (Python, JS, Java) changes what a user sees in Phoenix (span tree, span kind, status, attributes, sessions, metadata), when the user asks for evidence in Phoenix, a before/after comparison, or a round trip, or when adding a runnable example to a package.
---

# Phoenix Verify

Unit tests gate the push. When a change alters what a user sees in the Phoenix UI, pair the
tests with a real round trip: run an example that exports to local Phoenix, read the spans
back, and check one specific claim. The output of that check is the evidence that goes in the
PR or summary. The workflow is the same for every language; only the environment setup differs, so read
exactly one language file: [python.md](python.md), [javascript.md](javascript.md), or
[java.md](java.md). Two read-back paths return the same span JSON: the `px` CLI with `jq`
([readback-cli.md](readback-cli.md)), or the Phoenix MCP server
(`mcp__plugin_arize-phoenix_phoenix__execute`, [readback-mcp.md](readback-mcp.md)) when it is
connected. Read only the one you will use.

## Prerequisites

- **Phoenix reachable.** CLI: `px project list --format raw --no-progress > /dev/null && echo "phoenix ok at ${PHOENIX_HOST:-http://localhost:6006}"` must print the ok line. MCP: `call_tool("getProjects", {})` must return `data`. If either errors (`fetch failed`), stop and report NOT VERIFIED (template in step 7). `px auth status` also works as a check: it exits 5 and reports `"status":"unverified"` when the server is unreachable, but its output mixes auth state with connectivity, so the `project list` line is the simpler yes/no. Default host is `http://localhost:6006` (OTLP HTTP on `/v1/traces`, gRPC on `4317`); set `PHOENIX_HOST` for px if different, and point the example's exporter at the same host. Start one with `pip install arize-phoenix && phoenix serve` or `docker run -p 6006:6006 -p 4317:4317 arizephoenix/phoenix:latest`.
- **`px` and `jq` on PATH** (`npx @arizeai/phoenix-cli` also works), or the Phoenix MCP server connected. `--attribute` filters need Phoenix server >= 14.9.0.
- **Provider API keys** exported when the example calls a model. Prefer an offline example (canned replies, tool-only) when the claim does not depend on model output. Some packages have none; say so rather than inventing one.

## Workflow

Copy this checklist and work through it in order. `$SCRATCH` is your session scratchpad directory. `<pkg>` is the short instrumentor name (`openai`, `langchain4j`, `ag2`); `<scenario>` is the example's filename stem.

1. **State the claim.** One sentence naming the span, attribute, kind, or status that should be a certain way. Everything below proves or disproves this sentence. Decide the mode now: **single run** (verify existing behavior, or a new feature) or **before/after** (a fix or a parity check).
2. **Pick or write the example.** Examples live next to the package (location in your language file). Reuse an existing example that exercises the path; otherwise write one from the template in your language file. Examples use the plain OpenTelemetry SDK: a `TracerProvider` whose resource carries `openinference.project.name`, an OTLP exporter pointed at local Phoenix, and either a synchronous processor or an explicit flush before exit. New examples name the project `<pkg>-<scenario>`; existing ones mostly set another name or none at all, so read `<project>` from the example's resource (or its shared bootstrap), never from the filename. To run into a different project (before/after) or host, or when an example sets no project resource (spans land in `default`) or shares a bootstrap file, copy it to `$SCRATCH` and edit the copy; do not rewrite the committed example just to run it.
3. **Set up an isolated environment and prove it runs the working tree, not a release.** Your language file has one setup command and one proof command: Python is an editable install in a fresh venv proven by the module's `__file__`; JS is a workspace build proven by examples importing `../src`; Java is the Gradle composite build proven by the example's dependency on the sibling project. For before/after, keep two environments; ways to get a "before" build are in the same file.
4. **Confirm the project is empty, then run.** `px span list --project <project> --format raw --no-progress 2>&1 | head -c 200` must show `[]` or a `404 Not Found` error (the project does not exist yet); over MCP, `getSpans` on the project must return empty `data` or raise the 404. Anything else means old spans are present: pick a new name, because they are not evidence. Run the example and read its output: an exit code of 0 does not mean spans were exported. `Failed to export`, `Connection refused`, or `UNAVAILABLE` in the log means Phoenix never received them. For before/after, run into `<project>-before` and `<project>-after`.
5. **Read the spans back.** Fetch once to a file, check the count, then render from the file so every view sees the same spans (the script's own fetch uses px's default of the newest 100):

   ```bash
   S="$(git rev-parse --show-toplevel)/.agents/skills/phoenix-verify/scripts/span_tree.sh"   # absolute: steps 3 and 8 run from package dirs
   F="$SCRATCH/<project>.spans.json"
   px span list --project <project> --format raw --no-progress --limit 500 > "$F" && jq length "$F"   # count first
   "$S" "$F"            # name [KIND] status, nested
   "$S" "$F" keys       # attribute keys per span
   "$S" "$F" values     # attribute values minus output.value, llm.output_messages.*, llm.token_count.*, llm.finish_reason (good for diffs)
   "$S" "$F" errors     # ERROR spans + exception.message
   "$S" <project> errors -- --last-n-minutes 10 --limit 500    # live fetch instead; any px span list flags after --
   ```

   Name saved files after the project and keep the `.json` suffix (that is how the script tells a file from a project name). A shared name like `spans.json` gets overwritten by a parallel run. If jq reports `Invalid numeric literal`, px printed an error, not JSON; the script exits 1 with the same diagnosis. Targeted jq recipes and the before/after diff are in [readback-cli.md](readback-cli.md). Without px, use [readback-mcp.md](readback-mcp.md) instead; it renders the same tree, keys, values, errors, and diff from `getSpans`.
6. **Judge the claim.** Compare the output to step 1. If it does not match, the fix is not done. Do not weaken the claim to fit the output. An empty before/after diff is the right answer for a parity claim and the wrong answer for a fix.
7. **Report the evidence.** Use one of these shapes in the PR body or final message:

   ```
   Verified against local Phoenix (project `<project>`):
   <exact commands>
   <trimmed output showing the claim>
   Before (`<project>-before`): <how it differed, or "identical">   # before/after mode only
   Builds: before=<proof output>, after=<proof output>              # before/after mode only

   NOT VERIFIED: <what failed, e.g. Phoenix unreachable, example exited 1, 0 spans exported>
   Phoenix: <host used, and how it was set>
   <command> -> <error text>
   Example ran: yes/no. Spans read back: <n>. Claim status: unverified | falsified.
   ```
8. **Wire the example into the repo** only if you wrote or changed one. Add it to the package's examples README (create one if missing) or a `## Examples` section in the package README; if this is the package's first `examples/` directory, also add a row to the root `README.md` "Examples" table (`python/DEVELOPMENT.md` requires it). Declare any new direct dependencies where that language keeps them, and lint and type-check exactly as CI does (commands in your language file). Never commit API keys or cloud endpoints.

## Gotchas

- **Flush before exit.** Synchronous processors export inline; failures are log-only and never change the exit code. Batch processors need an explicit `forceFlush` (and `shutdown` in Java) before the process ends, or the last spans never arrive. A pending export keeps Node alive, but `process.exit()` or an unhandled rejection drops it.
- **Metadata is flattened.** Context metadata lands as `metadata.<key>`, not a `metadata` key. Filter with `startswith("metadata.")`. Sessions, users, and tags are `session.id`, `user.id`, `tag.tags`.
- **Zero spans looks like a 404.** A project that never received a span does not exist, so the read-back reports `Failed to resolve project` or `404`. To prove suppression, make one traced call and one suppressed call in the same run and assert the span count is `1`. Helpers per language are in your language file.
- **Newest first, capped.** `px span list` returns the newest 100 spans by default. Raise `--limit` for long traces or filter with `--trace-id`.
- **`llm.model_name` is the resolved snapshot** (`gpt-4o-mini-2024-07-18`), not the alias you requested. Do not `--attribute`-filter on the alias.
- **Package pins differ.** Check the package's own dependency versions before copying a template: for example `resourceFromAttributes` exists only in `@opentelemetry/resources` 2.x, while most JS instrumentation packages pin 1.x and use `new Resource({...})`.

## Related

- `phoenix-cli` skill: full `px` command reference, trace and session JSON shapes, GraphQL.
- `python-code-reviewer` / `java-code-reviewer`: run after the fix is proven, before opening the PR.

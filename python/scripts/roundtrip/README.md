# Live round trip

A live round trip proves one claim about an instrumentor end to end: call the real provider API
through the instrumentor, export the spans to Phoenix, read back what Phoenix stored, and check
the span tree and attributes. Unit tests replay cassettes into an in-memory exporter; this is the
check that the live API, the instrumentor, OTLP export and Phoenix agree.

There are no fixed scenarios. Whoever verifies an issue or a PR, a factory session or a
maintainer, writes a one-off script for the claim in front of them. This directory holds the two
pieces every such script needs:

| File | |
| --- | --- |
| `serve_phoenix.sh [dir]` | starts a pinned Phoenix in the background on `$PHOENIX_HOST` (default `http://localhost:6006`), or reuses one already up |
| `phoenix_spans.py` | reads a project's spans over the Phoenix REST API; `wait_for_spans`, `roots`, `children` and `render` for a script to import, and a CLI that prints the tree |

## Writing one

1. **State the claim as span facts.** From the issue or the PR, before running anything: which
   spans appear, under which parent, with which kind and status, and which attribute keys hold
   which values. Take expected values from the issue, the PR, `spec/` and the semantic
   conventions, never from output you have just seen.
2. **Write the script outside the repo**, in `/tmp` or a scratch directory, as a PEP 723 script
   so `uv run` builds its environment. Install the code under test editable, with absolute paths
   to the checkout being verified:

   ```python
   # /// script
   # requires-python = ">=3.10"
   # dependencies = ["anthropic", "opentelemetry-sdk", "opentelemetry-exporter-otlp-proto-http",
   #   "openinference-instrumentation", "openinference-instrumentation-anthropic",
   #   "openinference-semantic-conventions"]
   # [tool.uv.sources]
   # openinference-instrumentation = { path = "<checkout>/python/openinference-instrumentation", editable = true }
   # openinference-instrumentation-anthropic = { path = "<checkout>/python/instrumentation/openinference-instrumentation-anthropic", editable = true }
   # openinference-semantic-conventions = { path = "<checkout>/python/openinference-semantic-conventions", editable = true }
   # ///
   ```

3. **Export synchronously to a fresh project.** A `TracerProvider` whose resource sets
   `openinference.project.name` to a name no earlier run used, a `SimpleSpanProcessor` around
   `OTLPSpanExporter(f"{phoenix}/v1/traces")`, and `force_flush()` before reading back. Wrap the
   calls in a root span you start yourself and record its trace and span IDs, so the check can
   say "a direct child of my root" rather than "somewhere in the project".
4. **Call the real API the way the claim says**: the same client method, streaming mode, tools,
   context attributes or `TraceConfig`. Use the smallest model that exercises the path, a small
   `max_tokens`, and no more than about ten calls.
5. **Read back and check.** `wait_for_spans(phoenix, project, expected=<count>)`, then assert on
   what came back: the span count in your trace, each span's parent, kind, status and
   attributes, and any attribute that must be absent. Build attribute keys from
   `SpanAttributes`, `MessageAttributes` and the other semconv classes. Print one row per check
   with the expected and observed values, and `render(spans, attributes=True)` for the trace.
   Exit 0 when every check passed, 1 when one failed, and 2 when nothing could be verified (no
   key, Phoenix down, an export failed, no spans came back).
6. **Run the negative control.** Run the same script against the other side of the claim: `main`
   for a reported bug, the PR's base for a PR. A bug script should fail on `main`; a fix script
   should fail on the base and pass on the PR. A script that passes on both sides doesn't test
   the claim; fix the script, not the expectation.
7. **Report it**: the claim, the script, and both runs' check rows. Never commit the script, and
   never print the key.

## Where it runs

- **Factory triage and review.** The issue triager (through the bug reproducer) and the PR
  reviewer write and run a round trip when the claim is about emitted spans and the sandbox has a
  provider credential. For Anthropic that is a short-lived token in `ANTHROPIC_AUTH_TOKEN`,
  which the SDK reads by itself; see Provider keys in `factory/README.md`. `factory/bootstrap.sh phoenix` starts Phoenix there.
- **Locally.** The `phoenix-verify` skill, with your own key.

## Adding a provider

The sandbox needs a credential for the provider's API, scoped to its host; see Provider keys in
`factory/README.md`. Add the host to `allowed_hosts` in
`factory/environments/openinference-dev.yaml` and the variable the SDK reads to the key report in
`factory/bootstrap.sh`.

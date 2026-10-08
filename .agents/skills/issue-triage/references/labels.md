# Label taxonomy

How the labels on `Arize-ai/openinference` group into the families the triage
stages use, and how every package in the repo maps to its label. GitHub's
descriptions remain the source of truth for meaning; this file is the map.

Label names below were current when this file was written. Verify against
`gh label list` at the start of a run and prefer GitHub when they disagree.

## Language (stage 3)

| Label | Cues in the issue |
| --- | --- |
| `language: python` | `openinference-instrumentation-<x>` package names, `pip`/`uv`/`tox`, `.py` tracebacks, `from openinference.instrumentation…`, `TraceConfig(hide_inputs=…)`, `using_attributes` |
| `language: js` | `@arizeai/openinference-*` package names, `npm`/`pnpm`/`yarn`, `.ts`/`.js` stack frames, `node_modules`, `OITracer` with `traceConfig`, `setSession(context, …)`, Vercel AI SDK, Mastra, TanStack |
| `language: java` | `com.arize.instrumentation`, Gradle or Maven coordinates, `.java` stack frames, LangChain4j, Spring AI, ADK Java, `TraceConfig.builder()` |

`python`, `javascript` and `java` (no prefix) are dependabot's PR labels. Never
apply them to issues.

## Instrumentation packages (stage 3)

The label is `instrumentation: <name>`. For packages added after this file was
written, and for `new instrumentation` requests where no label exists yet, the
naming rule for the report is: **the package directory suffix**, hyphens kept
(`instrumentation: strands-agents`), lower case. Older labels predate the rule
and keep their spelling (`instrumentation: openai agents sdk`,
`instrumentation: google genai`).

| Label | Python (`python/instrumentation/openinference-instrumentation-…`) | JS (`js/packages/openinference-…`) | Java (`java/instrumentation/openinference-instrumentation-…`) |
| --- | --- | --- | --- |
| `instrumentation: ag2` | `ag2` | | |
| `instrumentation: agent-framework` | `agent-framework` | | |
| `instrumentation: agentspec` | `agentspec` | | |
| `instrumentation: agno` | `agno` | | |
| `instrumentation: annotation` | | | `annotation` |
| `instrumentation: anthropic` | `anthropic` | `instrumentation-anthropic` | |
| `instrumentation: autogen` | `autogen` | | |
| `instrumentation: autogen-agentchat` | `autogen-agentchat` | | |
| `instrumentation: bedrock` | `bedrock` | `instrumentation-bedrock` | |
| `instrumentation: bedrock-agent-runtime` | | `instrumentation-bedrock-agent-runtime` | |
| `instrumentation: beeai` | `beeai` | `instrumentation-beeai` | |
| `instrumentation: claude-agent-sdk` | `claude-agent-sdk` | `instrumentation-claude-agent-sdk` | |
| `instrumentation: cohere` | `cohere` | | |
| `instrumentation: crewai` | `crewai` | | |
| `instrumentation: dspy` | `dspy` | | |
| `instrumentation: google-adk` | `google-adk` | | `adk-java` |
| `instrumentation: google genai` | `google-genai` | `instrumentation-google-genai` | |
| `instrumentation: groq` | `groq` | | |
| `instrumentation: guardrails` | `guardrails` | | |
| `instrumentation: haystack` | `haystack` | | |
| `instrumentation: instructor` | `instructor` | | |
| `instrumentation: langchain` | `langchain` | `instrumentation-langchain`, `instrumentation-langchain-v0` | |
| `instrumentation: langchain4j` | | | `langchain4j` |
| `instrumentation: litellm` | `litellm` | | |
| `instrumentation: llama-index` | `llama-index` | | |
| `instrumentation: mastra` | | `mastra` | |
| `instrumentation: mcp` | `mcp` | `instrumentation-mcp` | |
| `instrumentation: mistralai` | `mistralai` | | |
| `instrumentation: ollama` | `ollama` | | |
| `instrumentation: openai` | `openai` | `instrumentation-openai` | |
| `instrumentation: openai agents sdk` | `openai-agents` | `instrumentation-openai-agents` | |
| `instrumentation: openlit` | `openlit` | | |
| `instrumentation: openllmetry` | `openllmetry` | | |
| `instrumentation: pipecat` | `pipecat` | | |
| `instrumentation: portkey` | `portkey` | | |
| `instrumentation: promptflow` | `promptflow` | | |
| `instrumentation: pydantic-ai` | `pydantic-ai` | | |
| `instrumentation: smolagents` | `smolagents` | | |
| `instrumentation: spring-ai` | | | `springAI` |
| `instrumentation: strands-agents` | `strands-agents` | | |
| `instrumentation: tanstack-ai` | | `tanstack-ai` | |
| `instrumentation: together` | `together` | | |
| `instrumentation: vercel` | | `vercel` | |
| `instrumentation: vertexai` | `vertexai` | | |

LangGraph issues are `instrumentation: langchain` — LangGraph runs through the
LangChain callback handler (see the Python package's `_tracer.py` and its
`tests/test_langgraph_tool_calls.py`; treat JS the same unless the issue says
otherwise).

Two labels sit alongside the family:

- `instrumentation` — any issue about an instrumentor package, existing or
  proposed. Apply it together with the specific label.
- `new instrumentation` — the library has no instrumentor in the named
  language. Do not also apply an `instrumentation: <name>` label that does not
  exist; recommend it.
- `integration: langflow` — how Langflow consumes OpenInference. Not an
  instrumentor in this repo.

## Components (stages 3 and 4)

| Label | Covers |
| --- | --- |
| `c/core` | The shared libraries: `python/openinference-instrumentation`, `js/packages/openinference-core`, `java/openinference-instrumentation`. `OITracer`, `TraceConfig` implementation, context helpers, `BlobUploader`, `safe_json_dumps` |
| `c/semcov` | `spec/*.md` and the three `openinference-semantic-conventions` packages. Attribute names, span kinds, MIME types |
| `c/genai` | OpenTelemetry GenAI interop: `js/packages/openinference-genai`, the Python `gen_ai.*` dual-write conversion, the conformance harness. Usually also `c/semcov` |
| `c/decorators` | Manual instrumentation helpers: Python `OITracer.chain/tool/agent` decorators and `start_as_current_span`, JS core trace helpers, Java annotations |
| `c/annotations` | `spec/annotations.md` — evaluation and human-feedback attributes |
| `c/ci` | `.github/workflows`, release-please, `python/tox.ini`, pnpm workspace config, Gradle build |
| `c/trace-config` | Stage 4 — masking / PII requirement |
| `c/context-attributes` | Stage 4 — context attribute propagation requirement |
| `c/suppress-tracing` | Stage 4 — tracing suppression requirement |
| `examples` | The runnable examples under each package and `js/examples` |
| `documentation` | Also a type (stage 1) |

For stage 5's raise rule, `c/core` and `c/semcov` count as "core or spec".
`c/genai`, `c/decorators`, `c/annotations`, `c/ci` and the three requirement
labels are dimensions, not subsystems — they do not raise complexity on their
own.

## Types (stage 1)

`bug`, `enhancement`, `documentation`, `cleanup`, `question`. Exactly one per
issue.

## Complexity (stage 5)

`size:S`, `size:M`, `size:L` on issues. `size:XS`, `size:XL`, `size:XXL` are
the PR bot's and never go on issues.

## Status and process — read, never write

These describe state a maintainer or a bot owns. The stages read them (a
`blocked` or `stale` issue never gates) and the report may recommend them.

`needs information` is the exception: stage 2 adds and removes it.

| Label | Read as |
| --- | --- |
| `blocked` | Not startable. Stage 8 treats a `Blocked by #NNNN` line in prose the same way |
| `stale`, `duplicate`, `wontfix`, `invalid`, `cannot reproduce` | Not a candidate for any gate |
| `needs attention` | A person must respond. Independent of triage |
| `priority: high`, `priority: low` | Maintainer prioritisation |
| `customer request` | Raised for an Arize or Phoenix customer |
| `help wanted` | Maintainers welcome outside help; not itself a gate |
| `triage` | Needs triage **and** drives the Slack digest — see SKILL.md → Finishing an issue |
| `backlog`, `roadmap`, `feature branch`, `security` | Maintainer planning; `security` may be recommended from stage 4a |

## Automation — never touch

| Label | Effect |
| --- | --- |
| `good-agent-issue` | Applying it runs `.github/workflows/claude-implement-issue.yml`: assigns the issue, adds `agent-in-progress`, opens a PR. Only the opt-in `good-agent-issue` gate policy may apply it |
| `agent-fix` | Starts the oss-support-agent fix pipeline |
| `agent-in-progress` | An agent is working the issue. Never edit its body or labels |
| `lgtm`, `autorelease: pending`, `autorelease: tagged`, `dependencies` | Release and review bots |

## Gate labels (stage 8)

| Label | Policy |
| --- | --- |
| `good first issue` | `references/good-first-issue.md` — on by default |
| `good student issue` | `references/good-student-issue.md` — on while Stanford CS146S runs; temporary |
| `good-agent-issue` | `references/good-agent-issue.md` — opt in only |

`references/gate-classification.md` decides all three in one pass.

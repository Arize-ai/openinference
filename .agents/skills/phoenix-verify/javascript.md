# Phoenix Verify: JavaScript / TypeScript

Read this file only when the instrumentor under test is a JS package. `$SCRATCH` is your
session scratchpad; `<pkg>` is the short name (`openai`, `langchain`).

## Where examples live

`js/packages/openinference-instrumentation-<pkg>/examples/<scenario>.ts`. They import `../src`
directly, so tsx compiles the working tree; the build is only for workspace dependencies whose
`main` points at `dist/`. Several packages share one `examples/instrumentation.ts` bootstrap
that sets the project name for every example.

## Setup, proof, run

```bash
cd js && pnpm install --frozen-lockfile -r
pnpm --filter "@arizeai/openinference-instrumentation-<pkg>..." run build   # package + workspace deps
cd packages/openinference-instrumentation-<pkg> && pnpm exec tsx examples/<scenario>.ts
# proof: the example's import of "../src"; some READMEs say `npx tsx`, but workspace tooling runs through pnpm
```

## Example template

Check the package's `@opentelemetry/resources` major first. Most instrumentation packages pin
1.x and use `new Resource({...})` (as in
`openinference-instrumentation-openai/examples/manual-instrumentation.ts`); `openinference-core`
and `openinference-genai` are on 2.x and use `resourceFromAttributes`.

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

### Scratchpad copy for before/after runs

Copy the example (and its shared bootstrap, if any) to `$SCRATCH` and take the project name from
`process.argv[2]` with the committed name as default, so both runs use identical code. Fix the
`../src` import to an absolute path into the worktree you are testing.

### Context attributes and suppression

| Context attributes | Suppress tracing |
| --- | --- |
| `setSession`, `setUser`, `setMetadata`, `setTags` from `@arizeai/openinference-core`, applied with `context.with(...)` | `suppressTracing(context.active())` from `@opentelemetry/core`, also applied with `context.with(...)` |

Make one traced call and one suppressed call in the same run, then assert the span count is 1
and the traced span carries `session.id`, `user.id`, `metadata.<key>`, and `tag.tags`.

## Getting a "before" build

| Situation | "Before" |
| --- | --- |
| Fix not yet applied | Run, apply the change, run again |
| Fix already in the working tree or branch | `git worktree add "$SCRATCH/wt-main" origin/main`, then `pnpm install` and the filtered build inside it; point the scratchpad copy's import at that worktree's `src` |
| Parity with the last npm release | In a scratch project, `pnpm add @arizeai/openinference-instrumentation-<pkg>@latest` and import from the package name instead of `../src` |

## Wiring a new example into the repo

- Add it to the package's `examples/README.md` or a `## Examples` section in the package README.
- New dependencies go in the package's `devDependencies`.
- Examples are excluded from oxlint and oxfmt, but `pnpm run type:check` (run by CI) covers
  `examples/**/*.ts` in several packages (`langchain`, `langchain-v0`, `beeai`, `vercel`,
  `openinference-genai`); run it in the package before opening the PR.

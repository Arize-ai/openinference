---
name: js-docs-sync
description: >
  Keep hand-written docs/ documentation in JS packages accurate and up to date
  with their source code. Use this skill whenever: (1) source files in a JS package
  that has a docs/ folder are modified — especially exports, function signatures,
  types, or public API changes, (2) the user asks to "update docs", "sync docs",
  "check if docs are accurate", "review the documentation", or similar, (3) new
  exports or features are added to a JS package and the docs need to reflect them.
  Also trigger when the user mentions documentation drift, stale examples, or
  missing API coverage in any JS package under js/packages/.
invocable: true
---

# JS Package Docs Sync

Verify and update hand-written `docs/` documentation in a JS package so it stays
accurate against the actual source code. These docs ship in the npm package (via
the `files` field in package.json) for coding agents to read from node_modules.

## When This Matters

The docs are the contract between the package and coding agents that consume it.
Stale signatures, missing exports, or invalid examples mean agents write broken
code. Source code is the single source of truth — docs must reflect it exactly.

## Workflow

### Step 1: Identify the Package

Determine which JS package to sync. If not clear from context, check which
packages have a `docs/` directory:

```bash
find js/packages -maxdepth 2 -name docs -type d
```

If only one package has docs, use it. Otherwise ask the user.

### Step 2: Understand the Doc Structure

Read `docs/README.md` in the target package. It contains:
- A **Documentation Guide** table mapping each doc file to its topic area
- An **All Exports at a Glance** section listing every public export
- A **Source Code Map** showing which source files contain what

This is your roadmap — it tells you what each doc file covers and where to look
in the source to verify it.

### Step 3: Identify What Changed

If this is triggered by source edits, focus on the changed files:

```bash
git diff --name-only HEAD~1 -- js/packages/<package>/src/
```

Map changed source files to the doc files that cover them (using the Source Code
Map in README.md). You only need to re-verify the affected docs.

If this is a full review (user asks to "check all docs"), verify everything.

### Step 4: Verify Each Doc File

For each doc file that needs checking, compare it against the source code.
Check these things in order:

#### 4a. Export Completeness

Read the barrel files (`src/index.ts` and any sub-module `index.ts` files) to get
the full list of public exports. Compare against what's listed in `docs/README.md`
under "All Exports at a Glance". Flag any export that is:
- In source but missing from docs (new export, needs documenting)
- In docs but not actually exported (removed or renamed, needs removing from docs)

#### 4b. Function Signatures

For each documented function signature, read the actual source and compare:
- Parameter names and types (including generics and template literals)
- Return types
- Optional vs required parameters
- Default values mentioned in docs vs actual defaults in source

Pay special attention to:
- Union types that may have changed (e.g., `| string` vs `| \`${SomeEnum}\``)
- Generic type parameters (e.g., `<Fn extends AnyFn>`)
- Overloaded signatures

#### 4c. Type/Interface Definitions

For each documented type or interface:
- Verify all fields exist with correct types and optionality
- Check for new fields added to the source that aren't in docs
- Check for removed fields that are still in docs

#### 4d. Enum Values

If the docs list enum values (like span kinds), verify the list is complete
against the actual enum definition. Enums grow over time — new values get added
but docs often lag behind.

#### 4e. Code Examples

For each code example in the docs:
- Verify imports reference real exported names
- Verify function calls use correct parameter shapes
- Check that TypeScript syntax is valid (watch for `[0.1, 0.2, ...]` — the
  spread operator needs an iterable; use `/* ... */` for truncated arrays)
- Verify any referenced types or constants actually exist

### Step 5: Update the Source Code Map

If source files were added, removed, or renamed, update the Source Code Map tree
in `docs/README.md` to match the current `src/` directory structure.

### Step 6: Apply Fixes

For each discrepancy found, edit the affected doc file. When updating:
- Match the existing documentation style (line width, heading levels, code fence
  language tags)
- Preserve existing prose and examples where still accurate — only change what's
  wrong
- If a new export needs a full doc section, follow the pattern of adjacent
  sections in the same file (signature block, then prose, then example)
- Update the "All Exports at a Glance" in README.md if exports changed

### Step 7: Verify

After making changes, do a final pass:
- Read each modified doc file to confirm it reads coherently
- Verify the package still builds: `cd js && pnpm run -r build`
- Spot-check that the README.md export list matches what the barrel files export

## Common Pitfalls

**Template literal types vs plain strings**: TypeScript template literal types
like `` `${SomeEnum}` `` restrict to specific string values. Don't document these
as `| string` — that's misleadingly broad. Show the actual type.

**Internal vs public exports**: A function may be `export`ed from its own file
but not re-exported through the barrel `index.ts` chain. Only document functions
that are reachable from the package's main entry point.

**Type-only exports**: Exports using `export type { ... }` are erased at runtime.
They should still be documented (they're part of the TypeScript API) but note
that they're type-only if relevant.

**Generated docs conflict**: The repo has TypeDoc-generated API docs at `js/docs/`
(gitignored). The hand-written docs in `js/packages/*/docs/` are separate — don't
confuse the two. Hand-written docs focus on usage patterns and examples, not
exhaustive API surface.

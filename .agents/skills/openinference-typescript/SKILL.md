---
name: openinference-typescript
description: TypeScript conventions for any TypeScript code in js/ (packages, examples, tests). Use whenever writing, reviewing, or modifying TypeScript in this repo.
---

# OpenInference TypeScript Conventions

Follow existing patterns in the package you're editing (see `js/AGENTS.md`), plus:

## Naming

- No single-letter variables, including loop counters (`index`, `span`, `chunk`).
- Booleans use verb prefixes: `isStreaming`, `hasError`, `shouldMask`.
- Functions start with a verb: `getAttributes`, `parseResponse`, `buildSpanName`.
- Extract complex conditions into named booleans.

## Functions

- 2+ parameters → a single destructured object, documented with JSDoc `@param params.foo`.
- Prefer composition over inheritance, and pure transforms over mutation.
- Don't read ambient state (`Date.now()`, config singletons, module constants) deep inside utilities; expose it as an optional param defaulting to a `DEFAULT_*` constant.

## Types

- No `any`; use `unknown` and narrow with type guards. If `any` is unavoidable, comment why.
- Sparse lookup maps must allow `undefined`: `Partial<Record<K, V>>` (most packages don't enable `noUncheckedIndexedAccess`).

## Reuse

Check `@arizeai/openinference-core` and sibling packages for existing utilities before writing helpers or adding dependencies.

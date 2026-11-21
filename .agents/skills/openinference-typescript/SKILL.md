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

## Documentation

Every declaration gets a TSDoc comment (`/** ... */`), whether or not it is exported:

- Functions, classes and class members, types, interfaces and their fields, and module-level constants.
- Say what it does and why it exists, not a restatement of the name. Note precedence rules, fallbacks, and what is skipped or dropped.
- Document every parameter with `@param`; for a destructured object use `@param params.foo`. Add `@returns` when the return value needs explaining, especially when it can be `undefined`.
- Public exports get an `@example` when usage is not obvious from the signature, and `@experimental` or `@deprecated` where they apply.
- Name the attributes involved in backticks (`llm.model_name`, `gen_ai.request.model`).
- A one-line `/** ... */` is fine for constants and simple fields.

Check coverage before finishing: no `function`, `const`, `type`, `interface` or `class` at module level without a doc comment directly above it.

## Type Guards

Narrow with the named guards from `@arizeai/openinference-core` (`src/utils/typeUtils.ts`) so conditions read as English: `isString`, `isNumber`, `isBoolean`, `isArray`, `isRecord`, `isDefined`, `isUndefined`, `isNullish`, `isPresent`. Don't write inline `typeof value === "string"`, `Array.isArray(value)`, `value === undefined` or `value == null` checks, and add new general-purpose guards to `typeUtils.ts`, not to feature modules.

## Reuse

Check `@arizeai/openinference-core` and sibling packages for existing utilities before writing helpers or adding dependencies.

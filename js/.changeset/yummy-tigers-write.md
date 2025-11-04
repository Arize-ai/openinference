---
"@arizeai/openinference-core": major
---

# feat: Add tracing capabilities with decorators and function wrappers

- **Function Wrapping**: `withSpan()`, `traceAgent()`, `traceTool()` ....
- **Decorators**: `@observe()` for class methods

**Function Wrapping:**

```typescript
const tracedLLM = traceAgent(callOpenAI, {
  attributes: { "llm.model": "gpt-4" },
});
```

**Decorators:**

```typescript
class Agent {
  @observe({ kind: "AGENT" })
  async makeDecision(context) {
    /* ... */
  }
}
```

**Custom Processing:**

```typescript
const traced = traceChain(fn, {
  attributes: { "service.name": "my-service" },
  processInput: (...args) => ({ "input.count": args.length }),
});
```

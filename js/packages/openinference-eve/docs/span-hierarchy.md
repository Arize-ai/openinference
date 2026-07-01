# Eve Span Hierarchy

## How Eve Creates Spans

Eve creates an `ai.eve.turn` root span for each conversational turn, then uses
the Vercel AI SDK internally which creates the standard `ai.streamText`,
`ai.streamText.doStream`, and `ai.toolCall` child spans.

A typical two-step agentic run (one tool call, then a final answer) produces:

```
ai.eve.turn                         AGENT  ← session.id extracted here
└── ai.streamText (step 0)          AGENT  ← first model call
    ├── ai.streamText.doStream       LLM    ← token counts, messages, model name
    └── ai.toolCall get_weather      TOOL   ← tool name, args, result
└── ai.streamText (step 1)          AGENT  ← second model call (final answer)
    └── ai.streamText.doStream       LLM    ← token counts
```

## Span Kind Assignment

| Span name | `openinference.span.kind` | How it's set |
|---|---|---|
| `ai.eve.turn` | `AGENT` | Set by `addEveAttributesToSpan` on the `ai.eve.turn` span name |
| `ai.streamText` | `AGENT` | Inherited from Vercel AI SDK processor (`ai.operationId = "ai.streamText"`) |
| `ai.streamText.doStream` | `LLM` | Inherited from Vercel AI SDK processor (`gen_ai.*` attributes) |
| `ai.toolCall` | `TOOL` | Inherited from Vercel AI SDK processor (`ai.operationId = "ai.toolCall"`) |

## `ai.eve.turn` Span

The root turn span carries Eve-specific context that applies to the whole turn:

```typescript
// Attributes Eve sets on ai.eve.turn:
{
  "operation.name": "ai.eve.turn",
  "eve.session.id": "sess_abc123",   // ← mapped to session.id
  "eve.version": "1.0.0",            // ← mapped to metadata.eve.version
  "eve.environment": "production",   // ← mapped to metadata.eve.environment
  "eve.turn.id": "turn_0",           // ← mapped to metadata.eve.turn.id
  "eve.turn.sequence": 0,            // ← mapped to metadata.eve.turn.sequence
  "eve.channel.kind": "channel:terminal", // ← mapped to metadata.eve.channel.kind
}
```

After processing by `OpenInferenceSimpleSpanProcessor`:

```typescript
{
  // Original Eve attributes (preserved)
  "eve.session.id": "sess_abc123",
  // ... other eve.* attributes ...

  // OpenInference attributes (added by this processor)
  "openinference.span.kind": "AGENT",
  "session.id": "sess_abc123",
  "metadata.eve.version": "1.0.0",
  "metadata.eve.environment": "production",
  "metadata.eve.turn.id": "turn_0",
  "metadata.eve.turn.sequence": 0,
  "metadata.eve.channel.kind": "channel:terminal",
}
```

## Child Spans

Eve injects `eve.*` attributes onto every span (not just the root), so all
child spans also receive `session.id` and `metadata.eve.*` attributes.

```typescript
// ai.streamText — after processing:
{
  "openinference.span.kind": "AGENT",     // set by Vercel processor
  "session.id": "sess_abc123",             // extracted from eve.session.id
  "metadata.eve.step.index": 0,            // step index as metadata
  "metadata.eve.turn.sequence": 0,         // other eve.* → metadata.eve.*
  "metadata.eve.environment": "development",
  // ...plus all Vercel AI SDK attributes...
}
```

## `operation.name` and `functionId`

Eve appends a user-provided `functionId` to the span name via `operation.name`:

```
operation.name = "ai.eve.turn my-agent"
                  ^^^^^^^^^^ ^^^^^^^^
                  base name  functionId suffix
```

`addEveAttributesToSpan` splits on the first space and looks up only the base
name in `EveFunctionNameToSpanKindMap`, so the span kind is assigned correctly
regardless of the `functionId` suffix.

## `isOpenInferenceSpan` Filter

Passing `spanFilter: isOpenInferenceSpan` skips spans that carry no
OpenInference-mapped attributes. In an Eve app, the spans that pass the filter
are exactly the four kinds above — all other framework/infrastructure spans
are dropped, keeping your trace focused on the AI pipeline.

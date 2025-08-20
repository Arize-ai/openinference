# Mastra I/O Message Support Enhancement Plan

## Problem Statement

The Mastra tracing library has two critical issues that impact Phoenix observability:

1. **Missing I/O Attributes**: Mastra agent spans lack proper `INPUT_VALUE` and `OUTPUT_VALUE` attributes
2. **Filtered Root Spans**: The root span (HTTP request) is filtered out, preventing Phoenix session I/O extraction

While Vercel AI SDK spans correctly set these attributes, Mastra agent spans only have structured attributes like `agent.*.argument.0` and `agent.*.result` without the corresponding OpenInference semantic conventions.

**Critical Issue**: Phoenix determines session "first input" and "last output" by analyzing root spans (`parent_id IS NULL`), but the current span filter removes the root HTTP span because it lacks `openinference.span.kind`.

### Current State Analysis

**Working (Vercel spans):**

```javascript
// ai.streamText span
'input.value': '{"messages":[{"role":"system","content":"..."},{"role":"user","content":[{"type":"text","text":"hi"}]}]}',
'output.value': "Hello! I'm here to help with any weather-related questions...",
'input.mime_type': 'application/json',
'output.mime_type': 'text/plain'
```

**Missing (Mastra spans):**

```javascript
// agent.getMostRecentUserMessage span
'agent.getMostRecentUserMessage.argument.0': '[{"role":"user","content":"hi",...},{"role":"assistant","content":"Hello!...",...}]',
'agent.getMostRecentUserMessage.result': '{"role":"user","content":"hi",...}',
// ❌ No input.value or output.value
```

**Filtered Out (Root span):**

```javascript
// POST /copilotkit span - TRUE ROOT SPAN
parentSpanContext: undefined,  // ← No parent = root span
'openinference.span.kind': undefined  // ← Filtered out by isOpenInferenceSpan
// ❌ Missing from exported traces, breaks Phoenix session I/O
```

### Root Cause

1. **Mastra I/O**: The Vercel `getOpenInferenceAttributes` function only processes Vercel AI SDK semantic conventions (`ai.*` attributes). Mastra spans use different attribute patterns (`agent.*`) that aren't recognized by the existing processing logic.

2. **Root Span Filtering**: The `isOpenInferenceSpan` filter requires `openinference.span.kind` to be set, but the root HTTP span has `openinference.span.kind: undefined`, causing it to be filtered out. This breaks Phoenix's session I/O extraction which depends on root spans.

## Phoenix Session I/O Requirements

Phoenix extracts session "first input" and "last output" using this logic:

```sql
-- First Input (conceptual)
SELECT span.attributes['input.value']
FROM spans
WHERE spans.parent_id IS NULL  -- Root spans only
  AND traces.project_session_rowid = :session_id
ORDER BY traces.start_time ASC
LIMIT 1

-- Last Output (conceptual)
SELECT span.attributes['output.value']
FROM spans
WHERE spans.parent_id IS NULL  -- Root spans only
  AND traces.project_session_rowid = :session_id
ORDER BY traces.start_time DESC
LIMIT 1
```

**Requirements for Phoenix session I/O:**

1. Root span must have `parent_id IS NULL`
2. Root span must pass `isOpenInferenceSpan` filter
3. Root span must have `input.value` and `output.value` attributes

## Solution Options Considered

### Option 1: Enhance Mastra Span Processing ✅ **CHOSEN**

Modify `addOpenInferenceAttributesToMastraSpan` to extract I/O from Mastra attributes.

**Pros:**

- Clean separation of concerns - Mastra logic stays in Mastra package
- No performance impact on Vercel processing
- Easy to customize for Mastra-specific patterns
- Maintains existing architecture

**Cons:**

- Need to duplicate some parsing logic from Vercel
- Mastra patterns are less standardized

### Option 2: Extend Vercel Processing ❌

Add Mastra patterns to `AISemanticConventions` and `getOpenInferenceAttributes`.

**Pros:** Reuses existing logic, centralized processing
**Cons:** Violates separation of concerns, performance impact, harder to maintain

### Option 3: Custom Span Filter ❌

Modify `isOpenInferenceSpan` to include Mastra spans without I/O.

**Pros:** Simple implementation
**Cons:** Doesn't solve core problem, inconsistent data

### Option 4: Post-Processing ❌

Add I/O processing after both Vercel and Mastra processing.

**Pros:** Non-invasive, handles both patterns
**Cons:** Performance overhead, complex execution flow

### Option 5: Enhanced Input Extraction ❌

Focus only on key spans with rich I/O data.

**Pros:** Targeted, high impact
**Cons:** Incomplete solution, manual span identification

## Implementation Plan

### Phase 1: Core Infrastructure ✅ **COMPLETED**

Enhance `packages/openinference-mastra/src/attributes.ts` with I/O extraction capabilities:

```typescript
import {
  MimeType,
  OpenInferenceSpanKind,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";
import { safelyJSONParse } from "@arizeai/openinference-core";

// Add these new functions to attributes.ts:

/**
 * Extracts INPUT_VALUE from Mastra agent function arguments
 */
const getMastraInputValue = (span: ReadableSpan): string | undefined => {
  // Look for agent.*.argument.0 patterns
  // Parse JSON and extract meaningful input text
  // Handle message arrays, simple strings, etc.
};

/**
 * Extracts OUTPUT_VALUE from Mastra agent function results
 */
const getMastraOutputValue = (span: ReadableSpan): string | undefined => {
  // Look for agent.*.result patterns
  // Parse JSON and extract meaningful output text
  // Handle various result formats
};

/**
 * Determines MIME type for Mastra I/O values
 */
const getMimeTypeFromMastraValue = (value: string): MimeType => {
  // Reuse logic from Vercel utils or implement similar
  // JSON detection vs plain text
};
```

### Phase 2: Integration ✅ **COMPLETED**

Modify `addOpenInferenceAttributesToMastraSpan` to call these functions:

```typescript
export const addOpenInferenceAttributesToMastraSpan = (span: ReadableSpan) => {
  // Existing logic...

  // Add I/O value extraction
  const inputValue = getMastraInputValue(span);
  if (inputValue) {
    span.attributes[SemanticConventions.INPUT_VALUE] = inputValue;
    span.attributes[SemanticConventions.INPUT_MIME_TYPE] =
      getMimeTypeFromMastraValue(inputValue);
  }

  const outputValue = getMastraOutputValue(span);
  if (outputValue) {
    span.attributes[SemanticConventions.OUTPUT_VALUE] = outputValue;
    span.attributes[SemanticConventions.OUTPUT_MIME_TYPE] =
      getMimeTypeFromMastraValue(outputValue);
  }
};
```

### Phase 3: Root Span Filter Fix 🚧 **PENDING**

Enable the root span to pass the `isOpenInferenceSpan` filter by setting the required span kind.

#### Problem

The root HTTP span has `openinference.span.kind: undefined` and gets filtered out, preventing Phoenix from accessing it for session I/O extraction.

#### Solution

Add root span processing to `OpenInferenceTraceExporter.export()` method:

```typescript
const enhanceRootSpan = (spans: ReadableSpan[]): void => {
  // Find root span (parentSpanContext === undefined)
  const rootSpan = spans.find((span) => span.parentSpanContext === undefined);
  if (!rootSpan) return;

  // Set AGENT span kind so it passes isOpenInferenceSpan filter
  rootSpan.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND] =
    OpenInferenceSpanKind.AGENT;
};
```

#### Integration Point

```typescript
// In OpenInferenceTraceExporter.export()
export(spans: ReadableSpan[], resultCallback: (result: ExportResult) => void) {
  let filteredSpans = spans.map((span) => {
    // ... existing processing
  });

  // NEW: Ensure root span passes filter
  enhanceRootSpan(filteredSpans);

  if (this.spanFilter) {
    filteredSpans = filteredSpans.filter(this.spanFilter);
  }
  super.export(filteredSpans, resultCallback);
}
```

**Result**: Root span will now be included in exported traces, enabling Phoenix session analysis.

### Phase 4: Session I/O Enhancement 🚧 **PENDING**

Populate the root span with session-level I/O data for Phoenix sessions view.

#### Design Decision: Direct Extraction from Raw Agent Attributes

Focus only on session-level I/O by directly extracting from raw `agent.*` attributes in child spans. This avoids the inefficient two-step process of setting individual span I/O attributes and then re-extracting them.

#### Implementation

```typescript
const addSessionIOToRootSpan = (spans: ReadableSpan[]): void => {
  const rootSpan = spans.find((span) => span.parentSpanContext === undefined);
  if (!rootSpan) return;

  // Extract session I/O directly from raw agent attributes in child spans
  const sessionInput = extractSessionInputFromAgentSpans(spans);
  const sessionOutput = extractSessionOutputFromAgentSpans(spans);

  if (sessionInput) {
    rootSpan.attributes[SemanticConventions.INPUT_VALUE] = sessionInput;
    rootSpan.attributes[SemanticConventions.INPUT_MIME_TYPE] = MimeType.TEXT;
  }

  if (sessionOutput) {
    rootSpan.attributes[SemanticConventions.OUTPUT_VALUE] = sessionOutput;
    rootSpan.attributes[SemanticConventions.OUTPUT_MIME_TYPE] = MimeType.TEXT;
  }
};

const extractSessionInputFromAgentSpans = (
  spans: ReadableSpan[],
): string | undefined => {
  // Look through child spans for the first meaningful input from agent.*.argument.0
  for (const span of spans) {
    if (span.parentSpanContext === undefined) continue; // skip root span

    const argumentKeys = Object.keys(span.attributes)
      .filter((key) => key.startsWith("agent.") && key.endsWith(".argument.0"))
      .sort();

    for (const key of argumentKeys) {
      const argumentValue = span.attributes[key] as string;
      if (!argumentValue) continue;

      const parsed = safelyJSONParse(argumentValue);
      if (parsed) {
        const content = extractMessageContent(parsed);
        if (content) return content;
      }

      if (argumentValue.trim()) return argumentValue;
    }
  }
  return undefined;
};

const extractSessionOutputFromAgentSpans = (
  spans: ReadableSpan[],
): string | undefined => {
  // Look through child spans for the last meaningful output from agent.*.result
  let lastOutput: string | undefined;

  for (const span of spans) {
    if (span.parentSpanContext === undefined) continue; // skip root span

    const resultKeys = Object.keys(span.attributes)
      .filter((key) => key.startsWith("agent.") && key.endsWith(".result"))
      .sort();

    for (const key of resultKeys) {
      const resultValue = span.attributes[key] as string;
      if (!resultValue || resultValue.includes("[Not Serializable]")) continue;

      const parsed = safelyJSONParse(resultValue);
      if (parsed) {
        const content = extractMessageContent(parsed);
        if (content) lastOutput = content;

        // Handle other structured result types
        if (typeof parsed === "object" && parsed !== null) {
          const meaningfulKeys = ["text", "content", "message", "value"];
          for (const propKey of meaningfulKeys) {
            if (typeof parsed[propKey] === "string" && parsed[propKey].trim()) {
              lastOutput = parsed[propKey];
              break;
            }
          }
        }
      } else if (
        resultValue.trim() &&
        !resultValue.startsWith("{") &&
        !resultValue.startsWith("[")
      ) {
        lastOutput = resultValue;
      }
    }
  }

  return lastOutput;
};
```

#### Integration Point

```typescript
// In OpenInferenceTraceExporter.export()
export(spans: ReadableSpan[], resultCallback: (result: ExportResult) => void) {
  let filteredSpans = spans.map((span) => {
    // 1. Process each child span individually (Phases 1-2)
    addOpenInferenceProjectResourceAttributeSpan(span);
    addOpenInferenceAttributesToSpan({...});
    addOpenInferenceAttributesToMastraSpan(span);
    return span;
  });

  // 2. Child spans now have INPUT_VALUE/OUTPUT_VALUE set
  enhanceRootSpan(filteredSpans);        // Phase 3: Filter fix
  addSessionIOToRootSpan(filteredSpans); // Phase 4: Session I/O

  if (this.spanFilter) {
    filteredSpans = filteredSpans.filter(this.spanFilter);
  }
  super.export(filteredSpans, resultCallback);
}
```

**Result**: Root span will have session-level I/O data, enabling Phoenix sessions view to display first input and last output.

## Technical Considerations

### 1. JSON Parsing Safety

- Use `safelyJSONParse` from `@arizeai/openinference-core`
- Handle malformed JSON gracefully
- Fallback to raw string if parsing fails

### 2. MIME Type Detection

- Reuse logic from Vercel utils or implement similar
- JSON vs plain text detection
- Consistent with existing Vercel behavior

### 3. Performance

- Only process spans with Mastra agent patterns
- Lazy evaluation of expensive operations
- Minimal impact on non-Mastra spans

### 4. Error Handling

- Graceful degradation when parsing fails
- Diagnostic logging for debugging
- Don't break span export on errors

## Expected Outcomes

After implementation, Mastra agent spans will have:

```javascript
// Before
'agent.getMostRecentUserMessage.argument.0': '[{"role":"user","content":"hi",...}]',
'agent.getMostRecentUserMessage.result': '{"role":"user","content":"hi",...}',
'openinference.span.kind': 'AGENT'

// After
'agent.getMostRecentUserMessage.argument.0': '[{"role":"user","content":"hi",...}]',
'agent.getMostRecentUserMessage.result': '{"role":"user","content":"hi",...}',
'openinference.span.kind': 'AGENT',
'input.value': 'Conversation with user and assistant messages', // ✅ Added
'input.mime_type': 'text/plain', // ✅ Added
'output.value': 'hi', // ✅ Added
'output.mime_type': 'text/plain' // ✅ Added
```

## Benefits

1. **Complete I/O Visibility**: All Mastra agent spans will have standardized I/O attributes
2. **Consistent Observability**: Same experience for Vercel and Mastra spans
3. **Better Filtering**: Spans will pass `isOpenInferenceSpan` checks
4. **Maintainable Architecture**: Changes isolated to Mastra package
5. **Performance Optimized**: No overhead on existing Vercel spans

## Next Steps

1. Implement core I/O extraction functions
2. Add comprehensive test coverage
3. Validate with real Mastra traces
4. Document new capabilities
5. Consider extending to other agent function patterns as needed

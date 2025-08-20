# Phoenix Tracing Core Concepts

This document explains key tracing concepts in Phoenix's architecture.

## Root Spans

### Definition

A **root span** is the topmost span in a trace hierarchy - the entry point that represents the beginning of a request or operation flow.

### Key Characteristics

- **No Parent**: Root spans have `parent_id = NULL` in the database
- **Trace Boundary**: Each root span defines a complete request from start to finish
- **Two Types in Phoenix**:
  1. **Explicit root spans**: `parent_id IS NULL`
  2. **Orphan spans**: `parent_id` references a non-existent span

### Examples in LLM Applications

- User query hitting a RAG application
- API request to a chatbot
- Batch processing job for document embeddings
- Evaluation run across multiple examples

### Database Model

```python
class Span:
    span_id: str                    # Unique identifier
    parent_id: Optional[str]        # NULL for root spans
    trace_id: str                   # Links all spans in a trace
    name: str                       # Operation name
    start_time: datetime
    end_time: datetime
```

## Session Input/Output Extraction

### Definition

Phoenix determines a session's **first input** and **last output** by analyzing the root spans of traces within that session.

### Data Flow

1. **Session Grouping**: Traces are grouped by `session_id`
2. **Root Span Analysis**: Only root spans (`parent_id IS NULL`) are considered
3. **Temporal Ordering**: Traces are ordered by `start_time`
4. **Attribute Extraction**: Input/output values come from OpenInference semantic conventions

### First Input Logic

```sql
-- Conceptual query
SELECT span.attributes['input.value']
FROM spans
JOIN traces ON spans.trace_rowid = traces.id
WHERE spans.parent_id IS NULL
  AND traces.project_session_rowid = :session_id
ORDER BY traces.start_time ASC, traces.id ASC
LIMIT 1
```

### Last Output Logic

```sql
-- Conceptual query
SELECT span.attributes['output.value']
FROM spans
JOIN traces ON spans.trace_rowid = traces.id
WHERE spans.parent_id IS NULL
  AND traces.project_session_rowid = :session_id
ORDER BY traces.start_time DESC, traces.id DESC
LIMIT 1
```

### Example Session Flow

```
Session: "user_chat_123"

10:00 AM - Trace 1 Root Span:
  input: "How do I install Phoenix?"
  output: "You can install with pip install arize-phoenix"

10:05 AM - Trace 2 Root Span:
  input: "What about Docker?"
  output: "Phoenix supports Docker deployment..."

10:10 AM - Trace 3 Root Span:
  input: "Thanks!"
  output: "You're welcome!"

Result:
- First Input: "How do I install Phoenix?"
- Last Output: "You're welcome!"
```

### Implementation Details

- **Location**: `src/phoenix/server/api/dataloaders/session_io.py`
- **DataLoader**: Uses `SessionIODataLoader` for efficient batch loading
- **Attributes**: Follows OpenInference semantic conventions:
  - `SpanAttributes.INPUT_VALUE` / `SpanAttributes.OUTPUT_VALUE`
  - `SpanAttributes.INPUT_MIME_TYPE` / `SpanAttributes.OUTPUT_MIME_TYPE`
- **Null Handling**: Returns `None` if no valid input/output found

### Key Constraints

- Only root spans are considered (not child spans)
- Uses trace-level timing, not individual span timing
- All traces must belong to the same session
- Preserves MIME type information for proper content handling

## Why This Matters

- **Root spans** define request boundaries and enable trace-level metrics
- **Session I/O** provides conversational context by capturing the full user journey from initial query to final response
- Both concepts are fundamental to Phoenix's observability and evaluation capabilities

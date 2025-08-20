# Mastra Tracing Implementation Guide

## Overview

Mastra implements comprehensive distributed tracing using **OpenTelemetry (OTEL)** as its core observability framework. The tracing system provides automatic instrumentation for all major components while supporting custom integrations with popular LLMOps and observability platforms.

## Architecture

### Core Components

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Telemetry     │───▶│    Decorators    │───▶│    Exporters    │
│   (Singleton)   │    │  (@withSpan,     │    │   (Storage,     │
│                 │    │  @InstrumentClass)│    │   Cloud, OTLP)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  OpenTelemetry  │    │  Proxy Wrapper   │    │  Observability  │
│   API/SDK       │    │   (traceClass,   │    │   Platforms     │
│                 │    │   traceMethod)   │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Key Classes

#### 1. Telemetry Class (`packages/core/src/telemetry/telemetry.ts`)

**Purpose**: Central orchestrator for all tracing operations

**Key Features**:

- Singleton pattern with global instance (`globalThis.__TELEMETRY__`)
- OpenTelemetry tracer management
- Dynamic proxy-based instrumentation
- Context propagation and baggage handling

```typescript
class Telemetry {
  public tracer: Tracer;
  name: string;

  static init(config: OtelConfig = {}): Telemetry;
  static get(): Telemetry;
  static getActiveSpan(): Span | undefined;

  traceClass<T>(instance: T, options): T;
  traceMethod<TMethod>(method: TMethod, context): TMethod;

  static setBaggage(baggage: Record<string, BaggageEntry>): Context;
  static withContext(ctx: Context, fn: () => void): void;
}
```

#### 2. Decorator System (`packages/core/src/telemetry/telemetry.decorators.ts`)

**Purpose**: Automatic instrumentation via TypeScript decorators

```typescript
// Method-level tracing
@withSpan({
  spanName: 'custom-operation',
  skipIfNoTelemetry: true,
  spanKind: SpanKind.CLIENT
})
async someMethod() { ... }

// Class-level tracing
@InstrumentClass({
  prefix: 'mastra',
  excludeMethods: ['getLogger', 'getTelemetry'],
  spanKind: SpanKind.INTERNAL
})
export class Mastra { ... }
```

## Configuration

### OtelConfig Type Definition

```typescript
type OtelConfig = {
  serviceName?: string; // Service identifier
  enabled?: boolean; // Enable/disable tracing
  tracerName?: string; // Custom tracer name
  sampling?: SamplingStrategy; // Trace sampling configuration
  disableLocalExport?: boolean; // Skip local storage export
  export?: {
    type: "otlp" | "console" | "custom";
    protocol?: "grpc" | "http";
    endpoint?: string;
    headers?: Record<string, string>;
  };
};
```

### Sampling Strategies

```typescript
type SamplingStrategy =
  | { type: "ratio"; probability: number } // Probabilistic sampling
  | { type: "always_on" } // Sample all traces
  | { type: "always_off" } // No sampling
  | { type: "parent_based"; root: { probability: number } }; // Inherit from parent
```

### Initialization

```typescript
// Basic setup
const mastra = new Mastra({
  telemetry: {
    serviceName: "my-ai-service",
    enabled: true,
    export: {
      type: "otlp",
      endpoint: "https://api.honeycomb.io/v1/traces",
      headers: {
        "x-honeycomb-team": process.env.HONEYCOMB_API_KEY,
      },
    },
  },
});

// Advanced configuration
const mastra = new Mastra({
  telemetry: {
    serviceName: "production-ai-agent",
    sampling: {
      type: "parent_based",
      root: { probability: 0.1 }, // 10% sampling for root spans
    },
    export: {
      type: "otlp",
      protocol: "grpc",
      endpoint: "https://otel-collector.company.com:4317",
    },
  },
});
```

## Automatic Instrumentation

### Components with Built-in Tracing

1. **Mastra Core** - Main orchestrator class
2. **Agents** - AI agent interactions and tool usage
3. **Storage Operations** - Database and storage calls
4. **Vector Operations** - Vector database interactions
5. **Workflows** - Step-based execution flows
6. **Integrations** - Third-party API calls
7. **Tools** - Tool executions and results
8. **LLM Calls** - Language model interactions

### Instrumentation Methods

#### 1. Decorator-based (Preferred)

```typescript
@InstrumentClass({ prefix: "my-service" })
class MyService {
  @withSpan({ spanName: "complex-operation" })
  async processData(data: any) {
    // Automatically traced
  }
}
```

#### 2. Proxy-based (Dynamic)

```typescript
const telemetry = Telemetry.get();
const tracedStorage = telemetry.traceClass(storage, {
  spanNamePrefix: "storage",
  excludeMethods: ["__setTelemetry"],
});
```

#### 3. Manual Tracing

```typescript
const telemetry = Telemetry.get();
const span = telemetry.tracer.startSpan("manual-operation");

try {
  // Your code here
  span.setAttributes({ "operation.type": "data-processing" });
  const result = await processData();
  span.setAttribute("result.count", result.length);
} catch (error) {
  span.recordException(error);
  span.setStatus({ code: SpanStatusCode.ERROR });
} finally {
  span.end();
}
```

## Span Data Structure

### Standard Span Format

```typescript
type Span = {
  id: string; // Unique span identifier
  parentSpanId: string | null; // Parent span reference
  traceId: string; // Trace identifier
  name: string; // Operation name
  scope: string; // Instrumentation scope
  kind: number; // Span kind (INTERNAL, CLIENT, SERVER, etc.)
  status: SpanStatus; // Success/error status
  events: SpanEvent[]; // Timeline events
  links: any[]; // Links to other spans
  attributes: Record<string, string | number | boolean>;
  startTime: number; // Start timestamp (nanoseconds)
  endTime: number; // End timestamp (nanoseconds)
  duration: number; // Duration (nanoseconds)
  other: SpanOther; // Additional metadata
  createdAt: string; // ISO creation timestamp
};
```

### Automatic Attributes

**Context Attributes**:

- `componentName` - Component generating the span
- `runId` - Execution run identifier
- `http.request_id` - HTTP request correlation

**Method Attributes**:

- `{spanName}.argument.{index}` - Serialized input arguments
- `{spanName}.result` - Serialized return value
- `{component}.name` - Component class name
- `{component}.method.name` - Method name

**Error Attributes**:

- Exception details and stack traces
- Error status codes and messages

## Export System

### 1. Storage Exporter (`packages/core/src/telemetry/storage-exporter.ts`)

Stores traces in configured Mastra storage backend for local development and debugging.

```typescript
class StorageExporter implements SpanExporter {
  export(
    spans: ReadableSpan[],
    resultCallback: (result: ExportResult) => void,
  ): void;
  forceFlush(): Promise<void>;
  shutdown(): Promise<void>;
}
```

### 2. Cloud Exporter (`packages/cloud/src/telemetry/index.ts`)

Native integration with Mastra Cloud platform.

```typescript
// Configuration
const mastra = new Mastra({
  telemetry: {
    export: {
      type: "custom",
      exporter: new MastraCloudExporter({
        endpoint: "https://cloud.mastra.ai/api/telemetry",
        accessToken: process.env.MASTRA_CLOUD_TOKEN,
      }),
    },
  },
});
```

### 3. Composite Exporter (`packages/core/src/telemetry/composite-exporter.ts`)

Enables simultaneous export to multiple destinations.

```typescript
const compositeExporter = new CompositeExporter({
  exporters: [
    new StorageExporter(storage),
    new OTLPTraceExporter({
      url: "https://api.honeycomb.io/v1/traces",
    }),
    new MastraCloudExporter(cloudConfig),
  ],
});
```

## Integration with LLMOps Platforms

### Supported Platforms

1. **Langfuse** - LLM observability and analytics
2. **Langsmith** - LangChain ecosystem tracing
3. **SigNoz** - Open-source APM
4. **New Relic** - Enterprise APM
5. **Braintrust** - AI evaluation platform
6. **Dash0** - Modern observability
7. **Traceloop** - AI-focused tracing
8. **Langwatch** - LLM monitoring
9. **Laminar** - AI observability
10. **Keywords AI** - AI platform analytics

### Integration Pattern

```typescript
// OTLP-compatible platforms (most common)
const mastra = new Mastra({
  telemetry: {
    export: {
      type: "otlp",
      endpoint: "https://platform-endpoint.com/v1/traces",
      headers: {
        Authorization: `Bearer ${process.env.PLATFORM_API_KEY}`,
        "Content-Type": "application/json",
      },
    },
  },
});

// Custom exporter for specialized platforms
import { createCustomExporter } from "./platform-exporter";

const mastra = new Mastra({
  telemetry: {
    export: {
      type: "custom",
      exporter: createCustomExporter({
        apiKey: process.env.PLATFORM_API_KEY,
        endpoint: "https://platform.com/traces",
      }),
    },
  },
});
```

## Development and Debugging

### Local Development Setup

```typescript
// Enable console export for debugging
const mastra = new Mastra({
  telemetry: {
    serviceName: "dev-service",
    export: { type: "console" },
  },
});
```

### Playground Integration

The Mastra CLI playground provides real-time trace visualization:

```bash
pnpm dev:playground
```

Features:

- **Live Trace Viewing**: Real-time span updates
- **Trace Tree Visualization**: Hierarchical span relationships
- **Filtering and Search**: Find specific operations
- **Performance Analysis**: Duration and error analysis

### Server API Endpoints

**Trace Storage**: `POST /api/telemetry`

- Receives OTLP trace data
- Processes and stores spans
- Returns storage confirmation

**Trace Retrieval**: `GET /api/telemetry`

- Query stored traces with filtering
- Support for time range, component, and status filters
- Returns trace trees with full span data

## Context Propagation

### Baggage System

Mastra uses OpenTelemetry baggage for cross-component context sharing:

```typescript
// Set context in one component
const ctx = Telemetry.setBaggage({
  componentName: { value: "ai-agent" },
  runId: { value: "run-123" },
  userId: { value: "user-456" },
});

// Context automatically propagates to child spans
Telemetry.withContext(ctx, () => {
  // All operations here inherit the baggage context
  agent.processRequest(userInput);
});
```

### Context Attributes

```typescript
// Automatically added to all spans in context
const { requestId, componentName, runId } = getBaggageValues(context);
span.setAttribute("http.request_id", requestId);
span.setAttribute("componentName", componentName);
span.setAttribute("runId", runId);
```

## Best Practices for LLMOps Integration

### 1. Custom Attribute Extraction

```typescript
// Extract LLM-specific metrics
@withSpan({ spanName: 'llm-completion' })
async callLLM(prompt: string, model: string) {
  const span = Telemetry.getActiveSpan();

  // Add LLM-specific attributes
  span?.setAttributes({
    'llm.model': model,
    'llm.prompt.length': prompt.length,
    'llm.provider': 'openai'
  });

  const response = await this.llmClient.complete(prompt);

  // Add response metrics
  span?.setAttributes({
    'llm.response.length': response.length,
    'llm.tokens.input': response.usage.prompt_tokens,
    'llm.tokens.output': response.usage.completion_tokens,
    'llm.tokens.total': response.usage.total_tokens,
    'llm.cost.usd': calculateCost(response.usage, model)
  });

  return response;
}
```

### 2. Custom Exporter Implementation

```typescript
import { SpanExporter, ReadableSpan } from "@opentelemetry/sdk-trace-base";

export class LLMOpsExporter implements SpanExporter {
  constructor(private config: { apiKey: string; endpoint: string }) {}

  export(
    spans: ReadableSpan[],
    callback: (result: ExportResult) => void,
  ): void {
    const llmSpans = spans.filter(
      (span) => span.name.includes("llm") || span.attributes["llm.model"],
    );

    // Transform to platform-specific format
    const transformedSpans = llmSpans.map((span) => ({
      traceId: span.spanContext().traceId,
      spanId: span.spanContext().spanId,
      operationName: span.name,
      startTime: span.startTime,
      duration: span.duration,
      tags: this.extractLLMTags(span.attributes),
      logs: span.events.map((event) => ({
        timestamp: event.time,
        fields: event.attributes,
      })),
    }));

    // Send to LLMOps platform
    this.sendToLLMOps(transformedSpans)
      .then(() => callback({ code: ExportResultCode.SUCCESS }))
      .catch((error) =>
        callback({
          code: ExportResultCode.FAILED,
          error,
        }),
      );
  }

  private extractLLMTags(attributes: any): Record<string, any> {
    return Object.keys(attributes)
      .filter((key) => key.startsWith("llm."))
      .reduce(
        (tags, key) => {
          tags[key] = attributes[key];
          return tags;
        },
        {} as Record<string, any>,
      );
  }
}
```

### 3. Sampling for High-Volume LLM Operations

```typescript
// Use ratio-based sampling for production LLM calls
const mastra = new Mastra({
  telemetry: {
    sampling: {
      type: "ratio",
      probability: 0.05, // Sample 5% of LLM calls
    },
    export: {
      type: "otlp",
      endpoint: process.env.LLMOPS_ENDPOINT,
    },
  },
});
```

### 4. Error Tracking and Performance Monitoring

```typescript
@withSpan({ spanName: 'agent-workflow' })
async executeAgentWorkflow(input: any) {
  const span = Telemetry.getActiveSpan();

  try {
    // Set initial attributes
    span?.setAttributes({
      'workflow.type': 'agent-execution',
      'workflow.input.size': JSON.stringify(input).length
    });

    const startTime = Date.now();
    const result = await this.processWorkflow(input);
    const duration = Date.now() - startTime;

    // Record performance metrics
    span?.setAttributes({
      'workflow.duration.ms': duration,
      'workflow.steps.count': result.steps.length,
      'workflow.success': true
    });

    return result;
  } catch (error) {
    // Record error details
    span?.setAttributes({
      'workflow.error.type': error.constructor.name,
      'workflow.error.recoverable': this.isRecoverableError(error),
      'workflow.success': false
    });

    span?.recordException(error);
    throw error;
  }
}
```

## Performance Considerations

### 1. Selective Instrumentation

```typescript
// Skip tracing for high-frequency, low-value operations
@withSpan({
  spanName: 'data-validation',
  skipIfNoTelemetry: true  // Skip if no active tracing
})
async validateInput(data: any) {
  // Light validation logic that doesn't need tracing in production
}
```

### 2. Attribute Optimization

```typescript
// Avoid large serialized objects in attributes
span.setAttribute("request.summary", {
  size: data.length,
  type: data.constructor.name,
  // Don't include full data payload
});
```

### 3. Batch Export Configuration

```typescript
// Configure batch processor for better performance
const batchProcessor = new BatchSpanProcessor(exporter, {
  maxExportBatchSize: 100,
  scheduledDelayMillis: 1000,
  exportTimeoutMillis: 5000,
});
```

## Troubleshooting

### Common Issues

1. **Missing Traces**: Check if telemetry is enabled and properly initialized
2. **High Memory Usage**: Reduce sampling rate or increase batch export frequency
3. **Export Failures**: Verify endpoint URLs and authentication credentials
4. **Missing Context**: Ensure proper context propagation in async operations

### Debugging Commands

```bash
# Check telemetry configuration
OTEL_LOG_LEVEL=debug pnpm dev:playground

# Verify trace export
curl -X GET http://localhost:4000/api/telemetry

# Test OTLP endpoint connectivity
curl -X POST ${OTEL_ENDPOINT}/v1/traces \
  -H "Content-Type: application/json" \
  -d '{"test": "connectivity"}'
```

This comprehensive guide provides the foundation for building advanced LLMOps integrations on top of Mastra's existing tracing infrastructure. The system's OpenTelemetry foundation ensures compatibility with most observability platforms while providing the flexibility to create custom integrations for specialized LLMOps tools.

# TypeScript Modernization Plan: Eliminate `any` Usage

## Executive Summary

This document outlines a comprehensive approach to eliminate all `any` usage in the Bedrock instrumentation codebase and replace it with proper TypeScript patterns. The current extensive use of `any` defeats the purpose of TypeScript's type system and prevents us from leveraging compile-time safety, IDE intelligence, and self-documenting code.

## Current State Analysis

The codebase currently has **59 TypeScript linting errors**, primarily from `any` usage across:
- Instrumentation core logic (22 errors)
- Type definitions (6 errors) 
- Request/response attribute extraction (4 errors)
- Test helpers (27 errors)

## Why This Matters: TypeScript Expert Perspective

### The Problems with `any`
1. **Defeats Type Safety**: `any` disables TypeScript's compile-time checking
2. **No IDE Intelligence**: Autocomplete, refactoring, and navigation are broken
3. **Runtime Errors**: Type errors surface at runtime instead of compile time
4. **Poor Documentation**: Code becomes self-documenting when properly typed
5. **Maintenance Burden**: Changes don't propagate through the type system

### The Benefits of Proper Typing
1. **Compile-time Safety**: Catch errors before they reach production
2. **IDE Intelligence**: Better autocomplete, refactoring, and navigation
3. **Self-Documenting Code**: Types serve as living documentation
4. **Refactoring Safety**: Changes propagate through the type system
5. **Performance**: TypeScript compiler optimizations work better

## Anti-Patterns Currently Used

### 1. `any` Everywhere
```typescript
// ❌ Current anti-pattern
private patch(moduleExports: any, moduleVersion?: string) {
  (original: any) => {
    return function patchedSend(this: unknown, command: any) {
```

### 2. `Record<string, any>` for Everything
```typescript
// ❌ Current anti-pattern
function extractInvocationParameters(
  requestBody: InvokeModelRequestBody,
): Record<string, any> {
  const invocationParams: Record<string, any> = {};
```

### 3. Type Guards with `any`
```typescript
// ❌ Current anti-pattern
export function isTextContent(content: any): content is TextContent {
```

### 4. Missing SDK Types
```typescript
// ❌ Current anti-pattern - not using AWS SDK types
export function setSpanAttribute(span: Span, key: string, value: any) {
```

## Implementation Plan

## Phase 1: AWS SDK Type Integration

### Problem Analysis
The AWS SDK provides comprehensive TypeScript types, but we're ignoring them and using `any` instead.

### Solution: Import and Use AWS SDK Types
```typescript
import { 
  InvokeModelCommand, 
  InvokeModelCommandInput,
  InvokeModelCommandOutput,
  InvokeModelWithResponseStreamCommand,
  InvokeModelWithResponseStreamCommandOutput,
  BedrockRuntimeClient 
} from "@aws-sdk/client-bedrock-runtime";
```

### Current Problem
```typescript
private patch(moduleExports: any, moduleVersion?: string) {
  if (moduleExports?.BedrockRuntimeClient) {
    this._wrap(
      moduleExports.BedrockRuntimeClient.prototype,
      "send",
      (original: any) => {
        return function patchedSend(this: unknown, command: any) {
```

### Proper TypeScript Solution
```typescript
interface BedrockModuleExports {
  BedrockRuntimeClient: typeof BedrockRuntimeClient;
}

type BedrockCommand = InvokeModelCommand | InvokeModelWithResponseStreamCommand;

private patch(moduleExports: BedrockModuleExports, moduleVersion?: string) {
  if (moduleExports?.BedrockRuntimeClient) {
    this._wrap(
      moduleExports.BedrockRuntimeClient.prototype,
      "send",
      (original: BedrockRuntimeClient['send']) => {
        return function patchedSend(this: BedrockRuntimeClient, command: BedrockCommand) {
```

### Implementation Steps
1. **Import AWS SDK Types**: Add comprehensive type imports
2. **Create Module Interface**: Define `BedrockModuleExports` interface
3. **Type Command Union**: Create `BedrockCommand` union type
4. **Update Method Signatures**: Replace all `any` with proper AWS SDK types
5. **Type Client Methods**: Use `BedrockRuntimeClient['send']` for method typing

## Phase 2: Domain-Specific Type Creation

### Problem Analysis
We're using `Record<string, any>` for domain-specific data structures instead of creating proper interfaces.

### Solution: Create Specific Domain Types

#### Invocation Parameters
```typescript
interface InvocationParameters {
  anthropic_version?: string;
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  top_k?: number;
  stop_sequences?: string[];
}

function extractInvocationParameters(
  requestBody: InvokeModelRequestBody,
): InvocationParameters {
  const invocationParams: InvocationParameters = {};
  
  if (requestBody.anthropic_version) {
    invocationParams.anthropic_version = requestBody.anthropic_version;
  }
  if (requestBody.max_tokens) {
    invocationParams.max_tokens = requestBody.max_tokens;
  }
  // ... etc
  
  return invocationParams;
}
```

#### Stream Processing Types
```typescript
interface StreamChunk {
  chunk?: {
    bytes: Uint8Array;
  };
}

interface StreamEventData {
  type: 'message_start' | 'content_block_start' | 'content_block_delta' | 'message_delta';
  message?: {
    usage?: UsageInfo;
  };
  content_block?: ToolUseContent | TextContent;
  delta?: {
    text?: string;
  };
  usage?: UsageInfo;
}

interface StreamResponseBody {
  id: string;
  type: "message";
  role: "assistant";
  model: string;
  content: (TextContent | ToolUseContent)[];
  stop_reason: string;
  stop_sequence: string | null;
  usage: UsageInfo;
}
```

### Implementation Steps
1. **Create Parameter Interface**: Define `InvocationParameters` with optional fields
2. **Create Stream Types**: Define interfaces for stream processing
3. **Create Response Types**: Define structured response body interfaces
4. **Update Function Signatures**: Replace `Record<string, any>` with specific types
5. **Add Type Guards**: Create proper type validation functions

## Phase 3: OpenTelemetry Type Alignment

### Problem Analysis
We're using `any` for OpenTelemetry attributes instead of using the provided `AttributeValue` type.

### Solution: Use OpenTelemetry Types Properly

#### Attribute Helpers
```typescript
import { AttributeValue } from "@opentelemetry/api";

export function setSpanAttribute(
  span: Span, 
  key: string, 
  value: AttributeValue | null | undefined
): void {
  if (value !== undefined && value !== null) {
    span.setAttribute(key, value);
  }
}

export function setSpanAttributes(
  span: Span, 
  attributes: Record<string, AttributeValue | null | undefined>
): void {
  Object.entries(attributes).forEach(([key, value]) => {
    setSpanAttribute(span, key, value);
  });
}
```

#### Span Attribute Keys (Advanced)
```typescript
type SpanAttributeKey = 
  | "llm.model_name"
  | "llm.provider" 
  | "llm.system"
  | "input.value"
  | "output.value"
  | "input.mime_type"
  | "output.mime_type"
  | "llm.invocation_parameters"
  | `llm.input_messages.${number}.message.role`
  | `llm.input_messages.${number}.message.content`
  | `llm.output_messages.${number}.message.role`
  | `llm.output_messages.${number}.message.content`
  | `llm.tools.${number}.tool.json_schema`;

export function setSpanAttribute(
  span: Span, 
  key: SpanAttributeKey, 
  value: AttributeValue | null | undefined
): void {
  if (value !== undefined && value !== null) {
    span.setAttribute(key, value);
  }
}
```

### Implementation Steps
1. **Import AttributeValue**: Use OpenTelemetry's proper type
2. **Update Helper Functions**: Replace `any` with `AttributeValue | null | undefined`
3. **Create Attribute Key Types**: Define specific span attribute keys (optional)
4. **Update All Callers**: Ensure all attribute setting uses proper types
5. **Add Type Assertions**: Where needed for complex attribute values

## Phase 4: Type Guard Modernization

### Problem Analysis
Current type guards use `any` and don't leverage TypeScript's discriminated unions properly.

### Solution: Discriminated Unions and Proper Type Guards

#### Content Type Discrimination
```typescript
type MessageContent = 
  | TextContent 
  | ImageContent 
  | ToolUseContent 
  | ToolResultContent;

// The discriminated union makes type guards much simpler
export function isTextContent(content: MessageContent): content is TextContent {
  return content.type === "text";
}

export function isImageContent(content: MessageContent): content is ImageContent {
  return content.type === "image";
}

export function isToolUseContent(content: MessageContent): content is ToolUseContent {
  return content.type === "tool_use";
}

export function isToolResultContent(content: MessageContent): content is ToolResultContent {
  return content.type === "tool_result";
}
```

#### Enhanced Type Guards for External Data
```typescript
// For data coming from external sources that might not be properly typed
export function isValidTextContent(content: unknown): content is TextContent {
  return (
    content !== null &&
    typeof content === "object" &&
    (content as Record<string, unknown>).type === "text" &&
    typeof (content as Record<string, unknown>).text === "string"
  );
}

export function isValidMessageContent(content: unknown): content is MessageContent {
  return (
    isValidTextContent(content) ||
    isValidImageContent(content) ||
    isValidToolUseContent(content) ||
    isValidToolResultContent(content)
  );
}
```

### Implementation Steps
1. **Create Discriminated Union**: Define `MessageContent` union type
2. **Simplify Type Guards**: Use discriminated union for internal type guards
3. **Add Validation Guards**: Create `isValid*` functions for external data
4. **Update Usage**: Replace `any` type guards with proper discriminated unions
5. **Add Array Helpers**: Create type-safe array processing functions

## Phase 5: Test Code Cleanup

### Problem Analysis
Test code has extensive `any` usage that prevents proper testing of type safety.

### Solution: Proper Test Types

#### Test Data Generators
```typescript
interface TestInvokeModelCommand {
  input: {
    modelId: string;
    body: string | Uint8Array;
  };
}

interface TestBedrockClient {
  send(command: TestInvokeModelCommand): Promise<InvokeModelCommandOutput>;
}

export function generateTestCommand(
  modelId: string, 
  body: InvokeModelRequestBody
): TestInvokeModelCommand {
  return {
    input: {
      modelId,
      body: JSON.stringify(body)
    }
  };
}
```

#### Mock Type Definitions
```typescript
interface MockSpanData {
  name: string;
  attributes: Record<string, AttributeValue>;
  status: {
    code: number;
  };
  events: unknown[];
  links: unknown[];
}

interface MockSpanExporter {
  getFinishedSpans(): MockSpanData[];
  reset(): void;
}
```

### Implementation Steps
1. **Create Test Interfaces**: Define proper types for test data
2. **Type Mock Functions**: Replace `any` with specific mock types
3. **Update Test Helpers**: Use proper types for test utilities
4. **Fix VCR Helpers**: Remove `any` from recording/playback code
5. **Add Type Assertions**: Where needed for test-specific requirements

## Advanced TypeScript Patterns

### Branded Types for Type Safety
```typescript
type ModelId = string & { __brand: 'ModelId' };
type SpanName = string & { __brand: 'SpanName' };

function createModelId(id: string): ModelId {
  return id as ModelId;
}

function createSpanName(name: string): SpanName {
  return name as SpanName;
}
```

### Conditional Types for Attribute Values
```typescript
type AttributeValueFor<T> = T extends string 
  ? string 
  : T extends number 
    ? number 
    : T extends boolean 
      ? boolean 
      : string;

function setTypedAttribute<T>(
  span: Span, 
  key: string, 
  value: T | null | undefined
): void {
  if (value !== undefined && value !== null) {
    span.setAttribute(key, value as AttributeValueFor<T>);
  }
}
```

### Utility Types for Better APIs
```typescript
type RequiredKeys<T> = {
  [K in keyof T]-?: {} extends Pick<T, K> ? never : K;
}[keyof T];

type OptionalKeys<T> = {
  [K in keyof T]-?: {} extends Pick<T, K> ? K : never;
}[keyof T];

type PartialBy<T, K extends keyof T> = Omit<T, K> & Partial<Pick<T, K>>;
```

## Implementation Strategy

### Phase-by-Phase Approach
1. **Phase 1**: Start with AWS SDK types (lowest risk, highest impact)
2. **Phase 2**: Create domain types (moderate risk, high impact)
3. **Phase 3**: OpenTelemetry alignment (low risk, moderate impact)
4. **Phase 4**: Type guard modernization (moderate risk, high impact)
5. **Phase 5**: Test cleanup (low risk, low impact)

### Testing Strategy
- Run TypeScript compiler after each phase
- Ensure all tests pass after each change
- Use `tsc --noEmit --strict` for maximum type checking
- Consider enabling `noImplicitAny` in tsconfig.json

### Risk Mitigation
- Make changes incrementally
- Keep existing functionality intact
- Test thoroughly after each phase
- Consider adding more comprehensive type tests

## Expected Outcomes

### Before Implementation
- 59 TypeScript linting errors
- No compile-time type safety
- Poor IDE support
- Difficult refactoring
- Runtime type errors

### After Implementation
- 0 TypeScript linting errors
- Full compile-time type safety
- Excellent IDE support with autocomplete
- Safe refactoring capabilities
- Self-documenting code through types

## Conclusion

This modernization plan transforms the codebase from a TypeScript project that uses `any` everywhere (essentially JavaScript with type annotations) to a properly typed TypeScript project that leverages the full power of the type system.

The key insight is that proper TypeScript isn't about adding type annotations to existing JavaScript code - it's about designing with types from the ground up to create safer, more maintainable, and more expressive code.

Each phase builds on the previous one, creating a comprehensive type-safe foundation that will make the codebase more robust and easier to maintain long-term.
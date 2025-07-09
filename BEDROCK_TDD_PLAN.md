# TDD Implementation Plan: JavaScript Bedrock Instrumentation with Tool Support

## Overview

Build JavaScript Bedrock instrumentation with **full Python parity**, including comprehensive tool calling support. Follow strict TDD red-green-refactor cycles with VCR testing for deterministic API recordings.

## Target Coverage (Python Parity)

Based on Python implementation analysis:

### Core APIs
1. **InvokeModel API** (Legacy): Raw JSON request/response, vendor-specific parsing
2. **Converse API** (Modern): Structured messages, system prompts, tool calling, multi-modal
3. **Streaming Variants**: Both InvokeModel and Converse streaming responses
4. **InvokeAgent**: Complex hierarchical agent workflow tracing

### Advanced Features
- **Tool Support**: Input tool calls, tool results, output tool calls
- **Multi-Modal**: Text, images (base64), complex message structures
- **Token Counting**: From response headers and usage objects
- **Error Handling**: Proper span status and exception recording
- **Context Propagation**: Session, user, metadata attributes

## Test-by-Test Implementation Strategy

### Phase 1: InvokeModel Foundation

#### Test 1: Basic InvokeModel Text Messages ✅
**Status**: COMPLETED - Test passing with full instrumentation
**File**: `test/recordings/should-create-spans-for-invokemodel-calls.json`

```typescript
it("should create spans for InvokeModel calls", async () => {
  // Tests basic text message exchange
  // Validates all core attributes (model, system, provider, tokens)
  // Verifies input/output message structure
  // Checks invocation parameters extraction
});
```

**Implementation Requirements**:
- AWS SDK v3 command pattern wrapping (`BedrockRuntimeClient.prototype.send`)
- Request/response body parsing for Anthropic Messages API
- Basic span lifecycle management (start, attributes, end)
- Token count extraction from response usage object
- Message structure conversion to OpenInference format

#### Test 2: Tool Call Support - Basic Function Call ✅
**Status**: COMPLETED - Test passing with tool calling support
**File**: `test/recordings/should-handle-tool-calling-with-function-definitions.json`

```typescript
it("should handle InvokeModel with tool definitions and calls", async () => {
  const toolDefinition = {
    name: "get_weather",
    description: "Get current weather for a location",
    input_schema: {
      type: "object",
      properties: {
        location: { type: "string" }
      }
    }
  };
  
  const command = new InvokeModelCommand({
    modelId: TEST_MODEL_ID,
    body: JSON.stringify({
      anthropic_version: "bedrock-2023-05-31",
      max_tokens: 100,
      tools: [toolDefinition],
      messages: [{ role: "user", content: "What's the weather in San Francisco?" }]
    })
  });
  
  // Expected response contains tool_use block
  // Verify tool call extraction and attribution
  // Check tool call span hierarchy (if implementing sub-spans)
});
```

**New Implementation Requirements**:
- Tool definition parsing and attribute extraction
- Tool call identification in response content blocks
- Tool call to OpenInference ToolCall conversion
- Proper message attribution for tool calls

#### Test 3: Tool Results Processing
**File**: `test/recordings/tools/invoke-model-tool-result.json`

```typescript
it("should handle tool result responses", async () => {
  const messages = [
    { role: "user", content: "What's the weather?" },
    { 
      role: "assistant", 
      content: [
        { type: "tool_use", id: "tool_1", name: "get_weather", input: { location: "SF" } }
      ]
    },
    {
      role: "user",
      content: [
        { 
          type: "tool_result", 
          tool_use_id: "tool_1", 
          content: "Sunny, 72°F"
        }
      ]
    },
    { role: "user", content: "Thanks! Anything else I should know?" }
  ];
  
  // Verify tool result parsing and message attribution
  // Check multi-turn conversation handling
  // Validate tool_call_id propagation
});
```

**New Implementation Requirements**:
- Tool result content type parsing
- Tool call ID tracking and attribution
- Multi-turn conversation message processing
- Tool result to OpenInference Message conversion

#### Test 4: Multi-Modal Messages (Text + Image)
**File**: `test/recordings/multimodal/invoke-model-image.json`

```typescript
it("should handle multi-modal messages with images", async () => {
  const imageData = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==";
  
  const messages = [{
    role: "user",
    content: [
      { type: "text", text: "What do you see in this image?" },
      { 
        type: "image", 
        source: { 
          type: "base64", 
          media_type: "image/png", 
          data: imageData 
        }
      }
    ]
  }];
  
  // Verify image content parsing and attribute extraction
  // Check multi-part message content handling
  // Validate image URL formatting for OpenInference
});
```

**New Implementation Requirements**:
- Multi-part message content processing
- Image source parsing and base64 handling
- Image URL formatting (`data:image/png;base64,{data}`)
- Mixed content type message support

#### Test 5: Error Handling and Edge Cases
**File**: `test/recordings/errors/invoke-model-error.json`

```typescript
it("should handle API errors gracefully", async () => {
  // Test invalid model ID, malformed requests, rate limiting
  // Verify proper span status setting (ERROR)
  // Check exception recording
  // Ensure span ends even on errors
});

it("should handle missing token counts gracefully", async () => {
  // Test responses with partial/missing usage data
  // Verify instrumentation doesn't crash
  // Check that available tokens are still captured
});
```

### Phase 2: Streaming Support

#### Test 6: InvokeModel Streaming - Basic Text
**File**: `test/recordings/streaming/invoke-model-stream-basic.json`

```typescript
it("should handle InvokeModelWithResponseStream", async () => {
  const command = new InvokeModelWithResponseStreamCommand({
    modelId: TEST_MODEL_ID,
    body: JSON.stringify({
      anthropic_version: "bedrock-2023-05-31",
      max_tokens: 100,
      messages: [{ role: "user", content: "Tell me a short story" }]
    })
  });
  
  // Verify streaming response handling
  // Check incremental content accumulation
  // Validate final span attributes after stream completion
});
```

**New Implementation Requirements**:
- Streaming response body parsing
- Event stream processing (content_block_start, content_block_delta, etc.)
- Response accumulation across stream chunks
- Proper span completion after stream ends

#### Test 7: Streaming Tool Calls
**File**: `test/recordings/streaming/invoke-model-stream-tools.json`

```typescript
it("should handle streaming responses with tool calls", async () => {
  // Test streaming tool call responses
  // Verify tool call assembly from stream chunks
  // Check proper tool call attribution in streaming context
});
```

**New Implementation Requirements**:
- Streaming tool call event parsing
- Tool call assembly from multiple stream events
- Tool ID tracking across stream boundaries

### Phase 3: Converse API Support

#### Test 8: Basic Converse API
**File**: `test/recordings/converse/converse-basic.json`

```typescript
it("should handle Converse API calls", async () => {
  const command = new ConverseCommand({
    modelId: TEST_MODEL_ID,
    messages: [{ role: "user", content: [{ text: "Hello!" }] }],
    inferenceConfig: { maxTokens: 100 }
  });
  
  // Verify Converse API specific attribute extraction
  // Check structured message format handling
  // Validate inference config parameter processing
});
```

**New Implementation Requirements**:
- Converse API command detection and handling
- Structured message format processing
- Inference config parameter extraction
- Response format differences from InvokeModel

#### Test 9: Converse with System Prompts
**File**: `test/recordings/converse/converse-system.json`

```typescript
it("should handle system prompts in Converse API", async () => {
  const command = new ConverseCommand({
    modelId: TEST_MODEL_ID,
    system: [{ text: "You are a helpful assistant." }],
    messages: [{ role: "user", content: [{ text: "Hello!" }] }]
  });
  
  // Verify system prompt extraction and attribution
  // Check system message integration into message sequence
});
```

#### Test 10: Converse Tool Calling (Modern API)
**File**: `test/recordings/converse/converse-tools.json`

```typescript
it("should handle tool calling via Converse API", async () => {
  const toolConfig = {
    tools: [{
      toolSpec: {
        name: "calculator",
        description: "Perform mathematical calculations",
        inputSchema: {
          json: {
            type: "object",
            properties: {
              expression: { type: "string" }
            }
          }
        }
      }
    }]
  };
  
  // Verify Converse-style tool definition handling
  // Check toolSpec vs tools format differences
  // Validate structured tool response parsing
});
```

**New Implementation Requirements**:
- Converse-specific tool definition format
- ToolSpec parsing and conversion
- Structured tool response handling
- Tool config vs tools parameter differences

### Phase 4: Advanced Features

#### Test 11: Converse Streaming
**File**: `test/recordings/streaming/converse-stream.json`

```typescript
it("should handle ConverseStream API", async () => {
  // Test Converse streaming with structured messages
  // Verify different streaming event format from InvokeModel
  // Check tool calls in streaming Converse responses
});
```

#### Test 12: InvokeAgent Support
**File**: `test/recordings/agent/invoke-agent-basic.json`

```typescript
it("should handle InvokeAgent with hierarchical spans", async () => {
  // Test complex agent workflow tracing
  // Verify nested span relationships (Agent → Tool → LLM)
  // Check knowledge base integration spans
  // Validate agent session and workflow tracking
});
```

**New Implementation Requirements**:
- InvokeAgent command detection
- Hierarchical span creation and management
- Agent workflow step identification
- Knowledge base span attribution

#### Test 13: Knowledge Base Integration
**File**: `test/recordings/agent/retrieve-and-generate.json`

```typescript
it("should handle Retrieve and RetrieveAndGenerate operations", async () => {
  // Test RAG operation instrumentation
  // Verify document retrieval span creation
  // Check retrieval score and metadata extraction
});
```

#### Test 14: Complex Multi-Agent Workflows
**File**: `test/recordings/agent/multi-agent-workflow.json`

```typescript
it("should handle complex agent collaboration", async () => {
  // Test multi-agent conversation flows
  // Verify proper span hierarchy for agent interactions
  // Check agent memory and state tracking
});
```

### Phase 5: Context and Configuration

#### Test 15: Context Propagation
```typescript
it("should propagate OpenInference context attributes", async () => {
  // Test session, user, metadata context propagation
  // Verify context managers work with Bedrock spans
  // Check suppression context handling
});
```

#### Test 16: Trace Configuration
```typescript
it("should respect TraceConfig settings", async () => {
  // Test privacy controls and data masking
  // Verify payload size limits
  // Check custom attribute filtering
});
```

## Implementation Architecture

### Core Components

#### 1. BedrockInstrumentation Class
```typescript
export class BedrockInstrumentation extends InstrumentationBase {
  protected _wrap(
    BedrockRuntimeClient.prototype,
    "send",
    this._wrapSend.bind(this)
  );
}
```

#### 2. Command Pattern Detection
```typescript
private _wrapSend(original: Function) {
  return function(command: any) {
    if (command instanceof InvokeModelCommand) {
      return handleInvokeModel(this, command, original);
    } else if (command instanceof ConverseCommand) {
      return handleConverse(this, command, original);
    } else if (command instanceof InvokeAgentCommand) {
      return handleInvokeAgent(this, command, original);
    }
    return original.apply(this, arguments);
  };
}
```

#### 3. Attribute Extraction Modules
```typescript
// src/attributes/
├── anthropic-attributes.ts    // Tool calls, messages, tokens
├── model-attributes.ts        // Model ID parsing, provider info
├── request-attributes.ts      // Input parameters, body parsing
└── response-attributes.ts     // Output parsing, token extraction
```

#### 4. Streaming Response Handlers
```typescript
// src/streaming/
├── invoke-model-stream.ts     // InvokeModel streaming handler
├── converse-stream.ts         // Converse streaming handler
└── response-accumulator.ts    // Stream event accumulation
```

## Success Criteria

### Test Coverage Goals
- **100% Python Parity**: All Python features implemented
- **Comprehensive Tool Support**: Input/output tool calls fully supported
- **Streaming Support**: All streaming APIs properly instrumented
- **Error Handling**: Graceful handling of all error conditions
- **Performance**: Minimal overhead, efficient stream processing

### Quality Standards
- **Semantic Conventions**: Strict compliance with OpenInference spec
- **VCR Testing**: All tests use real API recordings
- **TypeScript**: Full type safety and proper interfaces
- **Documentation**: Clear examples and API documentation

### Developer Experience
- **Easy Test Writing**: Simple pattern for adding new test scenarios
- **Clear Failures**: Descriptive error messages when tests fail
- **Fast Iteration**: Quick test runs with cached recordings
- **Debugging Support**: Easy to trace instrumentation issues

## Implementation Notes

### JavaScript vs Python Differences

#### Stream Handling
- **JS Advantage**: `stream.tee()` for parallel processing
- **Python**: Sequential accumulation with custom iterators

#### AWS SDK Patterns
- **JS**: Command-based (`client.send(command)`)
- **Python**: Method-based (`client.invoke_model()`)

#### Tool Call Processing
- **JS**: Native JSON/object manipulation
- **Python**: More complex type coercion

### Key Technical Challenges

1. **Streaming Tool Calls**: Assembly from multiple stream events
2. **Hierarchical Spans**: Agent workflow span relationships
3. **Context Propagation**: OpenTelemetry context in async streams
4. **Error Boundaries**: Proper span lifecycle in error conditions
5. **Memory Management**: Efficient stream processing without leaks

## Current Implementation Status

### ✅ Completed Features
- **VCR Testing Infrastructure**: Complete tooling suite with recording, validation, and cleanup
- **Basic InvokeModel Support**: Core instrumentation with span creation and attribute extraction
- **Tool Calling Support**: Tool definition parsing and tool call attribution
- **Semantic Conventions**: OpenInference-compliant attribute mapping
- **Code Quality**: Prettier formatting, ESLint compliance, TypeScript typing

### 🎯 Current Architecture
- **BedrockInstrumentation Class**: Extends InstrumentationBase with AWS SDK wrapping
- **Command Pattern Detection**: Identifies InvokeModelCommand and other command types
- **Attribute Extraction**: Modular functions for request/response processing
- **VCR Integration**: Nock-based recording system with credential sanitization

### 📋 Next Implementation Priorities
1. **Test 3**: Tool results processing and multi-turn conversations
2. **Test 4**: Multi-modal messages with image support
3. **Test 5**: Error handling and edge cases
4. **Phase 2**: Streaming support (InvokeModelWithResponseStream)
5. **Phase 3**: Converse API support

### 🔧 Technical Debt to Address
- Fix remaining ESLint violations (unused imports, explicit any types)
- Complete type checking compliance
- Add comprehensive error handling tests
- Implement streaming response support

This comprehensive test plan ensures full Python parity while leveraging JavaScript/TypeScript advantages for a robust, maintainable implementation.
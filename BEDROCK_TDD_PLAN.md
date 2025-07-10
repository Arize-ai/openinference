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

#### Test 3: Tool Results Processing ✅
**Status**: COMPLETED - Test passing with tool result processing
**File**: `test/recordings/should-handle-tool-result-responses.json`

```typescript
it("should handle tool result responses", async () => {
  const testData = generateToolResultMessage({
    initialPrompt: "What's the weather in Paris?",
    toolUseId: "toolu_123",
    toolName: "get_weather",
    toolInput: { location: "Paris, France" },
    toolResult: "The weather in Paris is currently 22°C and sunny.",
    followupPrompt: "Great! What should I wear?"
  });
  
  // Verifies tool result parsing and message attribution
  // Checks multi-turn conversation handling  
  // Validates tool_call_id propagation
  // Tests complex content arrays (tool_result + text)
});
```

**Completed Implementation**:
- ✅ Tool result content type parsing
- ✅ Tool call ID tracking and attribution  
- ✅ Multi-turn conversation message processing
- ✅ Tool result to OpenInference Message conversion
- ✅ Complex content array handling

## Priority 1: Core InvokeModel Completeness

### Phase 1 Remaining Tests (4 tests needed for complete InvokeModel coverage)

#### Test 4: Missing Token Count Handling
**File**: `test/recordings/should-handle-missing-token-counts-gracefully.json`

```typescript
it("should handle missing token counts gracefully", async () => {
  // Mock response with missing usage object
  const mockResponse = {
    id: "msg_test",
    content: [{ type: "text", text: "Response without tokens" }],
    // usage object intentionally omitted
  };
  
  // Verify instrumentation doesn't crash
  // Check that span completes successfully
  // Ensure available attributes are still captured
  // Validate graceful degradation
});
```

**Implementation Requirements**:
- Graceful handling when `response.usage` is undefined
- No crashes or exceptions when token data missing
- Optional token attribute setting
- Comprehensive error boundary testing

#### Test 5: Multi-Modal Messages (Text + Image)
**File**: `test/recordings/should-handle-multi-modal-messages-with-images.json`

```typescript
it("should handle multi-modal messages with images", async () => {
  const testData = generateMultiModalMessage({
    textPrompt: "What do you see in this image?",
    imageData: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
    mediaType: "image/png"
  });
  
  // Verify image content parsing and attribute extraction
  // Check multi-part message content handling
  // Validate image URL formatting: data:image/png;base64,{data}
  // Test mixed content type message support
});
```

**Implementation Requirements**:
- Multi-part message content processing
- Image source parsing and base64 handling
- Image URL formatting for OpenInference (`data:image/png;base64,{data}`)
- Mixed content type message support (text + image)
- Proper MIME type detection and attribute assignment

#### Test 6: API Error Handling
**File**: `test/recordings/should-handle-api-errors-gracefully.json`

```typescript
it("should handle API errors gracefully", async () => {
  // Test scenarios:
  // 1. Invalid model ID (400 error)
  // 2. Malformed request body (400 error)
  // 3. Rate limiting (429 error)
  // 4. Service unavailable (503 error)
  
  const invalidModelCommand = new InvokeModelCommand({
    modelId: "invalid-model-id",
    body: JSON.stringify({ messages: [{ role: "user", content: "test" }] })
  });
  
  // Verify proper span status setting (ERROR)
  // Check exception recording and error details
  // Ensure span ends even on errors
  // Validate error message attribution
});
```

**Implementation Requirements**:
- Proper span status setting (`span.setStatus({ code: SpanStatusCode.ERROR })`)
- Exception recording with error details
- Span lifecycle completion even on errors
- Error message and code attribution
- Different error type handling (4xx vs 5xx)

#### Test 7: Multiple Tools in Single Request
**File**: `test/recordings/should-handle-multiple-tools-in-single-request.json`

```typescript
it("should handle multiple tools in single request", async () => {
  const testData = generateToolCallMessage({
    prompt: "What's the weather in San Francisco and what's 15 * 23?",
    tools: [
      commonTools.weather,    // get_weather function
      commonTools.calculator, // calculate function  
      commonTools.webSearch   // web_search function
    ]
  });
  
  // Verify all tool definitions properly parsed
  // Check tool schema conversion for multiple tools
  // Validate tool call attribution when multiple tools available
  // Test complex tool interaction scenarios
});
```

**Implementation Requirements**:
- Multiple tool definition parsing in single request
- Tool schema conversion for all tool types
- Proper indexing and attribution of multiple tools
- Tool selection and call attribution validation
- Complex tool interaction handling

## Priority 2: Streaming Support

### Streaming InvokeModel Tests (3 tests for streaming completeness)

#### Test 8: InvokeModelWithResponseStream - Basic Text
**File**: `test/recordings/should-handle-invoke-model-with-response-stream.json`

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
  // Test stream event processing (content_block_start, content_block_delta, etc.)
});
```

**Implementation Requirements**:
- Streaming response body parsing
- Event stream processing (content_block_start, content_block_delta, message_stop)
- Response accumulation across stream chunks
- Proper span completion after stream ends
- Stream error boundary handling

#### Test 9: Streaming Tool Calls
**File**: `test/recordings/should-handle-streaming-tool-calls.json`

```typescript
it("should handle streaming responses with tool calls", async () => {
  // Test streaming tool call responses
  // Verify tool call assembly from stream chunks
  // Check proper tool call attribution in streaming context
  // Validate tool ID tracking across stream boundaries
});
```

**Implementation Requirements**:
- Streaming tool call event parsing
- Tool call assembly from multiple stream events
- Tool ID tracking across stream boundaries
- Incremental tool call building and validation

#### Test 10: Stream Error Handling
**File**: `test/recordings/should-handle-streaming-errors.json`

```typescript
it("should handle streaming errors gracefully", async () => {
  // Test connection drops during streaming
  // Verify incomplete stream handling
  // Check partial response processing
  // Validate span lifecycle on stream failures
});
```

**Implementation Requirements**:
- Stream connection error handling
- Partial response processing and recovery
- Span lifecycle management during stream failures
- Error attribution for streaming-specific issues

## Priority 3: Advanced Scenarios

### Advanced InvokeModel Tests (3 tests for complete coverage)

#### Test 11: Context Attributes
**File**: `test/recordings/should-handle-context-attributes.json`

```typescript
it("should propagate OpenInference context attributes", async () => {
  // Test session, user, metadata context propagation
  // Verify context managers work with Bedrock spans
  // Check suppression context handling
});
```

#### Test 12: Non-Anthropic Models  
**File**: `test/recordings/should-handle-non-anthropic-models.json`

```typescript
it("should handle non-Anthropic models via Bedrock", async () => {
  // Test Amazon Titan models
  // Test Mistral models  
  // Test Meta Llama models
  // Verify model-specific response parsing
});
```

#### Test 13: Large Payload Edge Cases
**File**: `test/recordings/should-handle-large-payloads.json`

```typescript
it("should handle large payloads and timeouts", async () => {
  // Test very long messages (approaching token limits)
  // Verify timeout handling
  // Check memory efficiency with large responses
  // Validate performance with complex conversations
});
```

**Implementation Requirements**:
- Memory efficient processing of large payloads
- Timeout handling and graceful degradation
- Performance optimization for complex conversations
- Resource cleanup and memory management

## InvokeModel API Test Coverage Summary

### ✅ **Completed Tests (3/13)**
1. **Test 1**: Basic InvokeModel Text Messages - COMPLETE
2. **Test 2**: Tool Call Support - Basic Function Call - COMPLETE  
3. **Test 3**: Tool Results Processing - COMPLETE

### 🎯 **Priority 1: Core Completeness (4 tests)**
4. **Test 4**: Missing Token Count Handling
5. **Test 5**: Multi-Modal Messages (Text + Image)
6. **Test 6**: API Error Handling
7. **Test 7**: Multiple Tools in Single Request

### 🚀 **Priority 2: Streaming Support (3 tests)**
8. **Test 8**: InvokeModelWithResponseStream - Basic Text
9. **Test 9**: Streaming Tool Calls
10. **Test 10**: Stream Error Handling

### ⭐ **Priority 3: Advanced Scenarios (3 tests)**
11. **Test 11**: Context Attributes
12. **Test 12**: Non-Anthropic Models
13. **Test 13**: Large Payload Edge Cases

### Next Implementation Goal
**Today's Focus**: Complete Priority 1 tests (Tests 4-7) to achieve comprehensive InvokeModel API coverage for all core functionality scenarios.

## Beyond InvokeModel API

### Future API Coverage (Not part of current InvokeModel focus)
- **Converse API Support**: Modern structured message API
- **InvokeAgent API Support**: Agent workflow tracing
- **RAG Operations**: Retrieve and RetrieveAndGenerate
- **Multi-Agent Collaboration**: Complex agent interactions

This plan ensures complete InvokeModel API instrumentation before expanding to other Bedrock APIs.

## Current Implementation Status

### ✅ Completed Features (Tests 1-3)
- **VCR Testing Infrastructure**: Complete tooling suite with recording, validation, and cleanup
- **Basic InvokeModel Support**: Core instrumentation with span creation and attribute extraction
- **Tool Calling Support**: Tool definition parsing and tool call attribution  
- **Tool Results Processing**: Multi-turn conversations and complex content arrays
- **Semantic Conventions**: OpenInference-compliant attribute mapping
- **Code Quality**: Prettier formatting, working TypeScript compilation

### 🎯 Current Architecture
- **BedrockInstrumentation Class**: Extends InstrumentationBase with AWS SDK wrapping
- **Command Pattern Detection**: Identifies InvokeModelCommand with dynamic model ID support
- **Attribute Extraction**: Modular functions for request/response processing including complex content
- **VCR Integration**: Nock-based recording system with credential sanitization and flexible model ID handling

### 📋 Next Implementation Priorities (Today's Goal)
1. **Test 4**: Missing token count handling
2. **Test 5**: Multi-modal messages with image support
3. **Test 6**: API error handling and edge cases
4. **Test 7**: Multiple tools in single request

### 🔧 Current Status
- **3 of 13 InvokeModel tests complete** (23% coverage)
- **Priority 1 focus**: Complete core InvokeModel functionality (Tests 4-7)
- **Future phases**: Streaming support and advanced scenarios

This focused approach ensures complete InvokeModel API coverage before expanding to other Bedrock APIs.
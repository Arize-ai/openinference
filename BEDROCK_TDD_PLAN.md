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

#### Test 4: Missing Token Count Handling ✅
**Status**: COMPLETED - Test passing with graceful token handling
**File**: `test/recordings/should-handle-missing-token-counts-gracefully.json`

```typescript
it("should handle missing token counts gracefully", async () => {
  // Modified response with missing usage object
  const mockResponse = {
    id: "msg_test", 
    content: [{ type: "text", text: "Response without tokens" }],
    // usage object intentionally omitted
  };
  
  // Verified instrumentation doesn't crash
  // Span completes successfully with SpanStatusCode.OK
  // Available attributes are still captured
  // Graceful degradation working correctly
});
```

**Completed Implementation**:
- ✅ Graceful handling when `response.usage` is undefined
- ✅ No crashes or exceptions when token data missing
- ✅ Optional token attribute setting (only set when present)
- ✅ Comprehensive error boundary testing
- ✅ Existing implementation already had proper error handling

#### Test 5: Multi-Modal Messages (Text + Image) ✅
**Status**: COMPLETED - Test passing with full multi-modal support
**File**: `test/recordings/should-handle-multi-modal-messages-with-images.json`

```typescript
it("should handle multi-modal messages with images", async () => {
  const imageData = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==";
  
  const command = new InvokeModelCommand({
    modelId: TEST_MODEL_ID,
    body: JSON.stringify({
      anthropic_version: "bedrock-2023-05-31",
      max_tokens: TEST_MAX_TOKENS,
      messages: [{
        role: "user",
        content: [
          { type: "text", text: "What do you see in this image?" },
          { type: "image", source: { type: "base64", media_type: "image/png", data: imageData }}
        ]
      }]
    })
  });
  
  // Verified image content parsing and attribute extraction
  // Multi-part message content handling working
  // Image URL formatting: data:image/png;base64,{data} ✅
  // Mixed content type message support confirmed
});
```

**Completed Implementation**:
- ✅ Multi-part message content processing via enhanced `_extractTextFromContent()`
- ✅ Image source parsing and base64 handling from `source.data` field
- ✅ Image URL formatting for OpenInference (`data:image/png;base64,{data}`) via new `_formatImageUrl()`
- ✅ Mixed content type message support (text + image combined in content string)
- ✅ Proper MIME type detection and attribute assignment using `source.media_type`
- ✅ Full OpenInference compliance for multi-modal observability

#### Test 6: API Error Handling ✅
**Status**: COMPLETED - Test passing with graceful error handling
**File**: `test/recordings/should-handle-api-errors-gracefully.json`

```typescript
it("should handle API errors gracefully", async () => {
  // Test invalid model ID (should trigger 400 error)
  const invalidModelCommand = new InvokeModelCommand({
    modelId: "invalid-model-id",
    body: JSON.stringify({
      anthropic_version: "bedrock-2023-05-31",
      max_tokens: TEST_MAX_TOKENS,
      messages: [{ role: "user", content: "This should fail" }]
    })
  });

  // Expect the API call to throw an error
  await expect(client.send(invalidModelCommand)).rejects.toThrow();

  // Verify span was created and marked as error
  const span = verifySpanBasics(spanExporter);
  expect(span.status.code).toBe(2); // SpanStatusCode.ERROR
  expect(span.status.message).toBeDefined();
});
```

**Completed Implementation**:
- ✅ Proper span status setting (`SpanStatusCode.ERROR`) for API errors
- ✅ Exception recording with error details (`span.recordException(error)`)
- ✅ Span lifecycle completion even on errors
- ✅ Error message and code attribution in span status
- ✅ Enhanced VCR infrastructure to handle HTTP error status codes (400, 500, etc.)
- ✅ Request attribute preservation in error spans for debugging context

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

### ✅ **Completed Tests (6/13)**
1. **Test 1**: Basic InvokeModel Text Messages - COMPLETE
2. **Test 2**: Tool Call Support - Basic Function Call - COMPLETE  
3. **Test 3**: Tool Results Processing - COMPLETE
4. **Test 4**: Missing Token Count Handling - COMPLETE ✅
5. **Test 5**: Multi-Modal Messages (Text + Image) - COMPLETE ✅
6. **Test 6**: API Error Handling - COMPLETE ✅

### 🎯 **Priority 1: Core Completeness (1 test remaining)**
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
**Current Focus**: Complete Priority 1 with Test 7 (Multiple Tools) to achieve comprehensive InvokeModel API coverage for all core functionality scenarios.

## Beyond InvokeModel API

### Future API Coverage (Not part of current InvokeModel focus)
- **Converse API Support**: Modern structured message API
- **InvokeAgent API Support**: Agent workflow tracing
- **RAG Operations**: Retrieve and RetrieveAndGenerate
- **Multi-Agent Collaboration**: Complex agent interactions

This plan ensures complete InvokeModel API instrumentation before expanding to other Bedrock APIs.

## Current Implementation Status

### ✅ Completed Features (Tests 1-6)
- **VCR Testing Infrastructure**: Complete tooling suite with recording, validation, and cleanup
- **Basic InvokeModel Support**: Core instrumentation with span creation and attribute extraction
- **Tool Calling Support**: Tool definition parsing and tool call attribution  
- **Tool Results Processing**: Multi-turn conversations and complex content arrays
- **Missing Token Count Handling**: Graceful degradation when usage data is unavailable
- **Multi-Modal Message Support**: Text + image content extraction with OpenInference data URL formatting
- **API Error Handling**: Graceful error handling with proper span status and exception recording
- **Enhanced VCR Infrastructure**: HTTP error status code support (400, 500, etc.) for realistic error testing
- **Semantic Conventions**: OpenInference-compliant attribute mapping
- **Code Quality**: Prettier formatting, working TypeScript compilation

### 🎯 Current Architecture
- **BedrockInstrumentation Class**: Extends InstrumentationBase with AWS SDK wrapping
- **Command Pattern Detection**: Identifies InvokeModelCommand with dynamic model ID support
- **Enhanced Content Extraction**: Modular functions for request/response processing including multi-modal content
- **Multi-Modal Support**: `_extractTextFromContent()` and `_formatImageUrl()` for image data extraction
- **VCR Integration**: Nock-based recording system with credential sanitization and flexible model ID handling
- **Error Resilience**: Graceful handling of missing usage data, malformed responses, and API errors
- **Refactored Test Infrastructure**: Clean separation of concerns with organized helper modules

### 📋 Next Implementation Priorities (Updated Goal)
1. ✅ **Test 4**: Missing token count handling - **COMPLETED**
2. ✅ **Test 5**: Multi-modal messages with image support - **COMPLETED**
3. ✅ **Test 6**: API error handling and edge cases - **COMPLETED**
4. **Test 7**: Multiple tools in single request - **NEXT**

### 🔧 Current Status
- **6 of 13 InvokeModel tests complete** (46% coverage)
- **Priority 1 progress**: 1 of 2 core functionality tests remaining (83% complete)
- **Recent achievements**: API error handling + test infrastructure refactoring

### 🔧 **Recent Refactoring Achievements**
**Conservative Test Infrastructure Refactoring Completed (7 incremental steps)**:
- ✅ **Step 1-2**: Extracted VCR helper functions to `test/helpers/vcr-helpers.ts`
- ✅ **Step 3**: Extracted span verification helpers to `test/helpers/test-helpers.ts`  
- ✅ **Step 4**: Moved recording path helpers to VCR module
- ✅ **Step 5**: Centralized test constants in `test/config/constants.ts`
- ✅ **Step 6**: Successfully converted test-data-generators.js to TypeScript with proper types
- ✅ **Step 7**: Cleaned up main test file imports and removed unused code

**Benefits Achieved**:
- **Better Organization**: Clear separation of concerns across focused modules
- **Enhanced Maintainability**: Easier to locate and modify specific test functionality
- **Type Safety**: Proper TypeScript interfaces for test data generation
- **Cleaner Main File**: Test file focuses on test scenarios, not infrastructure
- **Reusable Components**: VCR infrastructure ready for future API tests

This focused approach ensures complete InvokeModel API coverage before expanding to other Bedrock APIs.

## Cleanup Before Merge

Before merging to main branch, perform the following cleanup tasks to ensure production readiness:

### 🧹 **Test Data Generators Cleanup**
**File**: `test/helpers/test-data-generators.ts`

**Current Status**: Extensive generator functions (~400+ lines) with only ~5% actual usage
- **Keep (Middle Ground Approach)**:
  - `generateBasicTextMessage` - Core text message generation
  - `generateToolCallMessage` - Tool calling scenarios  
  - `generateToolResultMessage` - ✅ Currently used
  - `generateMultiModalMessage` - Multi-modal support (images)
  - `generateToolDefinition` - Tool schema generation
  - `commonTools` - Reusable tool definitions

- **Remove (Dead Code)**:
  - `generateConverseMessage` - Future Converse API (not InvokeModel scope)
  - `generateConverseWithTools` - Future Converse API
  - `generateAgentMessage` - Future InvokeAgent API  
  - `generateStreamingVariants` - Future streaming support
  - `errorScenarios` - Inline error generation more appropriate
  - `generateTestSuite` - Overly complex, unused
  - Associated TypeScript interfaces for removed functions

**Expected Result**: Reduce file from ~470 lines to ~200 lines while keeping essential generators

### 📁 **Project Memory Files Removal**
Remove all Claude project memory and planning files that should not be committed to the repository:

**Files to Delete**:
- `BEDROCK_TDD_PLAN.md` - Development planning document
- `BEST_PRACTICES.md` - Development methodology notes  
- `SESSION_SUMMARY.md` - Session history and progress tracking
- `VCR_TOOLING_PLAN.md` - VCR implementation planning
- `VCR_IMPLEMENTATION_SUMMARY.md` - Implementation summary
- Any other `*.md` files in package root that are development artifacts

**Files to Keep**:
- `README.md` - Package documentation
- `CLAUDE.md` - Should remain for development guidance (if desired)
- `CHANGELOG.md` - If present, keep for version history

### ✅ **Pre-Merge Checklist**
- [ ] Test data generators cleaned up (middle ground approach)
- [ ] All development memory files removed
- [ ] All tests passing (6/6 current tests)
- [ ] TypeScript compilation clean (`npx tsc --noEmit`)
- [ ] Linting passes (`npm run lint`)
- [ ] No debug console.logs in production code
- [ ] Package.json version updated appropriately
- [ ] CHANGELOG.md updated with new features (if applicable)

### 🎯 **Merge Readiness Criteria**
- **Functionality**: All 6 InvokeModel tests passing with comprehensive coverage
- **Code Quality**: Clean, maintainable code structure with proper TypeScript types
- **Documentation**: Essential documentation present, development artifacts removed
- **Performance**: VCR testing infrastructure efficient and reliable

This cleanup ensures the package is production-ready while maintaining development velocity for future enhancements.
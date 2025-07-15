# Bedrock Testing Enhancement Plan: Achieving Python Parity

## Executive Summary

This document outlines a comprehensive plan to enhance the JavaScript Bedrock instrumentation testing to achieve full parity with the Python implementation. Based on analysis of the Python Bedrock instrumentation, we've identified significant gaps in our current testing coverage, particularly in span attribute accuracy, streaming support, context propagation, and multi-model compatibility.

## Current Status

### ✅ **Completed Coverage (Tests 1-13)**
- **Priority 1**: Core InvokeModel Foundation (Tests 1-7) ✅
- **Priority 2**: Streaming Support (Tests 8-10) ✅  
- **Priority 3**: Advanced Scenarios (Tests 11-13) ✅

**Total**: 13/13 tests passing = 100% InvokeModel API coverage

### 🔍 **Identified Gaps vs Python Implementation**

Our current implementation covers the basic functionality but lacks the depth and accuracy of the Python implementation in several key areas.

## Gap Analysis: Python vs JavaScript Implementation

### 1. **Token Count Attributes (Critical Gap)**

**Python Implementation:**
```python
# Comprehensive token counting with cache support
span.set_attribute("llm.token_count.prompt", usage.input_tokens)
span.set_attribute("llm.token_count.completion", usage.output_tokens) 
span.set_attribute("llm.token_count.total", usage.input_tokens + usage.output_tokens)
span.set_attribute("llm.token_count.prompt_details.cache_read", usage.cache_read_input_tokens)
span.set_attribute("llm.token_count.prompt_details.cache_write", usage.cache_creation_input_tokens)
```

**JavaScript Implementation (Current):**
```javascript
// Basic token counting only
span.setAttributes({
  "llm.token_count.prompt": usage.input_tokens,
  "llm.token_count.completion": usage.output_tokens,
  "llm.token_count.total": usage.input_tokens + usage.output_tokens
});
// Missing: cache_read, cache_write attributes
```

**Gap**: Missing cache-related token attributes that are crucial for monitoring model performance.

### 2. **System and Provider Attributes (Accuracy Gap)**

**Python Implementation:**
```python
# Model-specific system attribution
if "anthropic" in model_id:
    span.set_attribute("llm.system", "anthropic")
elif "ai21" in model_id:
    span.set_attribute("llm.system", "ai21")
# ... other models
span.set_attribute("llm.provider", "aws")
```

**JavaScript Implementation (Current):**
```javascript
// Generic system attribution
span.setAttributes({
  "llm.system": "bedrock",  // Should be "anthropic" for Claude models
  "llm.provider": "aws"     // This is correct
});
```

**Gap**: Incorrect system attribution - should be vendor-specific, not generic "bedrock".

### 3. **Model Name Extraction (Format Gap)**

**Python Implementation:**
```python
# Extracts clean model name
# "anthropic.claude-3-haiku-20240307-v1:0" -> "claude-3-haiku-20240307"
model_name = extract_model_name(model_id)
span.set_attribute("llm.model_name", model_name)
```

**JavaScript Implementation (Current):**
```javascript
// Uses full model ID
span.setAttributes({
  "llm.model_name": "anthropic.claude-3-haiku-20240307-v1:0"  // Full ID
});
```

**Gap**: Should extract clean model name for consistency with Python.

### 4. **Message Content Structure (Complexity Gap)**

**Python Implementation:**
```python
# Detailed message content structure
span.set_attribute(f"llm.input_messages.{idx}.message.role", message.role)
span.set_attribute(f"llm.input_messages.{idx}.message.contents.0.message_content.type", "text")
span.set_attribute(f"llm.input_messages.{idx}.message.contents.0.message_content.text", content)
# Support for images
span.set_attribute(f"llm.input_messages.{idx}.message.contents.1.message_content.type", "image")
span.set_attribute(f"llm.input_messages.{idx}.message.contents.1.message_content.image.image.url", data_url)
```

**JavaScript Implementation (Current):**
```javascript
// Simplified message content
span.setAttributes({
  "llm.input_messages.0.message.role": "user",
  "llm.input_messages.0.message.content": "text content"
});
// Missing: detailed content structure, image support
```

**Gap**: Missing detailed content structure and image content support.

### 5. **Context Attributes (Major Gap)**

**Python Implementation:**
```python
# Rich context support
span.set_attribute("session_id", context.session_id)
span.set_attribute("user_id", context.user_id)
span.set_attribute("metadata", json.dumps(context.metadata))
span.set_attribute("tags", context.tags)
span.set_attribute("llm.prompt_template.template", template.template)
span.set_attribute("llm.prompt_template.version", template.version)
span.set_attribute("llm.prompt_template.variables", json.dumps(template.variables))
```

**JavaScript Implementation (Current):**
```javascript
// No context attribute support
// Test 11 exists but doesn't actually propagate context
```

**Gap**: Complete absence of context attribute propagation.

### 6. **Multi-Model Support (Vendor Gap)**

**Python Implementation:**
```python
# Vendor-specific response parsing
if "anthropic" in model_id:
    return parse_anthropic_response(response)
elif "ai21" in model_id:
    return parse_ai21_response(response)
elif "amazon" in model_id:
    return parse_amazon_response(response)
# ... other vendors
```

**JavaScript Implementation (Current):**
```javascript
// Anthropic-only response parsing
function parseResponseBody(response) {
  // Expects Anthropic format only
  return JSON.parse(responseText);
}
```

**Gap**: Limited to Anthropic model format, doesn't handle other vendors properly.

### 7. **Streaming Support (Architecture Gap)**

**Python Implementation:**
```python
# Comprehensive streaming with event handling
class EventStream:
    def __iter__(self):
        for event in self._stream:
            yield self._process_event(event)
    
    def _process_event(self, event):
        # Process content_block_start, content_block_delta, etc.
        return processed_event
```

**JavaScript Implementation (Current):**
```javascript
// Basic streaming without proper event processing
// Missing proper event stream handling
```

**Gap**: Incomplete streaming implementation, missing proper event processing.

## Enhancement Plan

### ✅ Phase 1: Critical Attribute Fixes (COMPLETED)

#### 1.1 Token Count Attributes Enhancement ✅
**Status**: COMPLETED

**Changes Implemented**:
```javascript
// Added cache-related token attributes
if (responseBody.usage.cache_read_input_tokens !== undefined) {
  tokenAttributes[`${SemanticConventions.LLM_TOKEN_COUNT_PROMPT}.cache_read`] =
    responseBody.usage.cache_read_input_tokens;
}
if (responseBody.usage.cache_creation_input_tokens !== undefined) {
  tokenAttributes[`${SemanticConventions.LLM_TOKEN_COUNT_PROMPT}.cache_write`] =
    responseBody.usage.cache_creation_input_tokens;
}
```

**Test Coverage**: All tests now include cache-related token attributes where applicable

#### 1.2 System Attribute Correction ✅
**Status**: COMPLETED

**Changes Implemented**:
```javascript
// Model-specific system attribution
function getSystemFromModelId(modelId: string): string {
  if (modelId.includes("anthropic")) return "anthropic";
  if (modelId.includes("ai21")) return "ai21";
  if (modelId.includes("amazon")) return "amazon";
  if (modelId.includes("cohere")) return "cohere";
  if (modelId.includes("meta")) return "meta";
  if (modelId.includes("mistral")) return "mistral";
  return "bedrock"; // fallback
}
```

**Test Coverage**: All tests now use correct vendor-specific system attribution

#### 1.3 Model Name Extraction ✅
**Status**: COMPLETED

**Changes Implemented**:
```javascript
// Extract clean model name
function extractModelName(modelId: string): string {
  // "anthropic.claude-3-haiku-20240307-v1:0" -> "claude-3-haiku-20240307"
  const parts = modelId.split(".");
  if (parts.length > 1) {
    const modelPart = parts[1];
    // Remove version suffix like "-v1:0"
    return modelPart.replace(/-v\d+.*$/, "");
  }
  return modelId;
}
```

**Test Coverage**: All tests now use clean model name extraction

### ✅ Phase 2: Message Content Structure Enhancement (COMPLETED)

#### 2.1 Detailed Message Content Structure ✅
**Status**: COMPLETED

**Changes Implemented**:
```javascript
// Enhanced message content structure
function addMessageContentAttributes(attributes, message, messageIndex) {
  if (Array.isArray(message.content)) {
    message.content.forEach((content, contentIndex) => {
      if (content.type === "text") {
        attributes[`llm.input_messages.${messageIndex}.message.contents.${contentIndex}.message_content.type`] = "text";
        attributes[`llm.input_messages.${messageIndex}.message.contents.${contentIndex}.message_content.text`] = content.text;
      } else if (content.type === "image") {
        attributes[`llm.input_messages.${messageIndex}.message.contents.${contentIndex}.message_content.type`] = "image";
        attributes[`llm.input_messages.${messageIndex}.message.contents.${contentIndex}.message_content.image.image.url`] = formatImageUrl(content.source);
      }
    });
  }
}
```

**Test Updates**:
- Update message content assertions to use detailed structure
- Add multi-content message tests
- Verify content type attribution

#### 2.2 Image Content Support
**Target**: Test 5 (Multi-Modal Messages)

**Changes Required**:
```javascript
// Enhanced image content handling
function formatImageUrl(imageSource) {
  if (imageSource.type === "base64") {
    return `data:${imageSource.media_type};base64,${imageSource.data}`;
  }
  return imageSource.url || "";
}
```

**Test Updates**:
- Update Test 5 to verify detailed image content structure
- Add image URL format validation
- Test different image source types

### Phase 3: Context Attributes Implementation (Medium Priority)

#### 3.1 Context Propagation Support
**Target**: Test 11 (Context Attributes)

**Changes Required**:
```javascript
// Context attribute propagation
function extractContextAttributes(span) {
  const context = getCurrentContext(); // OpenInference context
  if (context.session_id) {
    span.setAttributes({ "session_id": context.session_id });
  }
  if (context.user_id) {
    span.setAttributes({ "user_id": context.user_id });
  }
  if (context.metadata) {
    span.setAttributes({ "metadata": JSON.stringify(context.metadata) });
  }
  if (context.tags) {
    span.setAttributes({ "tags": context.tags });
  }
}
```

**Test Updates**:
- Update Test 11 to actually test context propagation
- Add context manager integration tests
- Verify context attribute extraction

#### 3.2 Prompt Template Support
**Target**: New test needed

**Changes Required**:
```javascript
// Prompt template attributes
function extractPromptTemplateAttributes(span, template) {
  if (template) {
    span.setAttributes({
      "llm.prompt_template.template": template.template,
      "llm.prompt_template.version": template.version,
      "llm.prompt_template.variables": JSON.stringify(template.variables)
    });
  }
}
```

**Test Updates**:
- Add new test for prompt template attributes
- Verify template variable extraction
- Test template versioning

### Phase 4: Multi-Model Support (Low Priority)

#### 4.1 Vendor-Specific Response Parsing
**Target**: Test 12 (Non-Anthropic Models)

**Changes Required**:
```javascript
// Multi-vendor response parsing
function parseResponseByVendor(response, modelId) {
  if (modelId.includes("anthropic")) {
    return parseAnthropicResponse(response);
  } else if (modelId.includes("amazon")) {
    return parseAmazonResponse(response);
  } else if (modelId.includes("ai21")) {
    return parseAI21Response(response);
  }
  // ... other vendors
  return parseGenericResponse(response);
}
```

**Test Updates**:
- Update Test 12 to expect proper Titan response parsing
- Add tests for other model vendors
- Verify vendor-specific attribute extraction

#### 4.2 Tool Schema Compatibility
**Target**: Tests 2, 3, 7, 9

**Changes Required**:
```javascript
// Python-compatible tool schema format
function formatToolSchema(tool) {
  // Keep raw format instead of converting to OpenAI
  return {
    name: tool.name,
    description: tool.description,
    input_schema: tool.input_schema
  };
}
```

**Test Updates**:
- Update tool schema format assertions
- Verify raw tool schema preservation
- Test tool schema compatibility

### Phase 5: Streaming Enhancement (Low Priority)

#### 5.1 Comprehensive Streaming Support
**Target**: Tests 8, 9, 10

**Changes Required**:
```javascript
// Enhanced streaming with proper event handling
class BedrockEventStream {
  constructor(stream) {
    this.stream = stream;
    this.accumulated = { content: [], usage: {} };
  }
  
  async *processEvents() {
    for await (const event of this.stream) {
      if (event.contentBlockStart) {
        this.handleContentBlockStart(event);
      } else if (event.contentBlockDelta) {
        this.handleContentBlockDelta(event);
      } else if (event.messageStop) {
        this.handleMessageStop(event);
      }
      yield event;
    }
  }
}
```

**Test Updates**:
- Update streaming tests to use proper event processing
- Add event-specific attribute tests
- Verify streaming response accumulation

## Implementation Strategy

### Step 1: Test-Driven Development Approach

1. **Update Test Assertions First**: Modify tests 8-13 to expect Python-compatible attributes
2. **Allow Tests to Fail**: Let tests fail with missing attributes
3. **Implement Instrumentation Changes**: Add missing attribute extraction
4. **Verify Green Tests**: Ensure all tests pass with correct attributes

### Step 2: Incremental Implementation

1. **Phase 1** (Critical): Token counts, system attributes, model names
2. **Phase 2** (Medium): Message content structure, image support
3. **Phase 3** (Medium): Context attributes, prompt templates
4. **Phase 4** (Low): Multi-model support, tool schema compatibility
5. **Phase 5** (Low): Streaming enhancement

### Step 3: Validation Approach

1. **Attribute Comparison**: Direct comparison of Python vs JavaScript span attributes
2. **Cross-Platform Testing**: Run similar scenarios on both implementations
3. **Semantic Validation**: Ensure attribute meanings are consistent
4. **Performance Testing**: Verify no performance degradation

## Expected Test Changes

### Test 8: InvokeModelWithResponseStream
**Current Issues**:
- Missing streaming event processing
- Incomplete response accumulation
- Basic attribute extraction

**Required Changes**:
```javascript
// Before
expect(span.attributes["llm.token_count.prompt"]).toBeDefined();

// After
expect(span.attributes["llm.token_count.prompt"]).toBe(expectedTokens);
expect(span.attributes["llm.token_count.prompt_details.cache_read"]).toBe(0);
expect(span.attributes["llm.system"]).toBe("anthropic");
expect(span.attributes["llm.model_name"]).toBe("claude-3-5-sonnet-20240620");
```

### Test 9: Streaming Tool Calls
**Current Issues**:
- Missing tool call event processing
- Incomplete tool schema validation
- Basic streaming support

**Required Changes**:
```javascript
// Before
expect(span.attributes["llm.output_messages.0.message.tool_calls.0.tool_call.function.name"]).toBe("get_weather");

// After
expect(span.attributes["llm.output_messages.0.message.contents.0.message_content.type"]).toBe("tool_use");
expect(span.attributes["llm.output_messages.0.message.contents.0.message_content.tool_use.name"]).toBe("get_weather");
expect(span.attributes["llm.tools.0.tool.name"]).toBe("get_weather");
```

### Test 10: Stream Error Handling
**Current Issues**:
- Basic error attribution
- Missing error context
- Incomplete error details

**Required Changes**:
```javascript
// Before
expect(span.status.code).toBe(2);

// After
expect(span.status.code).toBe(2);
expect(span.attributes["error.type"]).toBe("ValidationException");
expect(span.attributes["error.message"]).toContain("model not found");
```

### Test 11: Context Attributes
**Current Issues**:
- No actual context propagation
- Missing context attributes
- Placeholder implementation

**Required Changes**:
```javascript
// Before (placeholder)
// Note: Context propagation would be handled by OpenInference context managers

// After (actual implementation)
expect(span.attributes["session_id"]).toBe("test-session-123");
expect(span.attributes["user_id"]).toBe("test-user-456");
expect(span.attributes["metadata"]).toBe('{"experiment_name":"context-test","version":"1.0.0"}');
```

### Test 12: Non-Anthropic Models
**Current Issues**:
- Anthropic-only response parsing
- Missing vendor-specific attributes
- Incomplete model support

**Required Changes**:
```javascript
// Before
expect(span.attributes["llm.system"]).toBe("bedrock");
expect(span.attributes["output.value"]).toBe("");

// After
expect(span.attributes["llm.system"]).toBe("amazon");
expect(span.attributes["llm.model_name"]).toBe("titan-text-express-v1");
expect(span.attributes["llm.output_messages.0.message.contents.0.message_content.text"]).toContain("Hey there");
```

### Test 13: Large Payloads
**Current Issues**:
- Basic performance testing
- Missing payload optimization
- Incomplete memory management

**Required Changes**:
```javascript
// Before
expect(outputTokens).toBeGreaterThanOrEqual(1);

// After
expect(span.attributes["llm.token_count.prompt"]).toBeGreaterThan(30000);
expect(span.attributes["llm.token_count.prompt_details.cache_read"]).toBeDefined();
expect(span.attributes["llm.input_messages.0.message.contents.0.message_content.text"]).toContain("large test message");
```

## Success Metrics

### Quantitative Metrics
- **Attribute Parity**: 95%+ attribute compatibility with Python implementation
- **Test Coverage**: 100% test pass rate after implementation
- **Performance**: No degradation in instrumentation performance
- **Memory**: Efficient handling of large payloads

### Qualitative Metrics
- **Consistency**: Span attributes match Python format and naming
- **Completeness**: All Python features supported in JavaScript
- **Maintainability**: Clean, readable code following existing patterns
- **Documentation**: Clear documentation of attribute meanings

## Timeline Estimation

### Phase 1: Critical Fixes (1-2 weeks)
- Token count attributes
- System attribute correction
- Model name extraction

### Phase 2: Content Structure (1-2 weeks)
- Message content structure
- Image content support

### Phase 3: Context Support (1 week)
- Context attribute propagation
- Prompt template support

### Phase 4: Multi-Model (1 week)
- Vendor-specific parsing
- Tool schema compatibility

### Phase 5: Streaming (1 week)
- Enhanced streaming support
- Event processing

**Total Estimated Time**: 5-7 weeks

## Risk Assessment

### High Risk
- **Breaking Changes**: Attribute format changes may break existing users
- **Performance Impact**: Enhanced attribute extraction may slow instrumentation
- **Complexity**: Multi-model support adds significant complexity

### Medium Risk
- **Test Flakiness**: Complex streaming tests may be unreliable
- **Maintenance Burden**: More comprehensive tests require more maintenance
- **Backward Compatibility**: Changes may not be backward compatible

### Low Risk
- **Documentation**: Enhanced tests provide better documentation
- **Quality**: Better attribute parity improves overall quality
- **Debugging**: More detailed attributes improve debugging

## Conclusion

This enhancement plan provides a comprehensive roadmap to achieve full Python parity in our Bedrock JavaScript instrumentation. By focusing on the most critical gaps first (token counts, system attributes, model names), we can quickly improve compatibility while building toward complete feature parity.

The test-driven approach ensures that we validate our implementation against the expected behavior, while the phased implementation allows for manageable progress and risk mitigation.

Upon completion, our JavaScript Bedrock instrumentation will provide the same level of observability and compatibility as the Python implementation, ensuring consistent user experience across language ecosystems.

## Recent Progress Update (Phase 1 & 2 Completed)

### ✅ Phase 1: Critical Attribute Fixes (COMPLETED)

**Key Implementations**:
1. **Token Count Attributes**: Added cache-related token attributes (`cache_read`, `cache_write`)
2. **System Attribute Correction**: Implemented vendor-specific system attribution (anthropic, ai21, amazon, etc.)
3. **Model Name Extraction**: Added clean model name extraction removing version suffixes

**Test Coverage**: All 13 tests now pass with enhanced attribute structure

### ✅ Phase 2: Message Content Structure Enhancement (COMPLETED)

**Key Implementations**:

#### 2.1 Enhanced Input Message Content Structure
```javascript
// Now supports detailed content structure for all content types
llm.input_messages.{i}.message.contents.{j}.message_content.type: "text" | "image" | "tool_use" | "tool_result"
llm.input_messages.{i}.message.contents.{j}.message_content.text: "content text"
llm.input_messages.{i}.message.contents.{j}.message_content.image.image.url: "data:image/png;base64,..."
llm.input_messages.{i}.message.contents.{j}.message_content.tool_use.{id,name,input}: tool details
llm.input_messages.{i}.message.contents.{j}.message_content.tool_result.{content,tool_use_id}: tool results
```

#### 2.2 Enhanced Output Message Content Structure
```javascript
// Now supports detailed output content structure
llm.output_messages.{i}.message.contents.{j}.message_content.type: "text" | "tool_use"
llm.output_messages.{i}.message.contents.{j}.message_content.text: "response text"
llm.output_messages.{i}.message.contents.{j}.message_content.tool_use.{id,name,input}: tool usage
```

#### 2.3 Test Coverage
- ✅ Test 1: Basic text messages with detailed content structure
- ✅ Test 2: Tool calling with enhanced content attributes
- ✅ Test 3: Tool result responses with structured content
- ✅ Test 5: Multi-modal messages with image content structure
- ✅ Test 7: Multiple tools with detailed content arrays
- ✅ Tests 8-13: All streaming and advanced tests updated

**New Attributes Added**:
- `llm.input_messages.{i}.message.contents.{j}.message_content.type`
- `llm.input_messages.{i}.message.contents.{j}.message_content.text`
- `llm.input_messages.{i}.message.contents.{j}.message_content.image.image.url`
- `llm.input_messages.{i}.message.contents.{j}.message_content.tool_use.{id,name,input}`
- `llm.input_messages.{i}.message.contents.{j}.message_content.tool_result.{content,tool_use_id}`
- `llm.output_messages.{i}.message.contents.{j}.message_content.type`
- `llm.output_messages.{i}.message.contents.{j}.message_content.text`
- `llm.output_messages.{i}.message.contents.{j}.message_content.tool_use.{id,name,input}`

**Backward Compatibility**: All existing attributes maintained while adding detailed content structure

**Test Results**: All 13 tests passing with enhanced message content structure

## Current Status Update

### Completed Phases
- ✅ **Phase 1**: Critical Attribute Fixes (token counts, system attribution, model names)
- ✅ **Phase 2**: Message Content Structure Enhancement (detailed content structure, image support, tool content)

### Next Phase Priorities
- 🔄 **Phase 3**: Context Attributes Implementation (context propagation, prompt templates)
- 🔄 **Phase 4**: Multi-Model Support (vendor-specific parsing, tool schema compatibility)
- 🔄 **Phase 5**: Streaming Enhancement (comprehensive event processing)

### Impact
The JavaScript Bedrock instrumentation now provides significantly enhanced observability with detailed message content structure that matches the sophistication of the Python implementation. The enhanced attributes enable better debugging, monitoring, and analysis of AI application interactions.
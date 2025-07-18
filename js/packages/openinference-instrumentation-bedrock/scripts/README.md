# Bedrock Instrumentation Validation Scripts

This directory contains validation scripts for testing the Bedrock instrumentation in real-world scenarios.

## Scripts

### `validate-invoke-model.ts`

Tests the Bedrock InvokeModel API instrumentation by making actual API calls and verifying traces are created correctly.

### `validate-converse.ts`

Tests the Bedrock Converse API instrumentation by executing a comprehensive conversation scenario that combines multiple test cases into a single realistic conversation flow.

#### Prerequisites

1. **AWS Credentials**: Ensure you have valid AWS credentials configured:

   ```bash
   export AWS_ACCESS_KEY_ID="your-access-key"
   export AWS_SECRET_ACCESS_KEY="your-secret-key"
   export AWS_REGION="us-east-1"  # or your preferred region
   ```

   Or use AWS profiles:

   ```bash
   export AWS_PROFILE="your-profile"
   ```

2. **Phoenix Setup**: Configure Phoenix for trace collection:

   ```bash
   export PHOENIX_ENDPOINT="https://your-phoenix-instance.com/v1/traces"
   export PHOENIX_API_KEY="your-phoenix-api-key"
   ```

3. **Bedrock Access**: Ensure your AWS account has access to Bedrock models and the required model ID is available in your region.

#### Usage

```bash
# Run all validation scenarios
npm run validate:invoke-model

# Or use tsx directly
npx tsx scripts/validate-invoke-model.ts

# Run specific scenario
npx tsx scripts/validate-invoke-model.ts --scenario basic-text

# Test with different model
npx tsx scripts/validate-invoke-model.ts --model-id anthropic.claude-3-sonnet-20240229-v1:0

# Enable debug logging
npx tsx scripts/validate-invoke-model.ts --debug

# Use console output only (no Phoenix export)
npx tsx scripts/validate-invoke-model.ts --phoenix-endpoint console
```

#### Test Scenarios

The script replicates the test scenarios from the instrumentation test suite:

1. **basic-text**: Simple text message exchange
2. **tool-calling**: Function calling with tool definitions
3. **multi-modal**: Text + image content processing
4. **tool-results**: Tool result processing in multi-turn conversations
5. **multiple-tools**: Multiple tool definitions in single request
6. **streaming-basic**: Basic streaming response handling
7. **streaming-tools**: Streaming responses with tool calls
8. **streaming-errors**: Error handling in streaming scenarios
9. **context-attributes**: OpenInference context attribute propagation
10. **all**: Run all scenarios (default)

#### Output

The script provides detailed output including:

- Instrumentation verification status
- Test scenario results
- Response summaries
- Error details (if any)
- Trace export confirmation

---

## Converse API Validation

### `validate-converse.ts`

The Converse API validation script tests the modern Bedrock Converse API instrumentation through a single comprehensive conversation that combines multiple test scenarios.

#### Usage

```bash
# Run comprehensive Converse API validation
npm run validate:converse

# Or use tsx directly
npx tsx scripts/validate-converse.ts

# Test with different model
npx tsx scripts/validate-converse.ts --model-id anthropic.claude-3-5-sonnet-20240620-v1:0

# Enable debug logging
npx tsx scripts/validate-converse.ts --debug

# Use console output only (no Phoenix export)
npx tsx scripts/validate-converse.ts --phoenix-endpoint console
```

#### Comprehensive Test Scenario

The script executes a single realistic conversation that tests:

1. **System Prompt Integration**: Multiple system prompts that are concatenated
2. **Multi-Turn Conversation**: 3-turn conversation with full history
3. **Multi-Modal Content**: Text + image content processing
4. **Tool Configuration**: 3 available tools (data analysis, visualization, web search)
5. **Inference Configuration**: maxTokens, temperature, topP, stopSequences
6. **OpenInference Context**: Session, user, metadata, tags, prompt template attributes
7. **Complex Message Structure**: Detailed content structure with proper indexing
8. **Response Processing**: Tool call detection and response validation

#### Expected Output

The validation script provides comprehensive feedback on:

- ✅ Instrumentation verification and setup
- 📊 Comprehensive conversation execution
- 🔧 Tool call detection and analysis
- 💬 Response content validation
- 📈 Token usage statistics
- 📋 Complete test scenario coverage confirmation
- ⏳ Trace export to Phoenix

This single conversation scenario is designed to match the complexity and coverage of multiple unit tests, providing a realistic validation of the Converse API instrumentation in a production-like environment.


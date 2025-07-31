# Bedrock Instrumentation Validation Scripts

This directory contains validation scripts for testing the Bedrock instrumentation in real-world scenarios.

## Scripts

### `validate-invoke-model.ts`

Tests the Bedrock InvokeModel API instrumentation by making actual API calls and verifying traces are created correctly.

### `validate-converse-comprehensive.ts`

Tests the Bedrock Converse API instrumentation with 6 comprehensive scenarios that cover all major test cases: basic flow, multi-turn conversation, tool calling, multi-modal content, context attributes, and error handling.

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

## Converse API Comprehensive Validation

### `validate-converse-comprehensive.ts`

The Converse API comprehensive validation script tests the modern Bedrock Converse API instrumentation through 6 distinct scenarios that cover all major test cases and send separate spans to Phoenix for detailed analysis.

#### Usage

```bash
# Run all 6 comprehensive scenarios
npm run validate:converse-comprehensive

# Or use tsx directly
npx tsx scripts/validate-converse-comprehensive.ts

# Run specific scenario
npx tsx scripts/validate-converse-comprehensive.ts --scenario tool-calling

# Test with different model
npx tsx scripts/validate-converse-comprehensive.ts --model-id anthropic.claude-3-5-sonnet-20240620-v1:0

# Enable debug logging
npx tsx scripts/validate-converse-comprehensive.ts --debug

# Use console output only (no Phoenix export)
npx tsx scripts/validate-converse-comprehensive.ts --phoenix-endpoint console
```

#### Six Comprehensive Test Scenarios

The script executes 6 separate scenarios, each creating its own span in Phoenix:

1. **basic-flow**: Simple conversation with system prompt and inference configuration
2. **multi-turn**: Complex conversation history with multiple turns
3. **tool-calling**: Tool configuration and usage with multiple tools (weather + calculator)
4. **multi-modal**: Text + image content using PNG data
5. **context-attributes**: Comprehensive OpenInference context propagation (session, user, metadata, tags, prompt template)
6. **error-case**: API error handling with invalid model ID

#### Available Scenarios

Run individual scenarios with `--scenario <name>`:

- `basic-flow` - Simple conversation with system prompt
- `multi-turn` - Complex conversation history
- `tool-calling` - Tool configuration and usage
- `multi-modal` - Text + image content
- `context-attributes` - OpenInference context propagation
- `error-case` - API error handling
- `all` - Run all scenarios (default)

#### Expected Output

The validation script provides comprehensive feedback on:

- ✅ Instrumentation verification and setup
- 📊 Six distinct scenario executions
- 🔧 Tool call detection and analysis
- 🖼️ Multi-modal content processing
- 💬 Response content validation
- 📈 Token usage statistics for each scenario
- 📋 Complete test scenario coverage confirmation
- ⏳ Trace export to Phoenix (6 separate spans)

This comprehensive validation approach provides detailed testing of the Converse API instrumentation across all major use cases, with each scenario creating a separate trace in Phoenix for granular analysis and debugging.

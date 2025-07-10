# Bedrock Instrumentation Validation Scripts

This directory contains validation scripts for testing the Bedrock instrumentation in real-world scenarios.

## Scripts

### `validate-invoke-model.ts`

Tests the Bedrock InvokeModel API instrumentation by making actual API calls and verifying traces are created correctly.

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
6. **all**: Run all scenarios (default)

#### What It Validates

- ✅ Instrumentation is properly applied to the Bedrock client
- ✅ Spans are created for InvokeModel API calls
- ✅ Request attributes are correctly extracted
- ✅ Response attributes are correctly extracted
- ✅ Tool calling scenarios work properly
- ✅ Multi-modal content is processed correctly
- ✅ Traces are exported to Phoenix (when configured)

#### Output

The script provides detailed output including:
- Instrumentation verification status
- Test scenario results
- Response summaries
- Error details (if any)
- Trace export confirmation

#### Troubleshooting

1. **Instrumentation not patched**: Make sure the instrumentation is registered before creating the Bedrock client
2. **AWS credentials**: Verify your AWS credentials and region are correct
3. **Model access**: Ensure you have access to the specified Bedrock model in your region
4. **Phoenix connection**: Check your Phoenix endpoint and API key configuration

#### Local Development

When working on the instrumentation locally, this script helps verify:
- Changes are working correctly
- Local package linking is functioning
- All test scenarios pass in real-world conditions
- Traces appear correctly in Phoenix for manual verification

This is especially useful since pnpm workspaces automatically link local packages, making it easy to test changes without publishing.
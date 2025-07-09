# Enhanced VCR Tooling for Seamless TDD Development

## Current State Assessment
The existing VCR system works but needs tooling improvements for efficient TDD workflows. The current recording shows a successful basic Bedrock API call, but we need tools to:

1. **Easily record new API calls** as we write tests
2. **Manage multiple recording scenarios** (basic, tool calls, streaming, etc.)
3. **Validate AWS credentials** before attempting recordings
4. **Isolate test recordings** to prevent conflicts

## Required Tooling Additions

### 1. Recording Helper Scripts
Create npm scripts and helper functions to:
- **Record Mode Toggle**: Easy way to switch between record/replay modes
- **Credential Validation**: Check AWS env vars before recording
- **Recording Cleanup**: Clear old recordings and record fresh ones
- **Targeted Recording**: Record specific test scenarios

### 2. Enhanced Test Setup
Improve the test infrastructure to:
- **Per-Test Recordings**: Separate recording files for different test scenarios
- **Dynamic Recording Names**: Generate recording names based on test descriptions
- **AWS Credential Detection**: Automatically switch to recording mode when AWS creds available
- **Recording Verification**: Validate recorded responses contain expected data

### 3. Tool Call Recording Support
Add specialized recording for:
- **Tool-Enabled Requests**: Record API calls that include tool definitions
- **Tool Response Parsing**: Capture tool call responses with proper message attribution
- **Multi-Turn Conversations**: Record complex agent interactions
- **Streaming Tool Calls**: Handle streaming responses with tool invocations

### 4. Test Data Management
Create utilities for:
- **Test Data Generation**: Generate varied test inputs for comprehensive coverage
- **Response Validation**: Verify recorded responses match expected formats
- **Mock Data Sanitization**: Ensure no sensitive data in recordings
- **Recording Regeneration**: Easy way to refresh stale recordings

## Implementation Strategy

### Phase 1: Core Recording Tools
1. **Package.json Scripts**: Add recording management commands
2. **Environment Detection**: Auto-detect AWS credentials and switch modes
3. **Recording Helper Functions**: Utilities for test setup and cleanup
4. **Directory Structure**: Organize recordings by test suite and scenario

#### Planned NPM Scripts
```json
{
  "test:record": "node scripts/record-helper.js && jest .",
  "test:record:basic": "BEDROCK_RECORD_MODE=basic npm run test:record",
  "test:record:tools": "BEDROCK_RECORD_MODE=tools npm run test:record",
  "test:record:streaming": "BEDROCK_RECORD_MODE=streaming npm run test:record",
  "test:clear-recordings": "node scripts/clear-recordings.js",
  "test:validate-recordings": "node scripts/validate-recordings.js"
}
```

### Phase 2: Enhanced Test Infrastructure  
1. **Dynamic Recording Names**: Generate unique names per test
2. **Recording Validation**: Verify recorded data integrity
3. **Credential Management**: Safe handling of AWS credentials in tests
4. **Error Handling**: Graceful fallbacks when recording fails

### Phase 3: Advanced Recording Features
1. **Tool Call Recording**: Specialized support for tool-enabled requests
2. **Streaming Support**: Record and replay streaming responses
3. **Multi-Request Scenarios**: Handle complex agent workflows
4. **Performance Optimization**: Efficient recording storage and retrieval

## Detailed Implementation Plan

### 1. Recording Helper Script (`scripts/record-helper.js`)
```javascript
// Validates AWS credentials
// Sets up recording environment
// Provides feedback to developer
// Handles credential sanitization
```

### 2. Test Infrastructure Updates
- Modify existing test setup to support dynamic recording modes
- Add per-test recording isolation
- Implement automatic credential detection
- Add recording validation

### 3. Directory Structure
```
test/
├── recordings/
│   ├── basic/
│   │   ├── invoke-model-basic.json
│   │   └── invoke-model-error.json
│   ├── tools/
│   │   ├── invoke-model-with-tools.json
│   │   └── tool-calling-conversation.json
│   └── streaming/
│       ├── invoke-model-stream.json
│       └── converse-stream.json
├── helpers/
│   ├── recording-utils.js
│   └── test-data-generators.js
└── instrumentation.test.ts
```

### 4. Recording Management Utilities
- **Clear Recordings**: Remove old/stale recordings
- **Validate Recordings**: Ensure recordings have proper structure
- **Generate Test Data**: Create varied inputs for comprehensive testing
- **Sanitize Credentials**: Remove sensitive data from recordings

## Success Criteria

### Developer Experience
- Single command to record new API calls (`npm run test:record`)
- Automatic credential detection and mode switching
- Clear feedback when recordings are created/updated
- Easy test isolation without recording conflicts

### Test Reliability
- Deterministic test results with proper mocking
- Comprehensive coverage of Bedrock API scenarios
- Proper tool call and streaming response handling
- Sanitized recordings with no sensitive data

### TDD Workflow
- Write failing test → Record real API call → Implement instrumentation → Verify test passes
- Easy regeneration of recordings when API changes
- Support for incremental test development

## Current Progress

### ✅ Completed - Full Implementation
- **Recording Helper Script**: `scripts/record-helper.js` with AWS credential validation, environment detection, and user feedback
- **NPM Scripts Integration**: Complete suite of recording management commands
- **Recording Validation**: `scripts/validate-recordings.js` with security scanning and structure validation
- **Recording Cleanup**: `scripts/clear-recordings.js` with interactive and command-line modes
- **Test Infrastructure**: Recording utilities, test data generators, and Jest integration helpers
- **Directory Structure**: Organized recording categories (basic, tools, streaming, errors, multimodal, converse, agent)

### 🎯 Key Features Implemented
- **Security by Default**: Automatic credential sanitization and sensitive data detection
- **Developer Experience**: One-command recording (`npm run test:record`) with clear feedback
- **Test Reliability**: Deterministic VCR testing with proper isolation
- **TDD Workflow**: Seamless red-green-refactor cycle support
- **Comprehensive Tooling**: Validation, cleanup, and test data generation utilities

### 🚀 Ready for Implementation
The VCR tooling is complete and ready to support the TDD implementation of JavaScript Bedrock instrumentation. All success criteria met:
- ✅ Single command recording with credential detection
- ✅ Automatic mode switching and clear feedback
- ✅ Test isolation and deterministic results
- ✅ Sanitized recordings with security validation
- ✅ Organized structure for incremental test development

## Notes

- Current VCR system uses `nock.recorder.rec()` effectively
- Existing recording shows proper credential sanitization
- Test setup follows OpenAI instrumentation patterns
- Need to maintain Python parity for tool call support
- Focus on developer experience and test reliability

This plan provides a comprehensive approach to enhancing the VCR tooling for efficient TDD development of the JavaScript Bedrock instrumentation.
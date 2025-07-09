/**
 * Jest Setup Helper for AWS Bedrock Instrumentation Tests
 * 
 * This module provides Jest-specific utilities for setting up VCR recording
 * with proper test isolation and cleanup.
 */

const { validateRecordingEnvironment } = require('./recording-utils');

/**
 * Global setup that runs before all tests
 */
function setupGlobalEnvironment() {
  // Validate recording environment
  const validation = validateRecordingEnvironment();
  
  if (!validation.valid && validation.mode === 'record') {
    console.warn('⚠️  Recording environment issues detected:');
    validation.issues.forEach(issue => {
      console.warn(`   - ${issue}`);
    });
    console.warn('   Tests may fail in recording mode.');
  }
  
  // Set default timeout for tests that make API calls
  jest.setTimeout(30000);
  
  // Configure AWS SDK to use specific region
  if (!process.env.AWS_REGION) {
    process.env.AWS_REGION = 'us-east-1';
  }
}

/**
 * Creates a describe block with automatic recording setup
 */
function describeWithRecording(description, category, testFn) {
  describe(description, () => {
    const { createJestRecordingHelpers } = require('./recording-utils');
    const { beforeEach: setupRecording, afterEach: cleanupRecording } = createJestRecordingHelpers(category);
    
    let currentTestName = '';
    
    beforeEach(() => {
      // Extract test name from Jest context
      currentTestName = expect.getState().currentTestName || 'unknown-test';
      setupRecording(currentTestName);
    });
    
    afterEach(() => {
      cleanupRecording();
    });
    
    testFn();
  });
}

/**
 * Creates a test function with automatic recording
 */
function testWithRecording(description, category, testFn) {
  return test(description, async () => {
    const { withRecording } = require('./recording-utils');
    return withRecording(description, testFn, { category })();
  });
}

/**
 * Skips test if not in recording mode and no recording exists
 */
function testRequiresRecording(description, category, testFn) {
  const { getRecordingMode, generateRecordingFilename } = require('./recording-utils');
  const path = require('path');
  const fs = require('fs');
  
  const mode = getRecordingMode();
  const recordingFile = generateRecordingFilename(description, category);
  const recordingPath = path.join(__dirname, '..', 'recordings', recordingFile);
  
  if (mode !== 'record' && !fs.existsSync(recordingPath)) {
    test.skip(`${description} (no recording available)`, () => {});
    return;
  }
  
  testWithRecording(description, category, testFn);
}

/**
 * Creates parameterized tests with recordings
 */
function testCases(baseDescription, category, testCases, testFn) {
  describe(baseDescription, () => {
    testCases.forEach((testCase, index) => {
      const description = testCase.description || `case ${index + 1}`;
      testWithRecording(`${baseDescription} - ${description}`, category, () => {
        return testFn(testCase);
      });
    });
  });
}

/**
 * Helper for testing streaming responses
 */
function testStreamingWithRecording(description, category, testFn) {
  return testWithRecording(description, `streaming/${category}`, async () => {
    // Set longer timeout for streaming tests
    jest.setTimeout(60000);
    return testFn();
  });
}

/**
 * Helper for testing error scenarios
 */
function testErrorWithRecording(description, testFn) {
  return testWithRecording(description, 'errors', testFn);
}

/**
 * Utility to wait for async operations in streaming tests
 */
function waitForStreamCompletion(stream, timeout = 30000) {
  return new Promise((resolve, reject) => {
    const timeoutId = setTimeout(() => {
      reject(new Error('Stream timeout'));
    }, timeout);
    
    let chunks = [];
    
    stream.on('data', (chunk) => {
      chunks.push(chunk);
    });
    
    stream.on('end', () => {
      clearTimeout(timeoutId);
      resolve(chunks);
    });
    
    stream.on('error', (error) => {
      clearTimeout(timeoutId);
      reject(error);
    });
  });
}

/**
 * Common test patterns for Bedrock instrumentation
 */
const testPatterns = {
  /**
   * Tests basic span creation and attributes
   */
  basicSpanValidation: (spans, expectedAttributes = {}) => {
    expect(spans).toHaveLength(1);
    
    const span = spans[0];
    expect(span.name).toBeDefined();
    expect(span.attributes).toBeDefined();
    expect(span.status.code).toBe('OK');
    
    // Check for required OpenInference attributes
    expect(span.attributes['llm.request.type']).toBeDefined();
    expect(span.attributes['llm.model_name']).toBeDefined();
    
    // Check custom attributes
    Object.entries(expectedAttributes).forEach(([key, value]) => {
      expect(span.attributes[key]).toBe(value);
    });
  },
  
  /**
   * Tests token counting attributes
   */
  tokenCountValidation: (spans) => {
    const span = spans[0];
    expect(span.attributes['llm.usage.input_tokens']).toBeGreaterThan(0);
    expect(span.attributes['llm.usage.output_tokens']).toBeGreaterThan(0);
    expect(span.attributes['llm.usage.total_tokens']).toBeGreaterThan(0);
  },
  
  /**
   * Tests tool call attributes
   */
  toolCallValidation: (spans, expectedToolName) => {
    const span = spans[0];
    expect(span.attributes['llm.input.tool_calls']).toBeDefined();
    
    const toolCalls = JSON.parse(span.attributes['llm.input.tool_calls']);
    expect(Array.isArray(toolCalls)).toBe(true);
    expect(toolCalls.length).toBeGreaterThan(0);
    
    if (expectedToolName) {
      expect(toolCalls[0].function.name).toBe(expectedToolName);
    }
  },
  
  /**
   * Tests error handling
   */
  errorHandlingValidation: (spans, expectedErrorType) => {
    expect(spans).toHaveLength(1);
    
    const span = spans[0];
    expect(span.status.code).toBe('ERROR');
    expect(span.events).toBeDefined();
    
    const errorEvent = span.events.find(event => event.name === 'exception');
    expect(errorEvent).toBeDefined();
    
    if (expectedErrorType) {
      expect(errorEvent.attributes['exception.type']).toBe(expectedErrorType);
    }
  }
};

module.exports = {
  setupGlobalEnvironment,
  describeWithRecording,
  testWithRecording,
  testRequiresRecording,
  testCases,
  testStreamingWithRecording,
  testErrorWithRecording,
  waitForStreamCompletion,
  testPatterns
};
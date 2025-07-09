/**
 * Recording Utilities for AWS Bedrock Instrumentation Tests
 * 
 * This module provides utilities for managing VCR recordings with isolation
 * between tests and dynamic recording names based on test descriptions.
 */

const nock = require('nock');
const path = require('path');
const fs = require('fs');

/**
 * Configuration for recording utilities
 */
const config = {
  recordingsDir: path.join(__dirname, '..', 'recordings'),
  defaultMode: process.env.NOCK_RECORD_MODE || 'replay',
  sanitizeHeaders: true,
  enableReqHeaderRecording: false,
  enableReqBodyRecording: true
};

/**
 * Generates a safe filename from test description
 */
function generateRecordingFilename(testDescription, category = 'basic') {
  // Convert test description to a safe filename
  const safeDescription = testDescription
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, '') // Remove special characters
    .replace(/\s+/g, '-') // Replace spaces with hyphens
    .replace(/-+/g, '-') // Collapse multiple hyphens
    .replace(/^-|-$/g, ''); // Remove leading/trailing hyphens
  
  return `${category}/${safeDescription}.json`;
}

/**
 * Gets the recording mode for the current test environment
 */
function getRecordingMode() {
  // Check for test-specific recording mode
  const testMode = process.env.BEDROCK_RECORD_MODE;
  if (testMode && ['record', 'replay', 'lockdown'].includes(testMode)) {
    return testMode;
  }
  
  // Check for general nock recording mode
  const nockMode = process.env.NOCK_RECORD_MODE;
  if (nockMode && ['record', 'replay', 'lockdown'].includes(nockMode)) {
    return nockMode;
  }
  
  // Check if AWS credentials are available for auto-recording
  const hasCredentials = process.env.AWS_ACCESS_KEY_ID && process.env.AWS_SECRET_ACCESS_KEY;
  if (hasCredentials && !nockMode) {
    console.log('📹 AWS credentials detected, enabling recording mode');
    return 'record';
  }
  
  return config.defaultMode;
}

/**
 * Sanitizes request headers to remove sensitive information
 */
function sanitizeHeaders(headers) {
  const sanitized = { ...headers };
  
  // Remove or mask sensitive headers
  const sensitiveHeaders = [
    'authorization',
    'x-amz-security-token',
    'x-amz-date',
    'x-amz-content-sha256'
  ];
  
  sensitiveHeaders.forEach(header => {
    if (sanitized[header]) {
      sanitized[header] = '[REDACTED]';
    }
  });
  
  return sanitized;
}

/**
 * Sanitizes request/response bodies to remove sensitive information
 */
function sanitizeBody(body) {
  if (typeof body === 'string') {
    try {
      const parsed = JSON.parse(body);
      return JSON.stringify(sanitizeObject(parsed), null, 2);
    } catch (e) {
      // If not JSON, return as-is (might be binary or other format)
      return body;
    }
  } else if (typeof body === 'object') {
    return sanitizeObject(body);
  }
  
  return body;
}

/**
 * Recursively sanitizes object properties
 */
function sanitizeObject(obj) {
  if (Array.isArray(obj)) {
    return obj.map(item => sanitizeObject(item));
  } else if (obj && typeof obj === 'object') {
    const sanitized = {};
    for (const [key, value] of Object.entries(obj)) {
      // Don't sanitize response content, only request data
      if (key.toLowerCase().includes('token') || 
          key.toLowerCase().includes('key') ||
          key.toLowerCase().includes('secret')) {
        sanitized[key] = '[REDACTED]';
      } else {
        sanitized[key] = sanitizeObject(value);
      }
    }
    return sanitized;
  }
  
  return obj;
}

/**
 * Sets up nock recording for a specific test
 */
function setupRecording(testDescription, options = {}) {
  const {
    category = 'basic',
    mode = getRecordingMode(),
    sanitize = config.sanitizeHeaders
  } = options;
  
  const recordingFile = generateRecordingFilename(testDescription, category);
  const recordingPath = path.join(config.recordingsDir, recordingFile);
  
  // Ensure directory exists
  const recordingDir = path.dirname(recordingPath);
  if (!fs.existsSync(recordingDir)) {
    fs.mkdirSync(recordingDir, { recursive: true });
  }
  
  // Configure nock based on mode
  switch (mode) {
    case 'record':
      console.log(`📹 Recording API calls to: ${recordingFile}`);
      nock.recorder.rec({
        dont_print: true,
        output_objects: true,
        enable_reqheaders_recording: config.enableReqHeaderRecording,
        enable_reqbody_recording: config.enableReqBodyRecording
      });
      break;
      
    case 'replay':
      if (fs.existsSync(recordingPath)) {
        console.log(`📼 Using recorded responses from: ${recordingFile}`);
        const recordings = JSON.parse(fs.readFileSync(recordingPath, 'utf8'));
        nock.load(recordingPath);
      } else {
        console.warn(`⚠️  Recording file not found: ${recordingFile}`);
        console.warn('   Tests may fail. Run with BEDROCK_RECORD_MODE=record to create recordings.');
      }
      break;
      
    case 'lockdown':
      console.log(`🔒 Lockdown mode: no network calls allowed`);
      nock.disableNetConnect();
      break;
  }
  
  return {
    recordingFile,
    recordingPath,
    mode,
    
    // Cleanup function to be called after test
    cleanup: () => {
      if (mode === 'record') {
        const recordings = nock.recorder.play();
        
        if (recordings.length > 0) {
          // Sanitize recordings if enabled
          const sanitizedRecordings = sanitize ? 
            recordings.map(recording => ({
              ...recording,
              reqheaders: sanitizeHeaders(recording.reqheaders || {}),
              body: sanitizeBody(recording.body)
            })) : recordings;
          
          // Save recordings to file
          fs.writeFileSync(recordingPath, JSON.stringify(sanitizedRecordings, null, 2));
          console.log(`💾 Saved ${recordings.length} recordings to: ${recordingFile}`);
        }
        
        nock.recorder.clear();
      }
      
      // Clean up any remaining nock interceptors
      nock.cleanAll();
      nock.restore();
      
      if (mode === 'lockdown') {
        nock.enableNetConnect();
      }
    }
  };
}

/**
 * Jest helper for setting up recording in beforeEach/afterEach
 */
function createJestRecordingHelpers(category = 'basic') {
  let recordingSession = null;
  
  const beforeEach = (testDescription, options = {}) => {
    recordingSession = setupRecording(testDescription, { 
      category, 
      ...options 
    });
  };
  
  const afterEach = () => {
    if (recordingSession) {
      recordingSession.cleanup();
      recordingSession = null;
    }
  };
  
  return { beforeEach, afterEach };
}

/**
 * High-level test wrapper that automatically handles recording
 */
function withRecording(testDescription, testFn, options = {}) {
  return async () => {
    const session = setupRecording(testDescription, options);
    
    try {
      await testFn();
    } finally {
      session.cleanup();
    }
  };
}

/**
 * Validates that current environment is suitable for recording
 */
function validateRecordingEnvironment() {
  const mode = getRecordingMode();
  const issues = [];
  
  if (mode === 'record') {
    if (!process.env.AWS_ACCESS_KEY_ID) {
      issues.push('AWS_ACCESS_KEY_ID not set');
    }
    if (!process.env.AWS_SECRET_ACCESS_KEY) {
      issues.push('AWS_SECRET_ACCESS_KEY not set');
    }
    if (!process.env.AWS_REGION) {
      issues.push('AWS_REGION not set (will default to us-east-1)');
    }
  }
  
  return {
    mode,
    valid: issues.length === 0,
    issues
  };
}

module.exports = {
  setupRecording,
  createJestRecordingHelpers,
  withRecording,
  generateRecordingFilename,
  getRecordingMode,
  validateRecordingEnvironment,
  sanitizeHeaders,
  sanitizeBody,
  config
};
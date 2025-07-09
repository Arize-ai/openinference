#!/usr/bin/env node

/**
 * Recording Validation Script for AWS Bedrock Instrumentation
 * 
 * This script validates VCR recordings to ensure they have proper structure,
 * no sensitive data, and valid response formats for Bedrock API calls.
 */

const fs = require('fs');
const path = require('path');

// ANSI color codes for terminal output
const colors = {
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  red: '\x1b[31m',
  blue: '\x1b[34m',
  reset: '\x1b[0m',
  bold: '\x1b[1m'
};

/**
 * Validation rules for recordings
 */
const validationRules = {
  // Check for sensitive data patterns
  sensitivePatterns: [
    /AKIA[0-9A-Z]{16}/g, // AWS Access Key ID
    /[A-Za-z0-9/+=]{40}/g, // AWS Secret Access Key (rough pattern)
    /AWS4-HMAC-SHA256/g, // AWS Signature (should be sanitized)
    /"authorization":\s*"[^"]*"/gi, // Authorization headers
    /"x-amz-security-token":\s*"[^"]*"/gi, // Session tokens
  ],
  
  // Required fields for Bedrock recordings
  requiredFields: [
    'scope',
    'method',
    'path',
    'status',
    'response'
  ],
  
  // Expected Bedrock API endpoints
  validEndpoints: [
    /^https:\/\/bedrock-runtime\./,
    /^https:\/\/bedrock-agent-runtime\./,
    /\/model\/.*\/invoke$/,
    /\/model\/.*\/invoke-with-response-stream$/,
    /\/model\/.*\/converse$/,
    /\/model\/.*\/converse-stream$/,
    /\/agent\/.*\/session\/.*\/text$/
  ],
  
  // Valid HTTP methods for Bedrock
  validMethods: ['POST'],
  
  // Valid status codes
  validStatusCodes: [200, 400, 403, 404, 429, 500, 503]
};

/**
 * Validates a single recording file
 */
function validateRecording(filePath) {
  const results = {
    valid: true,
    errors: [],
    warnings: [],
    info: [],
    recordingCount: 0
  };
  
  try {
    const content = fs.readFileSync(filePath, 'utf8');
    let recordings;
    
    try {
      recordings = JSON.parse(content);
    } catch (parseError) {
      results.valid = false;
      results.errors.push(`Invalid JSON format: ${parseError.message}`);
      return results;
    }
    
    if (!Array.isArray(recordings)) {
      results.valid = false;
      results.errors.push('Recording file must contain an array of recordings');
      return results;
    }
    
    results.recordingCount = recordings.length;
    results.info.push(`Found ${recordings.length} recordings`);
    
    recordings.forEach((recording, index) => {
      validateSingleRecording(recording, index, results, filePath);
    });
    
  } catch (error) {
    results.valid = false;
    results.errors.push(`Failed to read file: ${error.message}`);
  }
  
  return results;
}

/**
 * Validates a single recording entry
 */
function validateSingleRecording(recording, index, results, filePath) {
  const recordingPrefix = `Recording ${index + 1}`;
  
  // Check required fields
  validationRules.requiredFields.forEach(field => {
    if (!recording.hasOwnProperty(field)) {
      results.errors.push(`${recordingPrefix}: Missing required field '${field}'`);
      results.valid = false;
    }
  });
  
  // Validate HTTP method
  if (recording.method && !validationRules.validMethods.includes(recording.method)) {
    results.warnings.push(`${recordingPrefix}: Unexpected HTTP method '${recording.method}'`);
  }
  
  // Validate status code
  if (recording.status && !validationRules.validStatusCodes.includes(recording.status)) {
    results.warnings.push(`${recordingPrefix}: Unexpected status code ${recording.status}`);
  }
  
  // Validate endpoint
  if (recording.path) {
    const isValidEndpoint = validationRules.validEndpoints.some(pattern => 
      pattern.test(recording.scope + recording.path) || pattern.test(recording.path)
    );
    if (!isValidEndpoint) {
      results.warnings.push(`${recordingPrefix}: Unexpected endpoint '${recording.path}'`);
    }
  }
  
  // Check for sensitive data
  const recordingString = JSON.stringify(recording);
  validationRules.sensitivePatterns.forEach(pattern => {
    const matches = recordingString.match(pattern);
    if (matches) {
      results.errors.push(`${recordingPrefix}: Found potential sensitive data: ${matches.length} matches for pattern ${pattern}`);
      results.valid = false;
    }
  });
  
  // Validate Bedrock-specific response structure
  if (recording.response) {
    validateBedrockResponse(recording, index, results, filePath);
  }
}

/**
 * Validates Bedrock-specific response structure
 */
function validateBedrockResponse(recording, index, results, filePath) {
  const recordingPrefix = `Recording ${index + 1}`;
  
  try {
    let responseBody;
    
    // Parse response body if it's a string
    if (typeof recording.response === 'string') {
      try {
        responseBody = JSON.parse(recording.response);
      } catch (e) {
        results.warnings.push(`${recordingPrefix}: Response body is not valid JSON`);
        return;
      }
    } else {
      responseBody = recording.response;
    }
    
    // Check for Bedrock-specific response fields based on endpoint
    if (recording.path) {
      if (recording.path.includes('/invoke') && !recording.path.includes('stream')) {
        // InvokeModel response validation
        validateInvokeModelResponse(responseBody, recordingPrefix, results);
      } else if (recording.path.includes('/converse')) {
        // Converse response validation
        validateConverseResponse(responseBody, recordingPrefix, results);
      } else if (recording.path.includes('/agent/')) {
        // Agent response validation
        validateAgentResponse(responseBody, recordingPrefix, results);
      }
    }
    
  } catch (error) {
    results.warnings.push(`${recordingPrefix}: Error validating response structure: ${error.message}`);
  }
}

/**
 * Validates InvokeModel API response structure
 */
function validateInvokeModelResponse(responseBody, prefix, results) {
  // Check for typical Anthropic response structure
  if (responseBody.content) {
    results.info.push(`${prefix}: Found Anthropic Messages API response`);
    
    if (Array.isArray(responseBody.content)) {
      responseBody.content.forEach((block, i) => {
        if (!block.type) {
          results.warnings.push(`${prefix}: Content block ${i} missing 'type' field`);
        }
      });
    }
  }
  
  // Check for usage information
  if (responseBody.usage) {
    results.info.push(`${prefix}: Found token usage information`);
    if (!responseBody.usage.input_tokens || !responseBody.usage.output_tokens) {
      results.warnings.push(`${prefix}: Usage information incomplete`);
    }
  }
  
  // Check for tool use
  if (responseBody.content && Array.isArray(responseBody.content)) {
    const hasToolUse = responseBody.content.some(block => block.type === 'tool_use');
    if (hasToolUse) {
      results.info.push(`${prefix}: Contains tool use response`);
    }
  }
}

/**
 * Validates Converse API response structure
 */
function validateConverseResponse(responseBody, prefix, results) {
  if (responseBody.output) {
    results.info.push(`${prefix}: Found Converse API response`);
    
    if (responseBody.output.message && responseBody.output.message.content) {
      const content = responseBody.output.message.content;
      if (Array.isArray(content)) {
        results.info.push(`${prefix}: Contains ${content.length} content blocks`);
      }
    }
  }
  
  // Check for usage
  if (responseBody.usage) {
    results.info.push(`${prefix}: Found Converse usage information`);
  }
  
  // Check for stop reason
  if (responseBody.stopReason) {
    results.info.push(`${prefix}: Stop reason: ${responseBody.stopReason}`);
  }
}

/**
 * Validates Agent API response structure
 */
function validateAgentResponse(responseBody, prefix, results) {
  // Agent responses are typically streaming, so structure varies
  if (responseBody.chunk) {
    results.info.push(`${prefix}: Found Agent streaming chunk`);
  }
  
  if (responseBody.trace) {
    results.info.push(`${prefix}: Contains agent trace information`);
  }
}

/**
 * Scans all recording files in the recordings directory
 */
function scanAllRecordings() {
  const recordingsDir = path.join(__dirname, '..', 'test', 'recordings');
  const results = {
    totalFiles: 0,
    validFiles: 0,
    invalidFiles: 0,
    totalRecordings: 0,
    errors: [],
    warnings: [],
    info: []
  };
  
  if (!fs.existsSync(recordingsDir)) {
    results.errors.push('Recordings directory does not exist');
    return results;
  }
  
  function scanDirectory(dir, relativePath = '') {
    const items = fs.readdirSync(dir);
    
    items.forEach(item => {
      const fullPath = path.join(dir, item);
      const relativeItemPath = path.join(relativePath, item);
      const stat = fs.statSync(fullPath);
      
      if (stat.isDirectory()) {
        scanDirectory(fullPath, relativeItemPath);
      } else if (item.endsWith('.json')) {
        results.totalFiles++;
        
        console.log(`\n${colors.blue}📁 Validating: ${relativeItemPath}${colors.reset}`);
        
        const fileResults = validateRecording(fullPath);
        results.totalRecordings += fileResults.recordingCount;
        
        if (fileResults.valid) {
          results.validFiles++;
          console.log(`  ${colors.green}✅ Valid${colors.reset}`);
        } else {
          results.invalidFiles++;
          console.log(`  ${colors.red}❌ Invalid${colors.reset}`);
        }
        
        // Print file-specific results
        fileResults.errors.forEach(error => {
          console.log(`    ${colors.red}❌ ${error}${colors.reset}`);
          results.errors.push(`${relativeItemPath}: ${error}`);
        });
        
        fileResults.warnings.forEach(warning => {
          console.log(`    ${colors.yellow}⚠️  ${warning}${colors.reset}`);
          results.warnings.push(`${relativeItemPath}: ${warning}`);
        });
        
        fileResults.info.forEach(info => {
          console.log(`    ${colors.blue}ℹ️  ${info}${colors.reset}`);
        });
      }
    });
  }
  
  scanDirectory(recordingsDir);
  return results;
}

/**
 * Prints summary of validation results
 */
function printSummary(results) {
  console.log(`\n${colors.bold}${colors.blue}📊 Validation Summary${colors.reset}\n`);
  
  console.log(`${colors.bold}Files:${colors.reset}`);
  console.log(`  Total files: ${results.totalFiles}`);
  console.log(`  Valid files: ${colors.green}${results.validFiles}${colors.reset}`);
  console.log(`  Invalid files: ${colors.red}${results.invalidFiles}${colors.reset}`);
  console.log(`  Total recordings: ${results.totalRecordings}`);
  
  console.log(`\n${colors.bold}Issues:${colors.reset}`);
  console.log(`  Errors: ${colors.red}${results.errors.length}${colors.reset}`);
  console.log(`  Warnings: ${colors.yellow}${results.warnings.length}${colors.reset}`);
  
  if (results.errors.length === 0 && results.warnings.length === 0) {
    console.log(`\n${colors.green}🎉 All recordings are valid!${colors.reset}`);
  } else {
    if (results.errors.length > 0) {
      console.log(`\n${colors.red}❌ Critical Issues:${colors.reset}`);
      console.log('   Please fix these errors before using recordings:');
      results.errors.slice(0, 5).forEach(error => {
        console.log(`   - ${error}`);
      });
      if (results.errors.length > 5) {
        console.log(`   ... and ${results.errors.length - 5} more errors`);
      }
    }
    
    if (results.warnings.length > 0) {
      console.log(`\n${colors.yellow}⚠️  Warnings:${colors.reset}`);
      console.log('   Consider reviewing these issues:');
      results.warnings.slice(0, 5).forEach(warning => {
        console.log(`   - ${warning}`);
      });
      if (results.warnings.length > 5) {
        console.log(`   ... and ${results.warnings.length - 5} more warnings`);
      }
    }
  }
  
  console.log(); // Empty line for spacing
}

/**
 * Main execution
 */
function main() {
  console.log(`${colors.bold}${colors.blue}🔍 Recording Validation Tool${colors.reset}\n`);
  
  try {
    const results = scanAllRecordings();
    printSummary(results);
    
    // Exit with appropriate code
    if (results.invalidFiles > 0) {
      process.exit(1);
    } else if (results.warnings.length > 0) {
      process.exit(0); // Warnings don't cause failure
    } else {
      process.exit(0);
    }
    
  } catch (error) {
    console.error(`${colors.red}❌ Error: ${error.message}${colors.reset}`);
    process.exit(1);
  }
}

// Run if called directly
if (require.main === module) {
  main();
}

module.exports = {
  validateRecording,
  scanAllRecordings,
  validationRules
};
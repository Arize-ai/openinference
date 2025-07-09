#!/usr/bin/env node

/**
 * Recording Helper Script for AWS Bedrock Instrumentation
 *
 * This script validates AWS credentials and sets up the recording environment
 * for VCR-style testing with nock. It provides feedback to developers about
 * recording status and handles credential sanitization.
 */

const fs = require("fs");
const path = require("path");

// ANSI color codes for terminal output
const colors = {
  green: "\x1b[32m",
  yellow: "\x1b[33m",
  red: "\x1b[31m",
  blue: "\x1b[34m",
  reset: "\x1b[0m",
  bold: "\x1b[1m",
};

/**
 * Checks if AWS credentials are available in environment variables
 */
function checkAWSCredentials() {
  const requiredVars = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"];
  const optionalVars = ["AWS_REGION", "AWS_SESSION_TOKEN"];

  const missing = requiredVars.filter((varName) => !process.env[varName]);
  const present = requiredVars.filter((varName) => process.env[varName]);
  const optional = optionalVars.filter((varName) => process.env[varName]);

  return {
    hasCredentials: missing.length === 0,
    missing,
    present,
    optional,
  };
}

/**
 * Validates AWS credential format (basic validation)
 */
function validateCredentialFormat() {
  const accessKeyId = process.env.AWS_ACCESS_KEY_ID;
  const secretAccessKey = process.env.AWS_SECRET_ACCESS_KEY;

  if (!accessKeyId || !secretAccessKey) {
    return { valid: false, issues: ["Missing credentials"] };
  }

  const issues = [];

  // Basic format validation
  if (
    !/^AKIA[0-9A-Z]{16}$/.test(accessKeyId) &&
    !/^ASIA[0-9A-Z]{16}$/.test(accessKeyId)
  ) {
    issues.push(
      "AWS_ACCESS_KEY_ID format may be invalid (expected AKIA... or ASIA...)",
    );
  }

  if (secretAccessKey.length !== 40) {
    issues.push(
      "AWS_SECRET_ACCESS_KEY length may be invalid (expected 40 characters)",
    );
  }

  return {
    valid: issues.length === 0,
    issues,
  };
}

/**
 * Ensures recording directories exist
 */
function ensureRecordingDirectories() {
  const baseDir = path.join(__dirname, "..", "test", "recordings");
  const subdirs = [
    "basic",
    "tools",
    "streaming",
    "errors",
    "multimodal",
    "converse",
    "agent",
  ];

  const created = [];

  // Create base recordings directory
  if (!fs.existsSync(baseDir)) {
    fs.mkdirSync(baseDir, { recursive: true });
    created.push("recordings/");
  }

  // Create subdirectories
  subdirs.forEach((subdir) => {
    const fullPath = path.join(baseDir, subdir);
    if (!fs.existsSync(fullPath)) {
      fs.mkdirSync(fullPath, { recursive: true });
      created.push(`recordings/${subdir}/`);
    }
  });

  return created;
}

/**
 * Gets the recording mode from environment variables
 */
function getRecordingMode() {
  const mode =
    process.env.BEDROCK_RECORD_MODE || process.env.NOCK_RECORD_MODE || "replay";
  const validModes = ["record", "replay", "lockdown"];

  if (!validModes.includes(mode)) {
    console.warn(
      `${colors.yellow}⚠️  Invalid recording mode: ${mode}. Using 'replay'${colors.reset}`,
    );
    return "replay";
  }

  return mode;
}

/**
 * Sets up environment variables for recording
 */
function setupRecordingEnvironment() {
  const credentialCheck = checkAWSCredentials();
  const mode = getRecordingMode();

  // Set nock recording mode
  process.env.NOCK_RECORD_MODE = mode;

  // If we have credentials and mode is not explicitly set, enable recording
  if (credentialCheck.hasCredentials && !process.env.BEDROCK_RECORD_MODE) {
    process.env.NOCK_RECORD_MODE = "record";
    process.env.BEDROCK_RECORD_MODE = "record";
  }

  // Set default region if not provided
  if (!process.env.AWS_REGION) {
    process.env.AWS_REGION = "us-east-1";
  }

  return {
    mode: process.env.NOCK_RECORD_MODE,
    region: process.env.AWS_REGION,
  };
}

/**
 * Prints status information to the console
 */
function printStatus() {
  const credentialCheck = checkAWSCredentials();
  const validation = validateCredentialFormat();
  const environment = setupRecordingEnvironment();
  const createdDirs = ensureRecordingDirectories();

  console.log(
    `\n${colors.bold}${colors.blue}🎬 AWS Bedrock Recording Helper${colors.reset}\n`,
  );

  // Credential status
  console.log(`${colors.bold}AWS Credentials:${colors.reset}`);
  if (credentialCheck.hasCredentials) {
    console.log(
      `  ${colors.green}✅ Found required credentials${colors.reset}`,
    );
    credentialCheck.present.forEach((varName) => {
      console.log(`     - ${varName}: ${colors.green}✓${colors.reset}`);
    });

    if (credentialCheck.optional.length > 0) {
      console.log(`  ${colors.blue}ℹ️  Optional credentials:${colors.reset}`);
      credentialCheck.optional.forEach((varName) => {
        console.log(`     - ${varName}: ${colors.blue}✓${colors.reset}`);
      });
    }
  } else {
    console.log(
      `  ${colors.red}❌ Missing required credentials${colors.reset}`,
    );
    credentialCheck.missing.forEach((varName) => {
      console.log(`     - ${varName}: ${colors.red}missing${colors.reset}`);
    });
  }

  // Validation status
  if (credentialCheck.hasCredentials) {
    console.log(`\n${colors.bold}Credential Validation:${colors.reset}`);
    if (validation.valid) {
      console.log(
        `  ${colors.green}✅ Credentials appear valid${colors.reset}`,
      );
    } else {
      console.log(`  ${colors.yellow}⚠️  Potential issues:${colors.reset}`);
      validation.issues.forEach((issue) => {
        console.log(`     - ${colors.yellow}${issue}${colors.reset}`);
      });
    }
  }

  // Recording mode
  console.log(`\n${colors.bold}Recording Mode:${colors.reset}`);
  const modeColor =
    environment.mode === "record"
      ? colors.green
      : environment.mode === "replay"
        ? colors.blue
        : colors.yellow;
  console.log(
    `  ${modeColor}📹 ${environment.mode.toUpperCase()}${colors.reset}`,
  );
  console.log(`  🌍 Region: ${environment.region}`);

  // Directory setup
  if (createdDirs.length > 0) {
    console.log(`\n${colors.bold}Created Directories:${colors.reset}`);
    createdDirs.forEach((dir) => {
      console.log(`  ${colors.green}✅ ${dir}${colors.reset}`);
    });
  }

  // Recording guidance
  console.log(`\n${colors.bold}Recording Guidance:${colors.reset}`);
  if (environment.mode === "record") {
    console.log(
      `  ${colors.green}🎥 Ready to record new API calls${colors.reset}`,
    );
    console.log(`     - Real AWS Bedrock API calls will be made`);
    console.log(`     - Responses will be saved for future replay`);
    console.log(`     - Credentials will be sanitized in recordings`);
  } else if (environment.mode === "replay") {
    console.log(`  ${colors.blue}📼 Using existing recordings${colors.reset}`);
    console.log(`     - No real API calls will be made`);
    console.log(`     - Tests will use saved responses`);
    console.log(`     - Set BEDROCK_RECORD_MODE=record to record new calls`);
  } else {
    console.log(
      `  ${colors.yellow}🔒 Lockdown mode - no network calls allowed${colors.reset}`,
    );
  }

  // Next steps
  console.log(`\n${colors.bold}Next Steps:${colors.reset}`);
  if (!credentialCheck.hasCredentials && environment.mode === "record") {
    console.log(
      `  ${colors.red}1. Set AWS credentials to enable recording${colors.reset}`,
    );
    console.log(`     export AWS_ACCESS_KEY_ID=your_access_key`);
    console.log(`     export AWS_SECRET_ACCESS_KEY=your_secret_key`);
    console.log(`     export AWS_REGION=us-east-1`);
  }
  console.log(`  ${colors.blue}1. Run tests: npm test${colors.reset}`);
  console.log(
    `  ${colors.blue}2. Clear recordings: npm run test:clear-recordings${colors.reset}`,
  );
  console.log(
    `  ${colors.blue}3. Validate recordings: npm run test:validate-recordings${colors.reset}`,
  );

  console.log(); // Empty line for spacing
}

/**
 * Main execution
 */
function main() {
  try {
    printStatus();

    // Exit with appropriate code
    const credentialCheck = checkAWSCredentials();
    const mode = getRecordingMode();

    if (mode === "record" && !credentialCheck.hasCredentials) {
      console.log(
        `${colors.yellow}⚠️  Recording mode requested but credentials missing. Tests will likely fail.${colors.reset}\n`,
      );
      process.exit(1);
    }

    process.exit(0);
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
  checkAWSCredentials,
  validateCredentialFormat,
  ensureRecordingDirectories,
  getRecordingMode,
  setupRecordingEnvironment,
};

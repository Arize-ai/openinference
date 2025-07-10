#!/usr/bin/env node

/**
 * Clear Recordings Script for AWS Bedrock Instrumentation
 *
 * This script safely removes VCR recordings to enable fresh recording
 * of API calls. It provides options for selective deletion and confirms
 * actions before proceeding.
 */

const fs = require("fs");
const path = require("path");
const readline = require("readline");

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
 * Creates readline interface for user interaction
 */
function createInterface() {
  return readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });
}

/**
 * Prompts user for confirmation
 */
function askQuestion(rl, question) {
  return new Promise((resolve) => {
    rl.question(question, (answer) => {
      resolve(answer.toLowerCase().trim());
    });
  });
}

/**
 * Scans recordings directory and returns file information
 */
function scanRecordings() {
  const recordingsDir = path.join(__dirname, "..", "test", "recordings");
  const recordings = {
    directories: [],
    files: [],
    totalSize: 0,
  };

  if (!fs.existsSync(recordingsDir)) {
    return recordings;
  }

  function scanDirectory(dir, relativePath = "") {
    const items = fs.readdirSync(dir);

    items.forEach((item) => {
      const fullPath = path.join(dir, item);
      const relativeItemPath = path.join(relativePath, item);
      const stat = fs.statSync(fullPath);

      if (stat.isDirectory()) {
        recordings.directories.push({
          path: fullPath,
          relativePath: relativeItemPath,
          name: item,
        });
        scanDirectory(fullPath, relativeItemPath);
      } else if (item.endsWith(".json")) {
        recordings.files.push({
          path: fullPath,
          relativePath: relativeItemPath,
          name: item,
          size: stat.size,
          modified: stat.mtime,
        });
        recordings.totalSize += stat.size;
      }
    });
  }

  scanDirectory(recordingsDir);
  return recordings;
}

/**
 * Displays current recordings information
 */
function displayRecordings(recordings) {
  console.log(`${colors.bold}📁 Current Recordings:${colors.reset}\n`);

  if (recordings.files.length === 0) {
    console.log(`  ${colors.yellow}No recording files found${colors.reset}`);
    return;
  }

  // Group files by directory
  const filesByDir = {};
  recordings.files.forEach((file) => {
    const dir = path.dirname(file.relativePath) || ".";
    if (!filesByDir[dir]) {
      filesByDir[dir] = [];
    }
    filesByDir[dir].push(file);
  });

  // Display files by directory
  Object.keys(filesByDir)
    .sort()
    .forEach((dir) => {
      console.log(`  ${colors.blue}📂 ${dir}/${colors.reset}`);
      filesByDir[dir]
        .sort((a, b) => a.name.localeCompare(b.name))
        .forEach((file) => {
          const sizeKB = (file.size / 1024).toFixed(1);
          const modifiedDate = file.modified.toLocaleDateString();
          console.log(`    📄 ${file.name} (${sizeKB} KB, ${modifiedDate})`);
        });
      console.log();
    });

  const totalSizeKB = (recordings.totalSize / 1024).toFixed(1);
  console.log(
    `  ${colors.bold}Total: ${recordings.files.length} files, ${totalSizeKB} KB${colors.reset}\n`,
  );
}

/**
 * Removes recording files based on options
 */
function removeRecordings(recordings, options = {}) {
  const { directory, confirm = true } = options;
  let filesToRemove = recordings.files;

  // Filter by directory if specified
  if (directory) {
    filesToRemove = recordings.files.filter((file) =>
      file.relativePath.startsWith(directory + "/"),
    );
  }

  if (filesToRemove.length === 0) {
    console.log(`${colors.yellow}No files to remove${colors.reset}`);
    return { removed: 0, errors: [] };
  }

  const results = { removed: 0, errors: [] };

  filesToRemove.forEach((file) => {
    try {
      fs.unlinkSync(file.path);
      results.removed++;
      console.log(
        `  ${colors.green}✅ Removed: ${file.relativePath}${colors.reset}`,
      );
    } catch (error) {
      results.errors.push({ file: file.relativePath, error: error.message });
      console.log(
        `  ${colors.red}❌ Failed to remove: ${file.relativePath} (${error.message})${colors.reset}`,
      );
    }
  });

  return results;
}

/**
 * Interactive mode for selective deletion
 */
async function interactiveMode() {
  const rl = createInterface();
  const recordings = scanRecordings();

  try {
    console.log(
      `${colors.bold}${colors.blue}🗑️  Interactive Recording Cleanup${colors.reset}\n`,
    );

    displayRecordings(recordings);

    if (recordings.files.length === 0) {
      console.log(`${colors.green}Nothing to clean up!${colors.reset}`);
      return;
    }

    console.log(`${colors.bold}Cleanup Options:${colors.reset}`);
    console.log(`  ${colors.blue}1.${colors.reset} Remove all recordings`);
    console.log(`  ${colors.blue}2.${colors.reset} Remove by directory`);
    console.log(`  ${colors.blue}3.${colors.reset} Cancel`);
    console.log();

    const choice = await askQuestion(rl, "Select option (1-3): ");

    switch (choice) {
      case "1":
        const confirmAll = await askQuestion(
          rl,
          `${colors.yellow}⚠️  Remove ALL ${recordings.files.length} recording files? (y/N): ${colors.reset}`,
        );

        if (confirmAll === "y" || confirmAll === "yes") {
          console.log(
            `\n${colors.blue}Removing all recordings...${colors.reset}`,
          );
          const results = removeRecordings(recordings);

          if (results.removed > 0) {
            console.log(
              `\n${colors.green}✅ Successfully removed ${results.removed} files${colors.reset}`,
            );
          }

          if (results.errors.length > 0) {
            console.log(
              `\n${colors.red}❌ Failed to remove ${results.errors.length} files:${colors.reset}`,
            );
            results.errors.forEach((err) => {
              console.log(`   - ${err.file}: ${err.error}`);
            });
          }
        } else {
          console.log(`${colors.blue}Cancelled${colors.reset}`);
        }
        break;

      case "2":
        // Get available directories
        const directories = [
          ...new Set(
            recordings.files
              .map((f) => path.dirname(f.relativePath))
              .filter((d) => d !== "."),
          ),
        ];

        if (directories.length === 0) {
          console.log(`${colors.yellow}No subdirectories found${colors.reset}`);
          break;
        }

        console.log(`\n${colors.bold}Available directories:${colors.reset}`);
        directories.forEach((dir, index) => {
          const filesInDir = recordings.files.filter((f) =>
            f.relativePath.startsWith(dir + "/"),
          ).length;
          console.log(
            `  ${colors.blue}${index + 1}.${colors.reset} ${dir}/ (${filesInDir} files)`,
          );
        });

        const dirChoice = await askQuestion(rl, "\nSelect directory number: ");
        const dirIndex = parseInt(dirChoice) - 1;

        if (dirIndex >= 0 && dirIndex < directories.length) {
          const selectedDir = directories[dirIndex];
          const filesInDir = recordings.files.filter((f) =>
            f.relativePath.startsWith(selectedDir + "/"),
          ).length;

          const confirmDir = await askQuestion(
            rl,
            `${colors.yellow}⚠️  Remove ${filesInDir} files from ${selectedDir}/? (y/N): ${colors.reset}`,
          );

          if (confirmDir === "y" || confirmDir === "yes") {
            console.log(
              `\n${colors.blue}Removing recordings from ${selectedDir}/...${colors.reset}`,
            );
            const results = removeRecordings(recordings, {
              directory: selectedDir,
            });

            if (results.removed > 0) {
              console.log(
                `\n${colors.green}✅ Successfully removed ${results.removed} files from ${selectedDir}/${colors.reset}`,
              );
            }

            if (results.errors.length > 0) {
              console.log(
                `\n${colors.red}❌ Failed to remove ${results.errors.length} files:${colors.reset}`,
              );
              results.errors.forEach((err) => {
                console.log(`   - ${err.file}: ${err.error}`);
              });
            }
          } else {
            console.log(`${colors.blue}Cancelled${colors.reset}`);
          }
        } else {
          console.log(`${colors.red}Invalid selection${colors.reset}`);
        }
        break;

      case "3":
      default:
        console.log(`${colors.blue}Cancelled${colors.reset}`);
        break;
    }
  } finally {
    rl.close();
  }
}

/**
 * Command line mode for scripted deletion
 */
function commandLineMode() {
  const args = process.argv.slice(2);
  const recordings = scanRecordings();

  if (args.includes("--help") || args.includes("-h")) {
    console.log(`${colors.bold}Clear Recordings Script${colors.reset}\n`);
    console.log("Usage:");
    console.log("  npm run test:clear-recordings         # Interactive mode");
    console.log(
      "  npm run test:clear-recordings -- --all        # Remove all files",
    );
    console.log(
      "  npm run test:clear-recordings -- --dir <name> # Remove directory",
    );
    console.log(
      "  npm run test:clear-recordings -- --dry-run    # Show what would be removed",
    );
    console.log("\nOptions:");
    console.log("  --all             Remove all recording files");
    console.log("  --dir <name>      Remove files from specific directory");
    console.log(
      "  --dry-run         Show files that would be removed without deleting",
    );
    console.log("  --force           Skip confirmation prompts");
    console.log("  --help, -h        Show this help message");
    return;
  }

  const isAll = args.includes("--all");
  const isDryRun = args.includes("--dry-run");
  const isForce = args.includes("--force");
  const dirIndex = args.indexOf("--dir");
  const directory =
    dirIndex >= 0 && dirIndex + 1 < args.length ? args[dirIndex + 1] : null;

  if (isDryRun) {
    console.log(
      `${colors.bold}${colors.blue}🔍 Dry Run - Files that would be removed:${colors.reset}\n`,
    );
    displayRecordings(recordings);

    if (directory) {
      const filteredFiles = recordings.files.filter((f) =>
        f.relativePath.startsWith(directory + "/"),
      );
      console.log(
        `${colors.blue}Would remove ${filteredFiles.length} files from ${directory}/${colors.reset}`,
      );
    } else if (isAll) {
      console.log(
        `${colors.blue}Would remove all ${recordings.files.length} files${colors.reset}`,
      );
    }
    return;
  }

  if (isAll || directory) {
    let filesToRemove = recordings.files;
    let description = "all recordings";

    if (directory) {
      filesToRemove = recordings.files.filter((f) =>
        f.relativePath.startsWith(directory + "/"),
      );
      description = `recordings from ${directory}/`;
    }

    if (filesToRemove.length === 0) {
      console.log(`${colors.yellow}No files to remove${colors.reset}`);
      return;
    }

    if (!isForce) {
      console.log(
        `${colors.yellow}⚠️  This will remove ${filesToRemove.length} files (${description})${colors.reset}`,
      );
      console.log(
        `${colors.yellow}Use --force to skip this confirmation${colors.reset}`,
      );
      return;
    }

    console.log(`${colors.blue}Removing ${description}...${colors.reset}`);
    const results = removeRecordings(recordings, { directory });

    if (results.removed > 0) {
      console.log(
        `\n${colors.green}✅ Successfully removed ${results.removed} files${colors.reset}`,
      );
    }

    if (results.errors.length > 0) {
      console.log(
        `\n${colors.red}❌ Failed to remove ${results.errors.length} files${colors.reset}`,
      );
      process.exit(1);
    }
  } else {
    // No specific options, use interactive mode
    interactiveMode();
  }
}

/**
 * Main execution
 */
function main() {
  try {
    commandLineMode();
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
  scanRecordings,
  removeRecordings,
  displayRecordings,
};

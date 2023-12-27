#!/usr/bin/env node
const fs = require("fs");
const os = require("os");
const path = require("path");

const appRoot = process.cwd();

const packageJsonPath = path.resolve(`${appRoot}/package.json`);
const packageJson = require(packageJsonPath);

const content = `// this is an auto-generated file. see js/scripts/version-update.js
export const VERSION = "${packageJson.version}";
`;

const filePath = path.join(appRoot, "src", "version.ts");

fs.writeFileSync(filePath, content);

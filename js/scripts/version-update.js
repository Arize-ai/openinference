#!/usr/bin/env node
/* eslint-disable @typescript-eslint/no-require-imports */
const fs = require("fs");
const path = require("path");

const appRoot = process.cwd();

const packageJsonPath = path.resolve(`${appRoot}/package.json`);
const packageJson = require(packageJsonPath);

const content = `// this is an auto-generated file. see js/scripts/version-update.js
export const VERSION = "${packageJson.version}";
`;

const filePath = path.join(appRoot, "src", "version.ts");

fs.writeFileSync(filePath, content);

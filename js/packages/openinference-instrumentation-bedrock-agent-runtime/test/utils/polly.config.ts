// test/utils/polly.config.ts

import { Polly } from "@pollyjs/core";
import NodeHttpAdapter from "@pollyjs/adapter-node-http";
import FsPersister from "@pollyjs/persister-fs";
import path from "path";

// Register globally once
Polly.register(NodeHttpAdapter);
Polly.register(FsPersister);

function sanitizeTestName(name: string): string {
  return name
    .toLowerCase()
    .replace(/\s+/g, "-") // Replace spaces with dashes
    .replace(/[^a-z0-9-_]/g, ""); // Remove all except alphanumeric, dash, underscore
}

export function createPolly(name: string, recordIfMissing = true): Polly {
  const testName =
    expect.getState().currentTestName?.replace(/[^\w\- ]+/g, "") ??
    "unnamed-test";
  const fullName = `${name} - ${sanitizeTestName(testName)}`;

  return new Polly(fullName, {
    adapters: ["node-http"],
    persister: "fs",
    recordIfMissing,
    matchRequestsBy: {
      headers: false,
      method: true,
      body: true,
      url: {
        protocol: true,
        hostname: true,
        pathname: true,
        query: true,
        port: false,
      },
    },
    persisterOptions: {
      fs: {
        recordingsDir: path.resolve(__dirname, "../recordings"),
      },
    },
  });
}

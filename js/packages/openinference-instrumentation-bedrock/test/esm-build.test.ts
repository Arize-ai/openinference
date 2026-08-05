import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { describe, expect, it } from "vitest";

describe("ESM build", () => {
  it("does not emit runtime imports for AWS SDK types", () => {
    const emittedTypes = readFileSync(
      resolve(__dirname, "../dist/esm/types/bedrock-types.js"),
      "utf8",
    );

    expect(emittedTypes).not.toContain("@aws-sdk/client-bedrock-runtime");
  });
});

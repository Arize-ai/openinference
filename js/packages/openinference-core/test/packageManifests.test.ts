import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

// vitest runs with the package directory as cwd (see openai's dualBuild test).
const packagesDir = join(process.cwd(), "..");

const manifests = readdirSync(packagesDir, { withFileTypes: true })
  .filter((entry) => entry.isDirectory())
  // Skip leftover directories (e.g. stale build output) that aren't packages.
  .filter((entry) => existsSync(join(packagesDir, entry.name, "package.json")))
  .map((entry) => {
    const pkg = JSON.parse(readFileSync(join(packagesDir, entry.name, "package.json"), "utf8")) as {
      name: string;
      esnext?: string;
      scripts?: { build?: string };
    };
    return { name: pkg.name, pkg };
  });

// `exports` is the only entry point modern resolvers honour for these
// packages; the non-standard `esnext` field is never read, so a dist/esnext
// build is compiled and published but unreachable.
describe("published package manifests", () => {
  it.each(manifests)("$name does not declare the unresolvable esnext field", ({ pkg }) => {
    expect(pkg.esnext).toBeUndefined();
  });

  it.each(manifests)("$name does not build the unreachable dist/esnext output", ({ pkg }) => {
    expect(pkg.scripts?.build ?? "").not.toContain("tsconfig.esnext.json");
  });
});

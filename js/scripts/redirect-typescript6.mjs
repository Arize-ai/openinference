/**
 * Redirect the `typescript` package to `@typescript/typescript6` so tools that
 * still need the TypeScript 6 compiler API (e.g. TypeDoc) keep working while
 * the workspace builds with TypeScript 7.
 */
import { register } from "node:module";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const hookUrl = pathToFileURL(
  join(dirname(fileURLToPath(import.meta.url)), "typescript6-resolve-hook.mjs"),
);
register(hookUrl);

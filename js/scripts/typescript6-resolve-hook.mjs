import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { pathToFileURL } from "node:url";

const require = createRequire(import.meta.url);
const typescript6Root = dirname(require.resolve("@typescript/typescript6/package.json"));
const typescript6Entry = pathToFileURL(require.resolve("@typescript/typescript6")).href;

export async function resolve(specifier, context, nextResolve) {
  if (specifier === "typescript") {
    return {
      shortCircuit: true,
      url: typescript6Entry,
    };
  }
  if (specifier.startsWith("typescript/")) {
    const subpath = specifier.slice("typescript/".length);
    return {
      shortCircuit: true,
      url: pathToFileURL(join(typescript6Root, subpath)).href,
    };
  }
  return nextResolve(specifier, context);
}

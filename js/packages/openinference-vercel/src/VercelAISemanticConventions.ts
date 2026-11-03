// Type-only names must be re-exported with `export type`: per-file transpilers such as
// esbuild (used by tsx to run the examples) cannot tell they are types and would emit a
// runtime re-export of a non-existent binding.
export {
  VercelAISemanticConventions,
  VercelAISemanticConventionsList,
} from "./AISemanticConventions.js";
export type { VercelAISemanticConvention } from "./AISemanticConventions.js";

// Back-compat exports
export { AISemanticConventions, AISemanticConventionsList } from "./AISemanticConventions.js";
export type { AISemanticConvention } from "./AISemanticConventions.js";

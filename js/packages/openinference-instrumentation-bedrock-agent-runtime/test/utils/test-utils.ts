import { BedrockAgentInstrumentation } from "../../src";

export function setModuleExportsForInstrumentation(
  instrumentation: BedrockAgentInstrumentation,
  moduleExports: unknown,
) {
  (instrumentation as BedrockAgentInstrumentation)[
    "_modules"
  ][0].moduleExports = moduleExports;
}

import { DefaultTraceConfig, traceConfigMetadata } from "./constants";
import { ConfigKey, TraceConfig, TraceConfigOptions } from "./types";
import { assertUnreachable, withSafety } from "../../utils";

const safelyParseInt = withSafety({ fn: parseInt });

type TraceConfigOptionMetadata = (typeof traceConfigMetadata)[ConfigKey];

/**
 * Parses an option based on its type
 * The order of precedence is: optionValue > envValue > defaultValue
 * @param key - The key of the option.
 * @param optionValue - The value passed in the options object.
 *
 */
function parseOption({
  optionValue,
  optionMetadata,
}: {
  optionValue?: number | boolean;
  optionMetadata: TraceConfigOptionMetadata;
}) {
  if (optionValue !== undefined) {
    return optionValue;
  }
  const envValue = process.env[optionMetadata.envKey];
  if (envValue !== undefined) {
    switch (optionMetadata.type) {
      case "number": {
        const maybeEnvNumber = safelyParseInt(envValue);
        return maybeEnvNumber != null && !isNaN(maybeEnvNumber)
          ? maybeEnvNumber
          : optionMetadata.default;
      }
      case "boolean":
        return envValue.toLowerCase() === "true";
      default:
        assertUnreachable(optionMetadata);
    }
  }

  return optionMetadata.default;
}

/**
 * Generates a full trace config object based on passed in options, environment variables, and default values.
 * The order of precedence is: optionValue > envValue > defaultValue
 * @param options - The user provided TraceConfigOptions.
 * @returns A full TraceConfig object with all options set to their final values.
 */
export function generateTraceConfig(options?: TraceConfigOptions): TraceConfig {
  if (options == null) {
    return DefaultTraceConfig;
  }
  return Object.entries(traceConfigMetadata).reduce(
    (config, [key, optionMetadata]) => {
      const configKey = key as ConfigKey;
      return {
        ...config,
        [configKey]: parseOption({
          optionValue: options[configKey],
          optionMetadata,
        }),
      };
    },
    {} as TraceConfig,
  );
}

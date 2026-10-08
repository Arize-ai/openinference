import { assertUnreachable, withSafety } from "../../utils";
import { DefaultTraceConfig, traceConfigMetadata } from "./constants";
import type {
  BooleanTraceConfigFlag,
  NumericTraceConfigFlag,
  TraceConfig,
  TraceConfigOptions,
} from "./types";

const safelyParseInt = withSafety({ fn: parseInt });

type TraceConfigOptionMetadata = BooleanTraceConfigFlag | NumericTraceConfigFlag;

/**
 * Parses an option based on its type
 * The order of precedence is: optionValue > envValue > defaultValue
 * @param key - The key of the option.
 * @param optionMetadata - The {@link TraceConfigOptionMetadata} for the option which includes its type, default value, and environment variable key.
 *
 */
function parseOption({
  optionValue,
  optionMetadata,
}: {
  optionValue?: boolean;
  optionMetadata: BooleanTraceConfigFlag;
}): boolean;
function parseOption({
  optionValue,
  optionMetadata,
}: {
  optionValue?: number;
  optionMetadata: NumericTraceConfigFlag;
}): number;
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
  return {
    hideLLMTools: parseOption({
      optionValue: options.hideLLMTools,
      optionMetadata: traceConfigMetadata.hideLLMTools,
    }),
    hideInputs: parseOption({
      optionValue: options.hideInputs,
      optionMetadata: traceConfigMetadata.hideInputs,
    }),
    hideOutputs: parseOption({
      optionValue: options.hideOutputs,
      optionMetadata: traceConfigMetadata.hideOutputs,
    }),
    hideInputMessages: parseOption({
      optionValue: options.hideInputMessages,
      optionMetadata: traceConfigMetadata.hideInputMessages,
    }),
    hideOutputMessages: parseOption({
      optionValue: options.hideOutputMessages,
      optionMetadata: traceConfigMetadata.hideOutputMessages,
    }),
    hideInputImages: parseOption({
      optionValue: options.hideInputImages,
      optionMetadata: traceConfigMetadata.hideInputImages,
    }),
    hideInputText: parseOption({
      optionValue: options.hideInputText,
      optionMetadata: traceConfigMetadata.hideInputText,
    }),
    hideOutputText: parseOption({
      optionValue: options.hideOutputText,
      optionMetadata: traceConfigMetadata.hideOutputText,
    }),
    hideEmbeddingVectors: parseOption({
      optionValue: options.hideEmbeddingVectors,
      optionMetadata: traceConfigMetadata.hideEmbeddingVectors,
    }),
    base64ImageMaxLength: parseOption({
      optionValue: options.base64ImageMaxLength,
      optionMetadata: traceConfigMetadata.base64ImageMaxLength,
    }),
    hidePrompts: parseOption({
      optionValue: options.hidePrompts,
      optionMetadata: traceConfigMetadata.hidePrompts,
    }),
  };
}

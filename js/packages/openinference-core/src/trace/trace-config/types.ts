import { AttributeValue } from "@opentelemetry/api";
import { OISpan } from "./OISpan";

/**
 * Tracing configuration options that can be set to hide or redact sensitive information from traces
 */
export type TraceConfigOptions = {
  hideInputs?: boolean;
  hideOutputs?: boolean;
  hideInputMessages?: boolean;
  hideOutputMessages?: boolean;
  hideInputImages?: boolean;
  hideInputText?: boolean;
  hideOutputText?: boolean;
  hideEmbeddingVectors?: boolean;
  base64ImageMaxLength?: number;
};

/**
 * A full tracing configuration object that includes all possible options with their final values
 */
export type TraceConfig = Readonly<Required<TraceConfigOptions>>;

export type TraceConfigKey = keyof TraceConfig;

type BooleanTraceConfigFlag = {
  /**
   * The default value for the flag
   */
  default: boolean;
  /**
   * The environment variable key to use to override the default value
   */
  envKey: string;
  /**
   *  The type of the flag
   */
  type: "boolean";
};

type NumericTraceConfigFlag = {
  /**
   * The default value for the flag
   */
  default: number;
  /**
   * The environment variable key to use to override the default value
   */
  envKey: string;
  /**
   * The type of the flag
   */
  type: "number";
};

/**
 * The default and environment information for a value on the TraceConfig
 * Used to generate a full TraceConfig object
 *
 * @example
 * ```typescript
 * {
 *   hideInputs: {
 *     default: DEFAULT_HIDE_INPUTS, // false
 *     envKey: OPENINFERENCE_HIDE_INPUTS, // "OPENINFERENCE_HIDE_INPUTS"
 *     type: "boolean",
 *   }
 * }
 * ```
 */
export type TraceConfigFlag = BooleanTraceConfigFlag | NumericTraceConfigFlag;

/**
 * The arguments for a masking rule, used to determine if a value should be masked (redacted or omitted)
 */
export type MaskingRuleArgs = {
  /**
   * The trace config to use to determine if the value should be masked
   */
  config: TraceConfig;
  /**
   * The key of the attribute to mask
   */
  key: string;
  /**
   * The value of the attribute to mask
   */
  value?: AttributeValue;
};

/**
 * A condition to determine if a value should be masked
 * and an action that masks (removes or redacts) the value if the condition is met
 *   @example
 * ```typescript
 *  const config = {hideInputText: true}
 *  const key = "llm.input_messages.0.message.content"
 *  if (maskInputTextRule.condition({ config, key })) {
 *    return maskInputTextRule.action()
 *  }
 * ```
 */
export type MaskingRule = {
  /**
   *
   * @param args The {@linkcode MaskingRuleArgs} to determine if the value should be masked
   * @returns true if the value should be masked, false otherwise
   */
  condition: (args: MaskingRuleArgs) => boolean;
  /**
   * An action to be applied if the condition is met
   * @returns A redacted value or undefined
   */
  action: () => AttributeValue | undefined;
};

/**
 * A callback that is called when a new active {@link OISpan} is created
 */
export type OpenInferenceActiveSpanCallback = (span: OISpan) => void;

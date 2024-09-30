import { TraceConfigKey, TraceConfig, TraceConfigFlag } from "./types";

/** Hides input value & messages */
export const OPENINFERENCE_HIDE_INPUTS = "OPENINFERENCE_HIDE_INPUTS";
/** Hides output value & messages */
export const OPENINFERENCE_HIDE_OUTPUTS = "OPENINFERENCE_HIDE_OUTPUTS";
/** Hides all input messages */
export const OPENINFERENCE_HIDE_INPUT_MESSAGES =
  "OPENINFERENCE_HIDE_INPUT_MESSAGES";
/** Hides all output messages */
export const OPENINFERENCE_HIDE_OUTPUT_MESSAGES =
  "OPENINFERENCE_HIDE_OUTPUT_MESSAGES";
/** Hides images from input messages */
export const OPENINFERENCE_HIDE_INPUT_IMAGES =
  "OPENINFERENCE_HIDE_INPUT_IMAGES";
/** Hides text from input messages */
export const OPENINFERENCE_HIDE_INPUT_TEXT = "OPENINFERENCE_HIDE_INPUT_TEXT";
/** Hides text from output messages */
export const OPENINFERENCE_HIDE_OUTPUT_TEXT = "OPENINFERENCE_HIDE_OUTPUT_TEXT";
/** Hides embedding vectors */
export const OPENINFERENCE_HIDE_EMBEDDING_VECTORS =
  "OPENINFERENCE_HIDE_EMBEDDING_VECTORS";
/** Limits characters of a base64 encoding of an image */
export const OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH =
  "OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH";

export const DEFAULT_HIDE_INPUTS = false;
export const DEFAULT_HIDE_OUTPUTS = false;

export const DEFAULT_HIDE_INPUT_MESSAGES = false;
export const DEFAULT_HIDE_OUTPUT_MESSAGES = false;

export const DEFAULT_HIDE_INPUT_IMAGES = false;
export const DEFAULT_HIDE_INPUT_TEXT = false;
export const DEFAULT_HIDE_OUTPUT_TEXT = false;

export const DEFAULT_HIDE_EMBEDDING_VECTORS = false;
export const DEFAULT_BASE64_IMAGE_MAX_LENGTH = 32000;

/** When a value is hidden, it will be replaced by this redacted value */
export const REDACTED_VALUE = "__REDACTED__";

/**
 * The default, environment, and type information for each value on the TraceConfig
 * Used to generate a full TraceConfig object with the correct types and default values
 */
export const traceConfigMetadata: Readonly<
  Record<TraceConfigKey, TraceConfigFlag>
> = {
  hideInputs: {
    default: DEFAULT_HIDE_INPUTS,
    envKey: OPENINFERENCE_HIDE_INPUTS,
    type: "boolean",
  },
  hideOutputs: {
    default: DEFAULT_HIDE_OUTPUTS,
    envKey: OPENINFERENCE_HIDE_OUTPUTS,
    type: "boolean",
  },
  hideInputMessages: {
    default: DEFAULT_HIDE_INPUT_MESSAGES,
    envKey: OPENINFERENCE_HIDE_INPUT_MESSAGES,
    type: "boolean",
  },
  hideOutputMessages: {
    default: DEFAULT_HIDE_OUTPUT_MESSAGES,
    envKey: OPENINFERENCE_HIDE_OUTPUT_MESSAGES,
    type: "boolean",
  },
  hideInputImages: {
    default: DEFAULT_HIDE_INPUT_IMAGES,
    envKey: OPENINFERENCE_HIDE_INPUT_IMAGES,
    type: "boolean",
  },
  hideInputText: {
    default: DEFAULT_HIDE_INPUT_TEXT,
    envKey: OPENINFERENCE_HIDE_INPUT_TEXT,
    type: "boolean",
  },
  hideOutputText: {
    default: DEFAULT_HIDE_OUTPUT_TEXT,
    envKey: OPENINFERENCE_HIDE_OUTPUT_TEXT,
    type: "boolean",
  },
  hideEmbeddingVectors: {
    default: DEFAULT_HIDE_EMBEDDING_VECTORS,
    envKey: OPENINFERENCE_HIDE_EMBEDDING_VECTORS,
    type: "boolean",
  },
  base64ImageMaxLength: {
    default: DEFAULT_BASE64_IMAGE_MAX_LENGTH,
    envKey: OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH,
    type: "number",
  },
};

export const DefaultTraceConfig: TraceConfig = {
  hideInputs: DEFAULT_HIDE_INPUTS,
  hideOutputs: DEFAULT_HIDE_OUTPUTS,
  hideInputMessages: DEFAULT_HIDE_INPUT_MESSAGES,
  hideOutputMessages: DEFAULT_HIDE_OUTPUT_MESSAGES,
  hideInputImages: DEFAULT_HIDE_INPUT_IMAGES,
  hideInputText: DEFAULT_HIDE_INPUT_TEXT,
  hideOutputText: DEFAULT_HIDE_OUTPUT_TEXT,
  hideEmbeddingVectors: DEFAULT_HIDE_EMBEDDING_VECTORS,
  base64ImageMaxLength: DEFAULT_BASE64_IMAGE_MAX_LENGTH,
};

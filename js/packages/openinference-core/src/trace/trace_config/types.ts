import { AttributeValue } from "@opentelemetry/api";
import { OpenInferenceSpan } from "./OpenInferenceSpan";

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

export type TraceConfig = Readonly<Required<TraceConfigOptions>>;

export type TraceConfigKey = keyof TraceConfig;

export type TraceConfigMetadata =
  | {
      default: boolean;
      envKey: string;
      type: "boolean";
    }
  | {
      default: number;
      envKey: string;
      type: "number";
    };

export type MaskingRuleArgs = {
  config: TraceConfig;
  key: string;
  value?: AttributeValue;
};

export type MaskingRule = {
  condition: (args: MaskingRuleArgs) => boolean;
  action: () => AttributeValue | undefined;
};

export type OpenInferenceActiveSpanCallback = (span: OpenInferenceSpan) => void;

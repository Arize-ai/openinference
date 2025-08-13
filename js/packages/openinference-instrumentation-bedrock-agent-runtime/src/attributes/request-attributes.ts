import { Attributes } from "@opentelemetry/api";
import {
  LLMProvider,
  MimeType,
  OpenInferenceSpanKind,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";
import { InvokeAgentCommand } from "@aws-sdk/client-bedrock-agent-runtime";
import { isAttributeValue } from "@opentelemetry/core";

export function extractBaseRequestAttributes(
  command: InvokeAgentCommand,
): Attributes {
  const attributes: Attributes = {
    [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.AGENT,
    [SemanticConventions.LLM_SYSTEM]: "bedrock",
    [SemanticConventions.INPUT_MIME_TYPE]: MimeType.TEXT,
    [SemanticConventions.LLM_PROVIDER]: LLMProvider.AWS,
  };

  // Add invocation parameters for model configuration
  const { inputText: _inputText, ...invocationParams } = command.input;
  const jsonParams = Object.fromEntries(
    Object.entries(invocationParams).filter(([, value]) =>
      isAttributeValue(value),
    ),
  );
  if (Object.keys(jsonParams).length > 0) {
    attributes[SemanticConventions.LLM_INVOCATION_PARAMETERS] =
      JSON.stringify(jsonParams);
  }
  attributes[SemanticConventions.INPUT_VALUE] = command.input?.inputText || "";
  return attributes;
}

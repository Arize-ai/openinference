import {
  LLMProvider,
  MimeType,
  OpenInferenceSpanKind,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";
import {
  InvokeAgentCommand,
  InvokeAgentRequest,
} from "@aws-sdk/client-bedrock-agent-runtime";

function extractInvocationParameters(
  requestBody: InvokeAgentRequest,
): Record<string, string> {
  const invocationParams: Record<string, string> = {};

  if (requestBody.agentId) {
    invocationParams.agentId = requestBody.agentId;
  }
  if (requestBody.agentAliasId) {
    invocationParams.agentAliasId = requestBody.agentAliasId;
  }
  if (requestBody.sessionId) {
    invocationParams.sessionId = requestBody.sessionId;
  }
  return invocationParams;
}

export function extractBaseRequestAttributes(
  command: InvokeAgentCommand,
): Record<string, string> {
  const attributes: Record<string, string> = {
    [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.LLM,
    [SemanticConventions.LLM_SYSTEM]: "bedrock",
    [SemanticConventions.INPUT_MIME_TYPE]: MimeType.JSON,
    [SemanticConventions.LLM_PROVIDER]: LLMProvider.AWS,
  };

  // Add invocation parameters for model configuration
  const invocationParams = extractInvocationParameters(command.input);
  if (Object.keys(invocationParams).length > 0) {
    attributes[SemanticConventions.LLM_INVOCATION_PARAMETERS] =
      JSON.stringify(invocationParams);
  }
  attributes[SemanticConventions.INPUT_VALUE] = command.input?.inputText || "";
  return attributes;
}

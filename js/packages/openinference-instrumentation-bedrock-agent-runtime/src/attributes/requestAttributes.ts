import type {
  InvokeAgentCommand,
  RetrieveAndGenerateCommand,
  RetrieveAndGenerateStreamCommand,
  RetrieveCommand,
  RetrieveCommandInput,
} from "@aws-sdk/client-bedrock-agent-runtime";
import type { Attributes } from "@opentelemetry/api";
import { diag } from "@opentelemetry/api";
import { isAttributeValue } from "@opentelemetry/core";

import { safelyJSONStringify, withSafety } from "@arizeai/openinference-core";
import {
  LLMProvider,
  MimeType,
  OpenInferenceSpanKind,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";

import { getInputAttributes, getLLMInvocationParameterAttributes } from "./attributeUtils";
import { extractRagInvocationParams, getModelNameAttributes } from "./ragAttributeExtractionUtils";

function extractBaseRequestAttributes(command: InvokeAgentCommand): Attributes {
  const attributes: Attributes = {
    [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.AGENT,
    [SemanticConventions.LLM_SYSTEM]: "bedrock",
    [SemanticConventions.INPUT_MIME_TYPE]: MimeType.TEXT,
    [SemanticConventions.LLM_PROVIDER]: LLMProvider.AWS,
  };

  // Add invocation parameters for model configuration
  const { inputText: _inputText, ...invocationParams } = command.input;
  const jsonParams = Object.fromEntries(
    Object.entries(invocationParams).filter(([, value]) => isAttributeValue(value)),
  );
  if (Object.keys(jsonParams).length > 0) {
    attributes[SemanticConventions.LLM_INVOCATION_PARAMETERS] =
      safelyJSONStringify(jsonParams) ?? undefined;
  }
  attributes[SemanticConventions.INPUT_VALUE] = command.input?.inputText || "";
  return attributes;
}

/**
 * Extracts base request attributes for Bedrock RAG (Retrieve and Generate) operations.
 *
 * This function builds a set of OpenTelemetry attributes for a RAG request, including
 * span kind, system/provider, model name, input text, and invocation parameters.
 *
 * @param command The RetrieveAndGenerateStreamCommand or RetrieveAndGenerateCommand instance.
 * @returns Attributes for OpenTelemetry span instrumentation.
 */
function extractRagBaseRequestAttributes(
  command: RetrieveAndGenerateStreamCommand | RetrieveAndGenerateCommand,
): Attributes {
  return {
    [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.RETRIEVER,
    [SemanticConventions.LLM_PROVIDER]: LLMProvider.AWS,
    ...getModelNameAttributes(command?.input),
    ...getInputAttributes(command?.input?.input?.text),
    ...extractRagInvocationParams(command?.input),
  };
}

/**
 * Extracts base request attributes for Bedrock Retrieve operations.
 *
 * This function builds a set of OpenTelemetry attributes for a Retrieve request, including
 * span kind, system/provider, input text, and invocation parameters.
 *
 * @param command The RetrieveCommand instance.
 * @returns Attributes for OpenTelemetry span instrumentation.
 */
function extractRetrieveBaseRequestAttributes(command: RetrieveCommand): Attributes {
  const input: RetrieveCommandInput = command?.input || {};
  const invocationParams: Record<string, unknown> = {
    knowledgeBaseId: input.knowledgeBaseId,
  };
  if (input?.nextToken) {
    invocationParams["nextToken"] = input.nextToken;
  }
  if (input?.retrievalConfiguration) {
    invocationParams["retrievalConfiguration"] = input.retrievalConfiguration;
  }
  return {
    [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.RETRIEVER,
    [SemanticConventions.LLM_PROVIDER]: LLMProvider.AWS,
    ...getInputAttributes(command?.input?.retrievalQuery?.text),
    ...getLLMInvocationParameterAttributes(invocationParams),
  };
}

export const safelyExtractBaseRequestAttributes = withSafety({
  fn: extractBaseRequestAttributes,
  onError: (err) => {
    diag.warn(
      `Openinference warning, unable to extract base request attributes, some spans may be missing or incomplete. Error: ${err instanceof Error ? err.message : err}`,
    );
  },
});

/**
 * Safely extracts base request attributes for Bedrock RAG (Retrieve and Generate) operations.
 *
 * This is a wrapper around extractRagBaseRequestAttributes that catches errors and logs warnings,
 * ensuring that instrumentation does not break if attribute extraction fails.
 *
 * @see extractRagBaseRequestAttributes
 */
export const safelyExtractBaseRagAttributes = withSafety({
  fn: extractRagBaseRequestAttributes,
  onError: (err) => {
    diag.warn(
      `Openinference warning, unable to extract base request attributes, some spans may be missing or incomplete. Error: ${err instanceof Error ? err.message : err}`,
    );
  },
});

/**
 * Safely extracts base request attributes for Bedrock Retrieve operations.
 *
 * This is a wrapper around extractRetrieveBaseRequestAttributes that catches errors and logs warnings,
 * ensuring that instrumentation does not break if attribute extraction fails.
 *
 * @see extractRetrieveBaseRequestAttributes
 */
export const safelyExtractBaseRetrieveAttributes = withSafety({
  fn: extractRetrieveBaseRequestAttributes,
  onError: (err) => {
    diag.warn(
      `Openinference warning, unable to extract base request attributes, some spans may be missing or incomplete. Error: ${err instanceof Error ? err.message : err}`,
    );
  },
});

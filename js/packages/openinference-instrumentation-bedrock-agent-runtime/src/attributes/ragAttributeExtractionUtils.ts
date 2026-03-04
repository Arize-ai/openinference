import type * as bedrockAgentRunTime from "@aws-sdk/client-bedrock-agent-runtime";
import type { Attributes } from "@opentelemetry/api";

import { safelyJSONStringify } from "@arizeai/openinference-core";

import { getObjectDataFromUnknown } from "../utils/jsonUtils";
import {
  getDocumentAttributes,
  getLLMInvocationParameterAttributes,
  getLLMModelNameAttributes,
  getOutputAttributes,
} from "./attributeUtils";
import type { DocumentReference } from "./types";

/**
 * Extracts invocation parameters from a Bedrock RAG (Retrieve and Generate) command input.
 *
 * This function serializes and collects relevant configuration and session parameters
 * from the input, returning them as OpenTelemetry attributes for trace enrichment.
 *
 * @param input Bedrock RetrieveAndGenerateCommandInput containing configuration and session info.
 * @returns OpenTelemetry attributes for invocation parameters.
 */
export function extractRagInvocationParams(
  input: bedrockAgentRunTime.RetrieveAndGenerateCommandInput,
): Attributes {
  const invocationParams: Attributes = {};
  if (input?.retrieveAndGenerateConfiguration) {
    invocationParams["retrieveAndGenerateConfiguration"] =
      safelyJSONStringify(input.retrieveAndGenerateConfiguration) || undefined;
  }
  if (input?.sessionConfiguration) {
    invocationParams["sessionConfiguration"] =
      safelyJSONStringify(input.sessionConfiguration) || undefined;
  }
  if (input?.sessionId) {
    invocationParams["sessionId"] = input.sessionId;
  }
  return getLLMInvocationParameterAttributes(invocationParams);
}

/**
 * Extracts the model name/ARN from a Bedrock RAG (Retrieve and Generate) command input.
 *
 * This function inspects the retrieveAndGenerateConfiguration to determine the model
 * used for the RAG operation, supporting both knowledge base and external sources.
 *
 * @param input Bedrock RetrieveAndGenerateCommandInput containing model configuration.
 * @returns OpenTelemetry attributes for the model name/ARN, or an empty object if not found.
 */
export function getModelNameAttributes(
  input: bedrockAgentRunTime.RetrieveAndGenerateCommandInput,
): Attributes {
  if (input?.retrieveAndGenerateConfiguration) {
    const retrieveAndGenerateConfig: bedrockAgentRunTime.RetrieveAndGenerateConfiguration =
      input.retrieveAndGenerateConfiguration;
    if (retrieveAndGenerateConfig?.type === "KNOWLEDGE_BASE") {
      return getLLMModelNameAttributes(
        retrieveAndGenerateConfig?.knowledgeBaseConfiguration?.modelArn,
      );
    }
  }
  if (input?.retrieveAndGenerateConfiguration?.externalSourcesConfiguration?.modelArn) {
    return getLLMModelNameAttributes(
      input.retrieveAndGenerateConfiguration.externalSourcesConfiguration.modelArn,
    );
  }
  return {};
}

function constructRagDocument(document: bedrockAgentRunTime.RetrievedReference): DocumentReference {
  const location = getObjectDataFromUnknown({
    data: document,
    key: "location",
  });
  return {
    metadata: document.metadata,
    content: document.content,
    ...(location && { location: location }),
  };
}

/**
 * Extracts document-level attributes from a list of Bedrock RAG citations.
 *
 * This function iterates over citations and their retrieved references, building
 * OpenTelemetry attributes for each document, including metadata, content, and location.
 *
 * @param citations Array of Bedrock Citation objects containing retrievedReferences.
 * @returns OpenTelemetry attributes for all referenced documents.
 */
export function extractRetrievedReferencesAttributes(
  citations: bedrockAgentRunTime.Citation[],
): Attributes {
  let attributes: Attributes = {};
  let index = 0;
  for (const citation of Array.isArray(citations) ? citations : []) {
    const documents: bedrockAgentRunTime.RetrievedReference[] = Array.isArray(
      citation?.retrievedReferences,
    )
      ? citation.retrievedReferences
      : [];
    for (const document of documents) {
      const ragDocument = constructRagDocument(document);
      attributes = {
        ...attributes,
        ...getDocumentAttributes(index, ragDocument),
      };
      index += 1;
    }
  }
  return attributes;
}

/**
 * Extracts response attributes from Bedrock RAG (Retrieve and Generate) operation results.
 *
 * This function processes the response from a Bedrock retrieve_and_generate operation
 * and extracts both the generated output and the retrieved document citations.
 * It handles the complex structure of RAG responses that include both generation
 * results and retrieval citations.
 *
 * @param response Response object from the retrieve_and_generate operation, containing:
 *                 - output: Generated text output
 *                 - citations: List of citations with retrieved references
 *                 - sessionId: Session identifier (optional)
 * @returns OpenTelemetry attributes containing:
 *          - Document attributes: Information about each retrieved document
 *            from citations, including IDs, content, scores, and metadata
 *          - Output attributes: The generated text response
 */
export function extractBedrockRagResponseAttributes(
  response: bedrockAgentRunTime.RetrieveAndGenerateCommandOutput,
): Attributes {
  const citations = Array.isArray(response?.citations) ? response.citations : [];
  const outputText = response?.output?.text;
  return {
    ...extractRetrievedReferencesAttributes(citations),
    ...getOutputAttributes(outputText),
  };
}

/**
 * Extracts response attributes from Bedrock Retrieve operation results.
 *
 * This function processes the response from a Bedrock Retrieve operation and extracts
 * document-level attributes for each retrieved reference. It is used to build OpenTelemetry
 * attributes for trace and span enrichment.
 *
 * @param response Bedrock RetrieveCommandOutput containing retrievalResults.
 * @returns OpenTelemetry attributes for all retrieved documents.
 */
export function extractBedrockRetrieveResponseAttributes(
  response: bedrockAgentRunTime.RetrieveCommandOutput,
): Attributes {
  const documents: bedrockAgentRunTime.RetrievedReference[] = Array.isArray(
    response?.retrievalResults,
  )
    ? response?.retrievalResults
    : [];
  let attributes: Attributes = {};
  documents.forEach((document, index) => {
    attributes = {
      ...attributes,
      ...getDocumentAttributes(index, constructRagDocument(document)),
    };
  });
  return attributes;
}

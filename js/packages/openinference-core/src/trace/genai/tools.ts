import { SemanticConventions } from "@arizeai/openinference-semantic-conventions";

import type { UnknownRecord } from "../../types";
import { isDefined, isNullish, isPresent, isRecord, isUndefined } from "../../utils";
import { getIndexedBuckets } from "./flattenedAttributeUtils";
import { GenAIToolType } from "./genAIAttributes";
import {
  isEmptyValue,
  parseJSONArray,
  parseJSONIfString,
  toNonEmptyString,
} from "./serializationUtils";

/** Fields whose presence marks a record as a tool definition. */
const TOOL_DEFINITION_FIELDS = ["name", "parameters", "description"];

/**
 * Adds `type: "function"` to a tool definition that has tool fields but no type.
 * @param toolDefinition - The tool definition
 * @returns The tool definition, copied only when a type is added
 */
function withDefaultToolType(toolDefinition: UnknownRecord): UnknownRecord {
  const hasToolFields = TOOL_DEFINITION_FIELDS.some((field) => isPresent(toolDefinition[field]));
  if (isNullish(toolDefinition.type) && hasToolFields) {
    return { ...toolDefinition, type: GenAIToolType.FUNCTION };
  }
  return toolDefinition;
}

/**
 * Normalizes an OpenAI tool definition:
 * `{ type: "function", function: { name, description, parameters } }`.
 *
 * @param params.toolDefinition - The tool definition
 * @param params.functionDefinition - Its nested `function` record
 */
function normalizeOpenAIToolDefinition({
  toolDefinition,
  functionDefinition,
}: {
  toolDefinition: UnknownRecord;
  functionDefinition: UnknownRecord;
}): UnknownRecord {
  const normalized: UnknownRecord = {
    type: toNonEmptyString(toolDefinition.type) ?? GenAIToolType.FUNCTION,
  };
  const name = toNonEmptyString(functionDefinition.name);
  if (name) {
    normalized.name = name;
  }
  const description = toNonEmptyString(functionDefinition.description);
  if (description) {
    normalized.description = description;
  }
  if (!isEmptyValue(functionDefinition.parameters)) {
    normalized.parameters = functionDefinition.parameters;
  }
  return normalized;
}

/**
 * Normalizes an Anthropic tool definition: `{ name, description, input_schema }`.
 * @param toolDefinition - The tool definition
 */
function normalizeAnthropicToolDefinition(toolDefinition: UnknownRecord): UnknownRecord {
  const { input_schema: inputSchema, ...rest } = toolDefinition;
  return withDefaultToolType({ ...rest, parameters: inputSchema });
}

/**
 * Normalizes a Bedrock Converse tool definition:
 * `{ toolSpec: { name, description, inputSchema: { json } } }`.
 * @param toolSpec - The nested `toolSpec` record of the tool definition
 */
function normalizeBedrockToolDefinition(toolSpec: UnknownRecord): UnknownRecord {
  const { inputSchema, ...rest } = toolSpec;
  const parameters = isRecord(inputSchema) ? inputSchema.json : undefined;
  return withDefaultToolType({ ...rest, ...(isPresent(parameters) ? { parameters } : {}) });
}

/**
 * Normalizes a recognized provider tool definition to the OTel GenAI shape.
 * Unknown provider shapes are preserved so the conversion remains lossless.
 * @param toolDefinition - The provider tool definition
 */
function normalizeProviderToolDefinition(toolDefinition: UnknownRecord): UnknownRecord {
  const functionDefinition = toolDefinition.function;
  if (isRecord(functionDefinition) && !isEmptyValue(functionDefinition)) {
    return normalizeOpenAIToolDefinition({ toolDefinition, functionDefinition });
  }
  const toolSpec = toolDefinition.toolSpec;
  if (isRecord(toolSpec) && !isEmptyValue(toolSpec)) {
    return normalizeBedrockToolDefinition(toolSpec);
  }
  const isAnthropicToolDefinition =
    isPresent(toolDefinition.input_schema) && isNullish(toolDefinition.parameters);
  if (isAnthropicToolDefinition) {
    return normalizeAnthropicToolDefinition(toolDefinition);
  }
  return withDefaultToolType({ ...toolDefinition });
}

/**
 * Normalizes an `llm.tools.<index>.tool.*` bucket. `tool.name` and
 * `tool.description` take precedence over the values inside `tool.json_schema`.
 * @param toolDefinition - The `tool.*` attributes of one tool
 * @returns The normalized tool definition, or undefined when it has neither a schema nor a name
 */
function normalizeOpenInferenceToolDefinition(
  toolDefinition: UnknownRecord,
): UnknownRecord | undefined {
  const normalizedSchema = normalizeToolDefinition(
    toolDefinition[SemanticConventions.TOOL_JSON_SCHEMA],
  );
  const name = toNonEmptyString(toolDefinition[SemanticConventions.TOOL_NAME]);
  const description = toNonEmptyString(toolDefinition[SemanticConventions.TOOL_DESCRIPTION]);
  if (isUndefined(normalizedSchema) && isUndefined(name)) {
    return undefined;
  }
  return {
    ...(normalizedSchema ?? { type: GenAIToolType.FUNCTION }),
    ...(isDefined(name) ? { name } : {}),
    ...(isDefined(description) ? { description } : {}),
  };
}

/**
 * Normalizes one tool definition to the OTel GenAI shape, from either an
 * OpenInference `tool.*` record or a provider tool definition.
 * @param toolDefinition - A tool definition, or a string holding it as JSON
 * @returns The normalized tool definition, or undefined when it is not a record
 */
function normalizeToolDefinition(toolDefinition: unknown): UnknownRecord | undefined {
  const parsedToolDefinition = parseJSONIfString(toolDefinition);
  if (!isRecord(parsedToolDefinition)) {
    return undefined;
  }
  const isOpenInferenceToolDefinition =
    SemanticConventions.TOOL_JSON_SCHEMA in parsedToolDefinition ||
    SemanticConventions.TOOL_NAME in parsedToolDefinition ||
    SemanticConventions.TOOL_DESCRIPTION in parsedToolDefinition;
  if (isOpenInferenceToolDefinition) {
    return normalizeOpenInferenceToolDefinition(parsedToolDefinition);
  }
  return normalizeProviderToolDefinition(parsedToolDefinition);
}

/**
 * Converts the tools advertised to the LLM (`llm.tools`) to the value of
 * `gen_ai.tool.definitions`. A JSON-marshaled `llm.tools` value takes
 * precedence over flattened `llm.tools.<index>.*` attributes.
 * @param attributes - The OpenInference span attributes
 */
export function getToolDefinitions(attributes: UnknownRecord): UnknownRecord[] {
  const rootToolDefinitions = parseJSONArray(attributes[SemanticConventions.LLM_TOOLS])
    .map(normalizeToolDefinition)
    .filter(isDefined);
  if (rootToolDefinitions.length > 0) {
    return rootToolDefinitions;
  }
  return getIndexedBuckets({ attributes, prefix: SemanticConventions.LLM_TOOLS })
    .map(normalizeToolDefinition)
    .filter(isDefined);
}

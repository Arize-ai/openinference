import { SemanticConventions } from "@arizeai/openinference-semantic-conventions";

import { isDefined, isNullish, isPresent, isRecord, isUndefined } from "../../utils";
import type { UnknownRecord } from "./conversionUtils";
import {
  getIndexedBuckets,
  isEmptyValue,
  parseJSONArray,
  parseJSONIfString,
  toNonEmptyString,
} from "./conversionUtils";
import { GenAIToolType } from "./genAIAttributes";

const TOOL_DEFINITION_FIELDS = ["name", "parameters", "description"];

function withDefaultToolType(toolDefinition: UnknownRecord): UnknownRecord {
  const hasToolFields = TOOL_DEFINITION_FIELDS.some((field) => isPresent(toolDefinition[field]));
  if (isNullish(toolDefinition.type) && hasToolFields) {
    return { ...toolDefinition, type: GenAIToolType.FUNCTION };
  }
  return toolDefinition;
}

/** `{ type: "function", function: { name, description, parameters } }` */
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

/** `{ name, description, input_schema }` */
function normalizeAnthropicToolDefinition(toolDefinition: UnknownRecord): UnknownRecord {
  const { input_schema: inputSchema, ...rest } = toolDefinition;
  return withDefaultToolType({ ...rest, parameters: inputSchema });
}

/**
 * Normalizes a recognized provider tool definition to the OTel GenAI shape.
 * Unknown provider shapes are preserved so the conversion remains lossless.
 */
function normalizeProviderToolDefinition(toolDefinition: UnknownRecord): UnknownRecord {
  const functionDefinition = toolDefinition.function;
  if (isRecord(functionDefinition) && !isEmptyValue(functionDefinition)) {
    return normalizeOpenAIToolDefinition({ toolDefinition, functionDefinition });
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

import type Anthropic from "@anthropic-ai/sdk";
import type { Attributes } from "@opentelemetry/api";

import type { TraceConfig } from "@arizeai/openinference-core";
import { getInputAttributes, safelyJSONStringify } from "@arizeai/openinference-core";
import { SemanticConventions } from "@arizeai/openinference-semantic-conventions";

import type { MessageRequestParams, MessageParam, ToolDefinition } from "../types";
import { safelyGetAttributes, toJSONOutput } from "../utils";
import { getAnthropicInputMessageAttributes } from "./content";

/**
 * Builds the request attributes of an Anthropic messages request.
 */
export function getAnthropicRequestAttributes(
  body: MessageRequestParams,
  traceConfig: TraceConfig,
): Attributes {
  // The system prompt is input message 0, so it is left out of the invocation
  // parameters, which hideInputs does not hide.
  const { messages: _messages, system: _system, ...invocationParameters } = body;
  // Each part is extracted on its own, so a malformed message or tool costs
  // only its own attributes.
  const attributes: Attributes = {
    [SemanticConventions.LLM_MODEL_NAME]: body.model,
    [SemanticConventions.LLM_REQUEST_MODEL_NAME]: body.model,
    ...safelyGetAttributes(() => getAnthropicInputMessagesAttributes(body, traceConfig)),
    ...safelyGetAttributes(() => getAnthropicToolsJSONSchema(body)),
  };
  Object.assign(attributes, getInputAttributes(toJSONOutput(body)));
  const invocationParametersJSON = safelyJSONStringify(invocationParameters);
  if (invocationParametersJSON != null) {
    attributes[SemanticConventions.LLM_INVOCATION_PARAMETERS] = invocationParametersJSON;
  }
  return attributes;
}

/**
 * Converts the body of an Anthropic messages request to LLM input messages
 */
function getAnthropicInputMessagesAttributes(
  body: MessageRequestParams,
  traceConfig: TraceConfig,
): Attributes {
  // The top-level system prompt is the first input message, ahead of the
  // conversation, so the prompt the model saw is recorded in order.
  // An empty system prompt is no message, as in the Python instrumentation.
  const system = body.system;
  const hasSystem =
    typeof system === "string" ? system !== "" : Array.isArray(system) && system.length > 0;
  const messages = Array.isArray(body.messages) ? body.messages : [];
  // Each message is mapped on its own, so a malformed one keeps its index
  // without dropping the others.
  const messagesAttributes = [
    ...(hasSystem && system != null
      ? [safelyGetAttributes(() => getAnthropicSystemMessageAttributes(system))]
      : []),
    ...messages.map((message: MessageParam) =>
      safelyGetAttributes(() => getAnthropicInputMessageAttributes(message, traceConfig)),
    ),
  ];
  return messagesAttributes.reduce<Attributes>((acc, messageAttributes, index) => {
    const indexPrefix = `${SemanticConventions.LLM_INPUT_MESSAGES}.${index}.`;
    // Flatten the attributes on the index prefix
    for (const [key, value] of Object.entries(messageAttributes)) {
      acc[`${indexPrefix}${key}`] = value;
    }
    return acc;
  }, {});
}

/**
 * Converts the top-level `system` request param, a string or a list of text
 * blocks, to a message with the system role.
 */
function getAnthropicSystemMessageAttributes(
  system: NonNullable<MessageRequestParams["system"]>,
): Attributes {
  const attributes: Attributes = { [SemanticConventions.MESSAGE_ROLE]: "system" };
  if (typeof system === "string") {
    attributes[SemanticConventions.MESSAGE_CONTENT] = system;
    return attributes;
  }
  system.forEach((block, index) => {
    const contentsIndexPrefix = `${SemanticConventions.MESSAGE_CONTENTS}.${index}.`;
    attributes[`${contentsIndexPrefix}${SemanticConventions.MESSAGE_CONTENT_TYPE}`] = "text";
    attributes[`${contentsIndexPrefix}${SemanticConventions.MESSAGE_CONTENT_TEXT}`] = block.text;
  });
  return attributes;
}

/**
 * Lists the tools offered to the model at the end of a request's messages: the
 * `tools` param, plus the tools a beta request's `tool_addition` blocks define
 * inline, minus those its `tool_removal` blocks withdraw, in conversation
 * order. A compaction's `tool_changes`, which stand in for the compacted turns',
 * count too. A `tool_addition` that references a tool by name offers again a
 * tool that was removed, and a tool is listed once however often it is added.
 */
function getAnthropicToolDefinitions(body: MessageRequestParams): ToolDefinition[] {
  const tools: ToolDefinition[] = Array.isArray(body.tools) ? body.tools : [];
  // Maps keep insertion order, so offered lists the tools in the order offered.
  const offered = new Map<string, ToolDefinition>();
  tools.forEach((tool, index) => offered.set(getToolKey(tool, index), tool));
  const defined = new Map(offered);
  const apply = (change: AnthropicToolChange) => {
    if (change?.type === "tool_addition") {
      const tool = change.tool;
      if (tool?.type === "tool_definition") {
        if (tool.definition != null) {
          const key = getToolKey(tool.definition, defined.size);
          // Deleting first moves a redefined tool to the end.
          offered.delete(key);
          offered.set(key, tool.definition);
          defined.set(key, tool.definition);
        }
      } else {
        const key = getToolReferenceKey(tool);
        const definition = key != null ? defined.get(key) : undefined;
        if (key != null && definition != null && !offered.has(key)) {
          offered.set(key, definition);
        }
      }
    } else if (change?.type === "tool_removal") {
      const key = getToolReferenceKey(change.tool);
      if (key != null) {
        offered.delete(key);
      }
    }
  };
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const message of messages) {
    if (!Array.isArray(message?.content)) {
      continue;
    }
    for (const part of message.content) {
      if (part?.type === "tool_addition" || part?.type === "tool_removal") {
        apply(part);
      } else if (part?.type === "compaction" && Array.isArray(part.tool_changes)) {
        // A compaction carries the tool changes of the turns it replaced.
        for (const change of part.tool_changes) {
          apply(change);
        }
      }
    }
  }
  return [...offered.values()];
}

type AnthropicToolChange =
  | Anthropic.Beta.Messages.BetaRequestToolAdditionBlock
  | Anthropic.Beta.Messages.BetaRequestToolRemovalBlock;

/**
 * Identifies a tool for the tool changes: an MCP toolset by its server, any
 * other tool by its name, and a tool with neither by its position.
 */
function getToolKey(tool: unknown, index: number): string {
  if (tool != null && typeof tool === "object") {
    if ("type" in tool && tool.type === "mcp_toolset" && "mcp_server_name" in tool) {
      return `mcp_toolset:${String(tool.mcp_server_name)}`;
    }
    if ("name" in tool && typeof tool.name === "string") {
      return `name:${tool.name}`;
    }
  }
  return `index:${index}`;
}

/**
 * The key of the tool a tool change references, as `getToolKey` gives it.
 */
function getToolReferenceKey(tool: AnthropicToolChange["tool"] | undefined): string | undefined {
  if (tool?.type === "tool_reference") {
    return `name:${tool.name}`;
  }
  if (tool?.type === "mcp_toolset_reference") {
    return `mcp_toolset:${tool.server_name}`;
  }
  return undefined;
}

/**
 * Converts each tool definition into the `llm.tools` attributes: its JSON
 * schema, plus the name and description the spec defines next to it. Built
 * here rather than with core's `getLLMAttributes({ tools })`, which records
 * only the JSON schema.
 */
function getAnthropicToolsJSONSchema(body: MessageRequestParams): Attributes {
  return getAnthropicToolDefinitions(body).reduce((acc: Attributes, tool, index) => {
    const toolPrefix = `${SemanticConventions.LLM_TOOLS}.${index}.`;
    if (tool == null || typeof tool !== "object") {
      return acc;
    }
    const toolJsonSchema = safelyJSONStringify(tool);
    if (toolJsonSchema) {
      acc[`${toolPrefix}${SemanticConventions.TOOL_JSON_SCHEMA}`] = toolJsonSchema;
    }
    if ("name" in tool && typeof tool.name === "string" && tool.name) {
      acc[`${toolPrefix}${SemanticConventions.TOOL_NAME}`] = tool.name;
    }
    if ("description" in tool && typeof tool.description === "string" && tool.description) {
      acc[`${toolPrefix}${SemanticConventions.TOOL_DESCRIPTION}`] = tool.description;
    }
    return acc;
  }, {});
}

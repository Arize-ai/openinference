import type Anthropic from "@anthropic-ai/sdk";
import type { Attributes } from "@opentelemetry/api";
import { diag } from "@opentelemetry/api";

import type { TraceConfig } from "@arizeai/openinference-core";
import { REDACTED_VALUE, safelyJSONStringify } from "@arizeai/openinference-core";
import { SemanticConventions } from "@arizeai/openinference-semantic-conventions";

import { MODULE_NAME } from "../constants";
import type {
  MessageParam,
  ContentBlock,
  ContentBlockParam,
  ToolUseContentBlock,
  ImageBlockParam,
  OutputMessage,
} from "../types";
import { safelyGetAttributes } from "../utils";

/**
 * Summarizes a server-side fallback handoff so the boundary is visible on the
 * span. Without this the block would occupy an index in the flattened
 * `message_contents` list while contributing no attributes, leaving a hole in
 * the list and dropping which model declined and why.
 *
 * @see https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback#server-side-fallback
 */
function getAnthropicFallbackContentAttributes(
  prefix: string,
  block: Anthropic.Beta.Messages.BetaFallbackBlock | Anthropic.Beta.Messages.BetaFallbackBlockParam,
): Attributes {
  // The response block types `trigger` as a refusal trigger, but the param
  // variant echoed back on a later turn declares it `unknown` — the server
  // accepts and ignores any object there — so it has to be narrowed.
  const trigger: unknown = block.trigger;
  const category =
    typeof trigger === "object" && trigger !== null && "category" in trigger
      ? trigger.category
      : undefined;
  const reason = typeof category === "string" ? ` (refusal: ${category})` : "";
  return {
    [`${prefix}${SemanticConventions.MESSAGE_CONTENT_TYPE}`]: "fallback",
    [`${prefix}${SemanticConventions.MESSAGE_CONTENT_TEXT}`]: `${block.from?.model} -> ${block.to?.model}${reason}`,
  };
}

export function getAnthropicInputMessageAttributes(
  message: MessageParam,
  traceConfig: TraceConfig,
): Attributes {
  const role = message.role;
  const attributes: Attributes = {
    [SemanticConventions.MESSAGE_ROLE]: role,
  };

  // Add the content based on type
  if (typeof message.content === "string") {
    attributes[SemanticConventions.MESSAGE_CONTENT] = message.content;
  } else if (Array.isArray(message.content)) {
    let toolIndex = 0;
    message.content.forEach((part, index) => {
      const contentsIndexPrefix = `${SemanticConventions.MESSAGE_CONTENTS}.${index}.`;
      if (part == null || typeof part !== "object") {
        return;
      }
      if (part.type === "tool_result") {
        attributes[`${SemanticConventions.MESSAGE_TOOL_CALL_ID}`] = part.tool_use_id;
        if (typeof part.content === "string") {
          attributes[SemanticConventions.MESSAGE_CONTENT] = part.content;
        } else if (Array.isArray(part.content)) {
          // Handle complex tool result content, with its base64 data redacted
          const contentJSON = getRedactedBlockJSON(part.content, traceConfig);
          if (contentJSON != null) {
            attributes[SemanticConventions.MESSAGE_CONTENT] = contentJSON;
          }
        }
        return;
      }
      const block = getAnthropicContentBlockAttributes({
        block: part,
        contentPrefix: contentsIndexPrefix,
        toolCallPrefix: `${SemanticConventions.MESSAGE_TOOL_CALLS}.${toolIndex}.`,
        traceConfig,
      });
      Object.assign(attributes, block.attributes);
      if (block.isToolCall) {
        toolIndex++;
      }
    });
  }

  return attributes;
}

/**
 * Converts the Anthropic message result to LLM output attributes
 */
export function getAnthropicOutputMessagesAttributes(
  message: OutputMessage,
  traceConfig: TraceConfig,
): Attributes {
  const attributes: Attributes = {};
  const indexPrefix = `${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.`;

  attributes[`${indexPrefix}${SemanticConventions.MESSAGE_ROLE}`] = message.role;
  let toolIndex = 0;
  // Handle content array
  message.content.forEach((content, contentIndex) => {
    const block = getAnthropicContentBlockAttributes({
      block: content,
      contentPrefix: `${indexPrefix}${SemanticConventions.MESSAGE_CONTENTS}.${contentIndex}.`,
      toolCallPrefix: `${indexPrefix}${SemanticConventions.MESSAGE_TOOL_CALLS}.${toolIndex}.`,
      traceConfig,
    });
    Object.assign(attributes, block.attributes);
    if (block.isToolCall) {
      toolIndex++;
    }
  });

  return attributes;
}

/**
 * Maps one content block, of a request or of a response, to its
 * `message_content` attributes, and a tool call to its `message.tool_calls`
 * entry too. Server and MCP tool calls are recorded as tool calls, so every
 * call the model made is visible, whoever ran it.
 */
function getAnthropicContentBlockAttributes({
  block,
  contentPrefix,
  toolCallPrefix,
  traceConfig,
}: {
  block: ContentBlock | ContentBlockParam;
  contentPrefix: string;
  toolCallPrefix: string;
  traceConfig: TraceConfig;
}): { attributes: Attributes; isToolCall: boolean } {
  try {
    return mapAnthropicContentBlock({ block, contentPrefix, toolCallPrefix, traceConfig });
  } catch (error) {
    // A malformed block is recorded as its type and JSON, so it keeps its
    // index and the blocks after it are still mapped.
    diag.debug(`Failed to map a ${MODULE_NAME} content block`, error);
    return {
      attributes: safelyGetAttributes(() =>
        getAnthropicGenericContentAttributes(contentPrefix, block, traceConfig),
      ),
      isToolCall: false,
    };
  }
}

function mapAnthropicContentBlock({
  block,
  contentPrefix,
  toolCallPrefix,
  traceConfig,
}: {
  block: ContentBlock | ContentBlockParam;
  contentPrefix: string;
  toolCallPrefix: string;
  traceConfig: TraceConfig;
}): { attributes: Attributes; isToolCall: boolean } {
  if (isToolUseContentBlock(block)) {
    return {
      attributes: getAnthropicToolUseContentAttributes({ block, contentPrefix, toolCallPrefix }),
      isToolCall: true,
    };
  }
  const attributes: Attributes = {};
  if (block.type === "text") {
    attributes[`${contentPrefix}${SemanticConventions.MESSAGE_CONTENT_TYPE}`] = "text";
    attributes[`${contentPrefix}${SemanticConventions.MESSAGE_CONTENT_TEXT}`] = block.text;
  } else if (block.type === "image" && block.source.type !== "file") {
    // A Files API image has no URL, so it takes the generic mapping below.
    Object.assign(attributes, getAnthropicImageContentAttributes(contentPrefix, block));
  } else if (block.type === "thinking") {
    attributes[`${contentPrefix}${SemanticConventions.MESSAGE_CONTENT_TYPE}`] = "reasoning";
    attributes[`${contentPrefix}${SemanticConventions.MESSAGE_CONTENT_TEXT}`] = block.thinking;
    if (block.signature) {
      attributes[`${contentPrefix}${SemanticConventions.MESSAGE_CONTENT_SIGNATURE}`] =
        block.signature;
    }
  } else if (block.type === "redacted_thinking") {
    attributes[`${contentPrefix}${SemanticConventions.MESSAGE_CONTENT_TYPE}`] = "reasoning";
    attributes[`${contentPrefix}${SemanticConventions.MESSAGE_CONTENT_DATA}`] = block.data;
  } else if (block.type === "fallback") {
    // On input, a prior turn's fallback boundary echoed back to the API
    Object.assign(attributes, getAnthropicFallbackContentAttributes(contentPrefix, block));
  } else {
    Object.assign(
      attributes,
      getAnthropicOtherContentAttributes(contentPrefix, block, traceConfig),
    );
  }
  return { attributes, isToolCall: false };
}

/**
 * type-guard for the content blocks that call a tool: a client tool
 * (`tool_use`), a server tool (`server_tool_use`) or an MCP tool
 * (`mcp_tool_use`).
 */
export function isToolUseContentBlock(
  block: ContentBlock | ContentBlockParam,
): block is ToolUseContentBlock {
  return (
    block.type === "tool_use" || block.type === "server_tool_use" || block.type === "mcp_tool_use"
  );
}

/**
 * Records a tool call both as a `tool_use` content item and as an entry of
 * `message.tool_calls`. Streamed input is kept as the raw JSON received when
 * it could not be parsed, as for a stream the caller left early.
 */
function getAnthropicToolUseContentAttributes({
  block,
  contentPrefix,
  toolCallPrefix,
}: {
  block: ToolUseContentBlock;
  contentPrefix: string;
  toolCallPrefix: string;
}): Attributes {
  const input: unknown = block.input;
  const argumentsJSON = typeof input === "string" ? input : (safelyJSONStringify(input) ?? "");
  return {
    [`${toolCallPrefix}${SemanticConventions.TOOL_CALL_ID}`]: block.id,
    [`${toolCallPrefix}${SemanticConventions.TOOL_CALL_FUNCTION_NAME}`]: block.name,
    [`${toolCallPrefix}${SemanticConventions.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}`]: argumentsJSON,
    [`${contentPrefix}${SemanticConventions.MESSAGE_CONTENT_TYPE}`]: "tool_use",
    [`${contentPrefix}${SemanticConventions.TOOL_CALL_ID}`]: block.id,
    [`${contentPrefix}${SemanticConventions.TOOL_CALL_FUNCTION_NAME}`]: block.name,
    [`${contentPrefix}${SemanticConventions.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}`]: argumentsJSON,
  };
}

/**
 * Records an image input. A base64 image becomes a data URI (which trace
 * config can truncate or hide) and a URL image its URL. A Files API image has
 * no URL, so it is recorded as its JSON, see {@link getAnthropicContentBlockAttributes}.
 */
function getAnthropicImageContentAttributes(prefix: string, block: ImageBlockParam): Attributes {
  const imagePrefix = `${prefix}${SemanticConventions.MESSAGE_CONTENT_IMAGE}.`;
  const attributes: Attributes = {
    [`${prefix}${SemanticConventions.MESSAGE_CONTENT_TYPE}`]: "image",
  };
  const source = block.source;
  if (source.type === "base64") {
    attributes[`${imagePrefix}type`] = "base64";
    attributes[`${imagePrefix}media_type`] = source.media_type;
    attributes[`${imagePrefix}${SemanticConventions.IMAGE_URL}`] =
      `data:${source.media_type};base64,${source.data}`;
  } else if (source.type === "url") {
    attributes[`${imagePrefix}${SemanticConventions.IMAGE_URL}`] = source.url;
  }
  return attributes;
}

/**
 * Records a content block without a dedicated mapping. A compaction summary
 * is text the model reads in place of the conversation it replaced. Any other
 * block (server tool results, MCP results and listings, container uploads,
 * documents, tool changes, and types this version does not know) is recorded
 * as its type and its JSON, so it still occupies its index in the contents
 * list rather than leaving a hole. Embedded base64 data (a PDF document, a
 * nested image) is replaced by a placeholder before the block is encoded, and
 * nested images are hidden with `hideInputImages`, so the JSON never carries
 * what the image and size rules would mask.
 */
function getAnthropicOtherContentAttributes(
  prefix: string,
  block: ContentBlock | ContentBlockParam,
  traceConfig: TraceConfig,
): Attributes {
  if (block.type === "compaction") {
    const attributes: Attributes = {
      [`${prefix}${SemanticConventions.MESSAGE_CONTENT_TYPE}`]: "text",
    };
    if (block.content != null) {
      attributes[`${prefix}${SemanticConventions.MESSAGE_CONTENT_TEXT}`] = block.content;
    }
    if (block.signature != null) {
      attributes[`${prefix}${SemanticConventions.MESSAGE_CONTENT_SIGNATURE}`] = block.signature;
    }
    if (block.encrypted_content != null) {
      attributes[`${prefix}${SemanticConventions.MESSAGE_CONTENT_ENCRYPTED_CONTENT}`] =
        block.encrypted_content;
    }
    return attributes;
  }
  return getAnthropicGenericContentAttributes(prefix, block, traceConfig);
}

/**
 * Records a content block as its type and its redacted JSON.
 */
function getAnthropicGenericContentAttributes(
  prefix: string,
  block: ContentBlock | ContentBlockParam,
  traceConfig: TraceConfig,
): Attributes {
  const attributes: Attributes = {
    [`${prefix}${SemanticConventions.MESSAGE_CONTENT_TYPE}`]: String(block.type),
  };
  const blockJSON = getRedactedBlockJSON(block, traceConfig);
  if (blockJSON != null) {
    attributes[`${prefix}${SemanticConventions.MESSAGE_CONTENT_TEXT}`] = blockJSON;
  }
  return attributes;
}

/**
 * Encodes a content block as JSON with its embedded base64 data replaced by a
 * placeholder (keeping its media type), and its nested images replaced by a
 * placeholder when `hideInputImages` is set.
 */
function getRedactedBlockJSON(block: object, traceConfig: TraceConfig): string | undefined {
  try {
    return JSON.stringify(block, (_key, value: unknown) => {
      if (value == null || typeof value !== "object" || !("type" in value)) {
        return value;
      }
      if (value.type === "image" && traceConfig.hideInputImages) {
        return { type: "image", source: REDACTED_VALUE };
      }
      if (value.type === "base64" && "data" in value) {
        return { ...value, data: REDACTED_VALUE };
      }
      return value;
    });
  } catch (error) {
    diag.debug(`Failed to encode a ${MODULE_NAME} content block`, error);
    return undefined;
  }
}

import { safelyJSONStringify } from "@arizeai/openinference-core";
import { SemanticConventions } from "@arizeai/openinference-semantic-conventions";

import { Attributes } from "@opentelemetry/api";

import {
  Response as ResponseType,
  ResponseCreateParamsBase,
  ResponseInputItem,
  ResponseOutputItem,
  ResponseStreamEvent,
} from "openai/resources/responses/responses";
import { Stream } from "openai/streaming";

/**
 * Get attributes for responses api Items that are not typical messages with role
 * @param item - The item to get attributes for
 * @param prefix - The prefix to use for the attributes
 * @returns The attributes for the item
 */
function getResponseItemAttributes(
  item: Exclude<ResponseInputItem | ResponseOutputItem, { role: string }>,
  prefix = "",
): Attributes {
  const attributes: Attributes = {};
  // all items that are not typical messages with role
  // things like images, files, etc.
  const toolCallPrefix = `${prefix}${SemanticConventions.MESSAGE_TOOL_CALLS}.0.`;

  switch (item.type) {
    case "function_call": {
      attributes[`${prefix}${SemanticConventions.MESSAGE_ROLE}`] = "assistant";
      // now using tool call prefix to simulate multiple tool calls per message
      attributes[`${toolCallPrefix}${SemanticConventions.TOOL_CALL_ID}`] =
        item.call_id;
      attributes[
        `${toolCallPrefix}${SemanticConventions.TOOL_CALL_FUNCTION_NAME}`
      ] = item.name;
      attributes[
        `${toolCallPrefix}${SemanticConventions.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}`
      ] = item.arguments;
      break;
    }
    case "function_call_output": {
      attributes[`${prefix}${SemanticConventions.MESSAGE_ROLE}`] = "tool";
      attributes[`${prefix}${SemanticConventions.MESSAGE_TOOL_CALL_ID}`] =
        item.call_id;
      if (typeof item.output === "string") {
        attributes[`${prefix}${SemanticConventions.MESSAGE_CONTENT}`] =
          item.output;
      } else {
        // TODO(2410): figure out how to serialize the list of tools
        attributes[`${prefix}${SemanticConventions.MESSAGE_CONTENT}`] =
          safelyJSONStringify(item.output) || undefined;
      }

      break;
    }
    case "reasoning": {
      attributes[`${prefix}${SemanticConventions.MESSAGE_ROLE}`] = "assistant";
      item.summary.forEach((summaryItem, index) => {
        const summaryItemPrefix = `${prefix}${SemanticConventions.MESSAGE_CONTENTS}.${index}.`;
        if (summaryItem.type === "summary_text") {
          attributes[
            `${summaryItemPrefix}${SemanticConventions.MESSAGE_CONTENT_TYPE}`
          ] = "summary_text";
          attributes[
            `${summaryItemPrefix}${SemanticConventions.MESSAGE_CONTENT_TEXT}`
          ] = summaryItem.text;
        }
      });
      break;
    }
    case "item_reference": {
      break;
    }
    case "file_search_call": {
      if (!item.results) {
        // its a tool call
        attributes[`${prefix}${SemanticConventions.MESSAGE_ROLE}`] =
          "assistant";
        // now using tool call prefix to simulate multiple tool calls per message
        attributes[`${toolCallPrefix}${SemanticConventions.TOOL_CALL_ID}`] =
          item.id;
        attributes[
          `${toolCallPrefix}${SemanticConventions.TOOL_CALL_FUNCTION_NAME}`
        ] = item.type;
        attributes[
          `${toolCallPrefix}${SemanticConventions.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}`
        ] = JSON.stringify(item.queries);
      } else {
        // its a tool call output
        attributes[`${prefix}${SemanticConventions.MESSAGE_ROLE}`] = "tool";
        attributes[`${prefix}${SemanticConventions.MESSAGE_TOOL_CALL_ID}`] =
          item.id;
        attributes[`${prefix}${SemanticConventions.MESSAGE_CONTENT}`] =
          JSON.stringify(item.results);
      }
      break;
    }
    case "computer_call": {
      attributes[`${prefix}${SemanticConventions.MESSAGE_ROLE}`] = "assistant";
      // now using tool call prefix to simulate multiple tool calls per message
      attributes[`${toolCallPrefix}${SemanticConventions.TOOL_CALL_ID}`] =
        item.id;
      attributes[
        `${toolCallPrefix}${SemanticConventions.TOOL_CALL_FUNCTION_NAME}`
      ] = item.type;
      attributes[
        `${toolCallPrefix}${SemanticConventions.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}`
      ] = JSON.stringify(item.action);
      break;
    }
    case "computer_call_output": {
      attributes[`${prefix}${SemanticConventions.MESSAGE_ROLE}`] = "tool";
      attributes[`${prefix}${SemanticConventions.MESSAGE_TOOL_CALL_ID}`] =
        item.call_id;
      attributes[`${prefix}${SemanticConventions.MESSAGE_CONTENT}`] =
        JSON.stringify(item.output);
      break;
    }
    case "web_search_call": {
      attributes[`${prefix}${SemanticConventions.MESSAGE_ROLE}`] = "assistant";
      // now using tool call prefix to simulate multiple tool calls per message
      attributes[`${toolCallPrefix}${SemanticConventions.TOOL_CALL_ID}`] =
        item.id;
      attributes[
        `${toolCallPrefix}${SemanticConventions.TOOL_CALL_FUNCTION_NAME}`
      ] = item.type;
      // web search call does not share its arguments with the caller
      // it will show "undefined" in the arguments when traced
      break;
    }
  }
  return attributes;
}

/**
 * Get attributes for responses api Item, input or output
 * Non message items are detected and handled by {@link getResponseItemAttributes}
 * @param itemMessage - The message item to get attributes for
 * @param prefix - The prefix to use for the attributes
 * @returns The attributes for the message item
 */
function getResponseItemMessageAttributes(
  itemMessage: ResponseInputItem | ResponseOutputItem,
  prefix = "",
): Attributes {
  const message =
    typeof itemMessage === "string"
      ? ({ content: itemMessage, role: "user" } satisfies ResponseInputItem)
      : itemMessage;
  if (!("role" in message)) {
    return getResponseItemAttributes(message, prefix);
  }
  const role = message.role;
  const attributes: Attributes = {
    [`${prefix}${SemanticConventions.MESSAGE_ROLE}`]: role,
  };
  // add contents from message
  if (typeof message.content === "string") {
    attributes[`${prefix}${SemanticConventions.MESSAGE_CONTENT}`] =
      message.content;
  } else if (Array.isArray(message.content)) {
    message.content.forEach((part, index) => {
      const contentsIndexPrefix = `${prefix}${SemanticConventions.MESSAGE_CONTENTS}.${index}.`;
      if (part.type === "input_text") {
        attributes[
          `${contentsIndexPrefix}${SemanticConventions.MESSAGE_CONTENT_TYPE}`
        ] = "input_text";
        attributes[
          `${contentsIndexPrefix}${SemanticConventions.MESSAGE_CONTENT_TEXT}`
        ] = part.text;
      } else if (part.type === "output_text") {
        attributes[
          `${contentsIndexPrefix}${SemanticConventions.MESSAGE_CONTENT_TYPE}`
        ] = "output_text";
        attributes[
          `${contentsIndexPrefix}${SemanticConventions.MESSAGE_CONTENT_TEXT}`
        ] = part.text;
      } else if (part.type === "input_image") {
        attributes[
          `${contentsIndexPrefix}${SemanticConventions.MESSAGE_CONTENT_TYPE}`
        ] = "input_image";
        if (part.image_url) {
          attributes[
            `${contentsIndexPrefix}${SemanticConventions.MESSAGE_CONTENT_IMAGE}.${SemanticConventions.IMAGE_URL}`
          ] = part.image_url;
        }
      } else if (part.type === "input_file") {
        // TODO: Handle input file
      } else if (part.type === "refusal") {
        attributes[
          `${contentsIndexPrefix}${SemanticConventions.MESSAGE_CONTENT_TYPE}`
        ] = "refusal";
        attributes[
          `${contentsIndexPrefix}${SemanticConventions.MESSAGE_CONTENT_TEXT}`
        ] = part.refusal;
      }
    });
  }
  // add role based fields to attributes
  // as of now, there are no role based fields as they are handled by alternatively typed
  // items
  switch (message.role) {
    case "user": {
      break;
    }
    case "assistant": {
      break;
    }
    case "system": {
      break;
    }
    case "developer": {
      break;
    }
  }
  return attributes;
}

export function getResponsesInputMessagesAttributes(
  body: ResponseCreateParamsBase,
): Attributes {
  const attributes: Attributes = {};
  const items: ResponseInputItem[] = [];
  if (body.instructions) {
    items.push({ content: body.instructions, role: "system" });
  }
  if (typeof body.input === "string") {
    items.push({ content: body.input, role: "user" });
  } else {
    items.push(...(body.input ?? []));
  }
  items.forEach((item, index) => {
    const indexPrefix = `${SemanticConventions.LLM_INPUT_MESSAGES}.${index}.`;
    const itemAttributes = getResponseItemMessageAttributes(item, indexPrefix);
    Object.entries(itemAttributes).forEach(([key, value]) => {
      attributes[key] = value;
    });
  });

  return attributes;
}

export function getResponsesUsageAttributes(
  response: ResponseType,
): Attributes {
  if (response.usage) {
    return {
      [SemanticConventions.LLM_TOKEN_COUNT_COMPLETION]:
        response.usage.output_tokens,
      [SemanticConventions.LLM_TOKEN_COUNT_PROMPT]: response.usage.input_tokens,
      [SemanticConventions.LLM_TOKEN_COUNT_TOTAL]: response.usage.total_tokens,
      [SemanticConventions.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ]:
        response.usage.input_tokens_details?.cached_tokens,
      [SemanticConventions.LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING]:
        response.usage.output_tokens_details?.reasoning_tokens,
      // no audio tokens for response inputs or outputs
    };
  }
  return {};
}

export function getResponsesOutputMessagesAttributes(
  response: ResponseType,
): Attributes {
  const attributes: Attributes = {};
  const items = response.output;
  items.forEach((item, index) => {
    const indexPrefix = `${SemanticConventions.LLM_OUTPUT_MESSAGES}.${index}.`;
    const itemAttributes = getResponseItemMessageAttributes(item, indexPrefix);
    Object.entries(itemAttributes).forEach(([key, value]) => {
      attributes[key] = value;
    });
  });

  return attributes;
}

export async function consumeResponseStreamEvents(
  stream: Stream<ResponseStreamEvent>,
): Promise<ResponseType | undefined> {
  let response: ResponseType | undefined;

  for await (const event of stream) {
    switch (event.type) {
      case "response.completed": {
        response = event.response;
        break;
      }
    }
  }

  return response;
}

/**
 * Copyright 2025 IBM Corp.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import { diag } from "@opentelemetry/api";
import { BaseAgent } from "beeai-framework/agents/base";
import { ChatModel } from "beeai-framework/backend/chat";
import type { EventMeta } from "beeai-framework/emitter/types";
import { getProp } from "beeai-framework/internals/helpers/object";
import { Serializable } from "beeai-framework/internals/serializable";
import { Tool } from "beeai-framework/tools/base";

import { isObjectWithStringKeys } from "@arizeai/openinference-core";
import {
  LLMAttributePostfixes,
  MessageAttributePostfixes,
  MimeType,
  OpenInferenceSpanKind,
  SemanticAttributePrefixes,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";

import {
  errorEventName,
  errorLLMEventName,
  errorToolEventName,
  finishLLMEventName,
  finishToolEventName,
  newTokenLLMEventName,
  partialUpdateEventName,
  retryEventName,
  retryToolEventName,
  startEventName,
  startLLMEventName,
  startToolEventName,
  successEventName,
  successLLMEventName,
  successToolEventName,
  toolErrorEventName,
  toolStartEventName,
  toolSuccessEventName,
  updateEventName,
} from "../config";

function getMessageParts(message: unknown): { role?: string; text: string } {
  if (!isObjectWithStringKeys(message)) return { text: "" };
  const role = typeof message.role === "string" ? message.role : undefined;
  const content = Array.isArray(message.content) ? message.content : [];
  const text = content
    .filter(isObjectWithStringKeys)
    .filter((part) => part.type === "text" && typeof part.text === "string")
    .map((part) => String(part.text))
    .join("");
  return { role, text };
}

function parserLLMInputMessages(messages: readonly unknown[]) {
  return messages.reduce((acc: Record<string, string>, item, key) => {
    const { role, text } = getMessageParts(item);
    if (role != null) {
      acc[
        `${SemanticAttributePrefixes.llm}.${LLMAttributePostfixes.input_messages}.${key}.${SemanticAttributePrefixes.message}.${MessageAttributePostfixes.role}`
      ] = role;
    }
    acc[
      `${SemanticAttributePrefixes.llm}.${LLMAttributePostfixes.input_messages}.${key}.${SemanticAttributePrefixes.message}.${MessageAttributePostfixes.content}`
    ] = text;
    return acc;
  }, {});
}

function parseLLMOutputMessages(messages: readonly unknown[]) {
  return messages.reduce((acc: Record<string, string>, item, key) => {
    const { role, text } = getMessageParts(item);
    if (role != null) {
      acc[
        `${SemanticAttributePrefixes.llm}.${LLMAttributePostfixes.output_messages}.${key}.${SemanticAttributePrefixes.message}.${MessageAttributePostfixes.role}`
      ] = role;
    }
    acc[
      `${SemanticAttributePrefixes.llm}.${LLMAttributePostfixes.output_messages}.${key}.${SemanticAttributePrefixes.message}.${MessageAttributePostfixes.content}`
    ] = text;
    return acc;
  }, {});
}

const matchesEvent = (name: string, events: readonly string[]): boolean => events.includes(name);

export function getSerializedObjectSafe(dataObject: unknown, meta: EventMeta<unknown>) {
  try {
    // agent events
    if (
      matchesEvent(meta.name, [startEventName, successEventName, errorEventName, retryEventName]) &&
      meta.creator instanceof BaseAgent
    ) {
      const event = isObjectWithStringKeys(dataObject) ? dataObject : {};
      const agentMeta = isObjectWithStringKeys(event.meta) ? event.meta : undefined;
      const tools = Array.isArray(event.tools) ? event.tools.filter(isObjectWithStringKeys) : [];
      const memory = isObjectWithStringKeys(event.memory) ? event.memory : undefined;
      const messages = memory && Array.isArray(memory.messages) ? memory.messages : [];
      const error = event.error instanceof Error ? event.error : undefined;
      const data = event.data;
      return {
        [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.AGENT,
        ...(typeof agentMeta?.iteration === "number" && { iteration: agentMeta.iteration }),
        ...(tools?.length > 0 && {
          [SemanticConventions.LLM_TOOLS]: tools.map((tool) => ({
            [SemanticConventions.TOOL_NAME]: typeof tool.name === "string" ? tool.name : undefined,
            [SemanticConventions.TOOL_DESCRIPTION]:
              typeof tool.description === "string" ? tool.description : undefined,
            "tool.options": tool.options,
          })),
        }),
        ...(messages.length > 0 && {
          [SemanticConventions.INPUT_MIME_TYPE]: MimeType.JSON,
          [SemanticConventions.INPUT_VALUE]: JSON.stringify(messages),
        }),
        ...(error && {
          "exception.message": error.message,
          "exception.stacktrace": error.stack,
          "exception.type": error.name,
        }),
        ...(data != null
          ? {
              [SemanticConventions.OUTPUT_MIME_TYPE]: MimeType.JSON,
              [SemanticConventions.OUTPUT_VALUE]: JSON.stringify(data),
            }
          : {}),
      };
    }

    // update events
    if (matchesEvent(meta.name, [updateEventName, partialUpdateEventName])) {
      const event = isObjectWithStringKeys(dataObject) ? dataObject : {};
      const data = isObjectWithStringKeys(event.data) ? event.data : {};

      const output = data.final_answer || data.tool_output;
      return {
        [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.AGENT,
        ...(typeof data.thought === "string" && { thought: data.thought }),
        ...(typeof data.tool_name === "string" && {
          [SemanticConventions.TOOL_NAME]: data.tool_name,
        }),
        ...(data.tool_input != null
          ? { [SemanticConventions.TOOL_PARAMETERS]: JSON.stringify(data.tool_input) }
          : {}),
        ...(typeof output === "string" && {
          [SemanticConventions.OUTPUT_MIME_TYPE]: MimeType.JSON,
          [SemanticConventions.OUTPUT_VALUE]: output,
        }),
      };
    }

    // tool events (from agent)
    if (matchesEvent(meta.name, [toolErrorEventName, toolStartEventName, toolSuccessEventName])) {
      const event = isObjectWithStringKeys(dataObject) ? dataObject : {};
      const data = isObjectWithStringKeys(event.data) ? event.data : {};
      const iteration = isObjectWithStringKeys(data.iteration) ? data.iteration : undefined;
      const tool = isObjectWithStringKeys(data.tool) ? data.tool : undefined;
      const error = data.error instanceof Error ? data.error : undefined;
      const output = data.result instanceof Serializable ? data.result.createSnapshot() : undefined;

      return {
        [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.TOOL,
        ...(typeof iteration?.thought === "string" && { thought: iteration.thought }),
        ...(data?.input
          ? {
              [SemanticConventions.TOOL_PARAMETERS]: JSON.stringify(data.input),
            }
          : {}),
        ...(typeof tool?.description === "string" && {
          [SemanticConventions.TOOL_DESCRIPTION]: tool.description,
        }),
        ...(typeof tool?.name === "string" && {
          [SemanticConventions.TOOL_NAME]: tool.name,
        }),
        ...(error && {
          "exception.message": error.message,
          "exception.stacktrace": error.stack,
          "exception.type": error.name,
        }),
        ...(output != null ? { [SemanticConventions.OUTPUT_VALUE]: JSON.stringify(output) } : {}),
      };
    }
    // tool events native
    if (
      [
        startToolEventName,
        successToolEventName,
        finishToolEventName,
        errorToolEventName,
        retryToolEventName,
      ].some((eventName) => eventName === meta.name) &&
      meta.creator instanceof Tool
    ) {
      if (!dataObject) {
        return {
          [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.TOOL,
        };
      }
      const event = isObjectWithStringKeys(dataObject) ? dataObject : {};
      const { input, output } = event;
      const error = event.error instanceof Error ? event.error : undefined;

      return {
        [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.TOOL,
        ...(input != null ? { [SemanticConventions.TOOL_PARAMETERS]: JSON.stringify(input) } : {}),
        ...(output != null ? { [SemanticConventions.OUTPUT_VALUE]: JSON.stringify(output) } : {}),
        ...(error && {
          "exception.message": error.message,
          "exception.stacktrace": error.stack,
          "exception.type": error.name,
        }),
      };
    }

    // llm events
    if (
      matchesEvent(meta.name, [
        successLLMEventName,
        startLLMEventName,
        errorLLMEventName,
        newTokenLLMEventName,
      ]) &&
      meta.creator instanceof ChatModel
    ) {
      const event = isObjectWithStringKeys(dataObject) ? dataObject : {};
      const value = isObjectWithStringKeys(event.value) ? event.value : undefined;
      const input = isObjectWithStringKeys(event.input) ? event.input : undefined;
      const usage = value && isObjectWithStringKeys(value.usage) ? value.usage : undefined;
      const inputMessages = input && Array.isArray(input.messages) ? input.messages : [];
      const outputMessages = value && Array.isArray(value.messages) ? value.messages : [];
      const error = event.error instanceof Error ? event.error : undefined;

      const creatorSnapshot = meta.creator.createSnapshot();
      const creator: Record<string, unknown> = isObjectWithStringKeys(creatorSnapshot)
        ? creatorSnapshot
        : {};
      return {
        [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.LLM,
        ...(typeof usage?.completionTokens === "number" && {
          [SemanticConventions.LLM_TOKEN_COUNT_COMPLETION]: usage.completionTokens,
        }),
        ...(typeof usage?.promptTokens === "number" && {
          [SemanticConventions.LLM_TOKEN_COUNT_PROMPT]: usage.promptTokens,
        }),
        ...(typeof usage?.totalTokens === "number" && {
          [SemanticConventions.LLM_TOKEN_COUNT_TOTAL]: usage.totalTokens,
        }),
        ...(inputMessages.length > 0 && {
          [SemanticConventions.INPUT_MIME_TYPE]: MimeType.JSON,
          [SemanticConventions.INPUT_VALUE]: JSON.stringify(inputMessages),
          ...parserLLMInputMessages(inputMessages),
        }),
        ...(outputMessages.length > 0 && {
          [SemanticConventions.OUTPUT_MIME_TYPE]: MimeType.JSON,
          [SemanticConventions.OUTPUT_VALUE]: JSON.stringify(outputMessages),
          ...parseLLMOutputMessages(outputMessages),
        }),
        ...(error && {
          "exception.message": error.message,
          "exception.stacktrace": error.stack,
          "exception.type": error.name,
        }),
        ...(typeof creator.providerId === "string" && {
          [SemanticConventions.LLM_PROVIDER]: creator.providerId,
          [`${SemanticAttributePrefixes.metadata}.${LLMAttributePostfixes.provider}`]:
            creator.providerId,
        }),
        ...(typeof creator.modelId === "string" && {
          [SemanticConventions.LLM_MODEL_NAME]: creator.modelId,
          [`${SemanticAttributePrefixes.metadata}.${LLMAttributePostfixes.model_name}`]:
            creator.modelId,
        }),
        ...(creator.parameters != null
          ? {
              [SemanticConventions.LLM_INVOCATION_PARAMETERS]: JSON.stringify(creator.parameters),
            }
          : {}),
      };
    }
    if (meta.name === finishLLMEventName && meta.creator instanceof ChatModel) {
      return {
        [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.LLM,
      };
    }

    // other events
    const data = getProp(dataObject, ["data"], dataObject);
    if (data instanceof Serializable) {
      return { data: JSON.stringify(data.createSnapshot()), test: "hallo" };
    }

    return { data: JSON.stringify(data) };
  } catch (e) {
    diag.warn("Failed to parse event data", e);
    return null;
  }
}

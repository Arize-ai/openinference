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

import { Serializable } from "beeai-framework/internals/serializable";
import { instrumentationLogger } from "beeai-framework/instrumentation/logger";
import { getProp } from "beeai-framework/internals/helpers/object";
import { EventMeta, InferCallbackValue } from "beeai-framework/emitter/types";
import { BeeCallbacks } from "beeai-framework/agents/bee/types";
import { ChatModel, ChatModelEvents } from "beeai-framework/backend/chat";
import { BaseAgent } from "beeai-framework/agents/base";
import { Message, MessageContentPart } from "beeai-framework/backend/message";
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
import {
  LLMAttributePostfixes,
  MessageAttributePostfixes,
  MimeType,
  OpenInferenceSpanKind,
  SemanticAttributePrefixes,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";
import { Tool, ToolEvents } from "beeai-framework/tools/base";

function parserLLMInputMessages(
  messages: readonly Message<MessageContentPart, string>[],
) {
  return messages.reduce(
    (acc, item, key) => ({
      ...acc,
      [`${SemanticAttributePrefixes.llm}.${LLMAttributePostfixes.input_messages}.${key}.${SemanticAttributePrefixes.message}.${MessageAttributePostfixes.role}`]:
        item.role,
      [`${SemanticAttributePrefixes.llm}.${LLMAttributePostfixes.input_messages}.${key}.${SemanticAttributePrefixes.message}.${MessageAttributePostfixes.content}`]:
        item.content
          .filter((c) => c.type === "text")
          .map((c) => c.text)
          .join(""),
    }),
    {},
  );
}

function parseLLMOutputMessages(
  messages: readonly Message<MessageContentPart, string>[],
) {
  return messages.reduce(
    (acc, item, key) => ({
      ...acc,
      [`${SemanticAttributePrefixes.llm}.${LLMAttributePostfixes.output_messages}.${key}.${SemanticAttributePrefixes.message}.${MessageAttributePostfixes.role}`]:
        item.role,
      [`${SemanticAttributePrefixes.llm}.${LLMAttributePostfixes.output_messages}.${key}.${SemanticAttributePrefixes.message}.${MessageAttributePostfixes.content}`]:
        item.content
          .filter((c) => c.type === "text")
          .map((c) => c.text)
          .join(""),
    }),
    {},
  );
}

export function getSerializedObjectSafe(
  dataObject: unknown,
  meta: EventMeta<unknown>,
) {
  // agent events
  if (
    [startEventName, successEventName, errorEventName, retryEventName].includes(
      meta.name as keyof BeeCallbacks,
    ) &&
    meta.creator instanceof BaseAgent
  ) {
    const { meta, tools, memory, error, data } =
      dataObject as InferCallbackValue<
        | BeeCallbacks["start"]
        | BeeCallbacks["error"]
        | BeeCallbacks["success"]
        | BeeCallbacks["retry"]
      >;
    return {
      [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
        OpenInferenceSpanKind.AGENT,
      iteration: meta.iteration,
      ...(tools?.length > 0 && {
        [SemanticConventions.LLM_TOOLS]: tools.map((tool) => ({
          [SemanticConventions.TOOL_NAME]: tool.name,
          [SemanticConventions.TOOL_DESCRIPTION]: tool.description,
          "tool.options": tool.options,
        })),
      }),
      ...(memory.messages.length > 0 && {
        [SemanticConventions.INPUT_MIME_TYPE]: MimeType.JSON,
        [SemanticConventions.INPUT_VALUE]: JSON.stringify(memory.messages),
      }),
      ...(error && {
        "exception.message": error.message,
        "exception.stacktrace": error.stack,
        "exception.type": error.name,
      }),
      ...(data && {
        [SemanticConventions.OUTPUT_MIME_TYPE]: MimeType.JSON,
        [SemanticConventions.OUTPUT_VALUE]: JSON.stringify(data),
      }),
    };
  }

  // update events
  if (
    [updateEventName, partialUpdateEventName].includes(
      meta.name as keyof BeeCallbacks,
    )
  ) {
    const { data } = dataObject as InferCallbackValue<
      BeeCallbacks["partialUpdate"] | BeeCallbacks["update"]
    >;

    const output = data.final_answer || data.tool_output;
    return {
      [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
        OpenInferenceSpanKind.AGENT,
      thought: data.thought,
      ...(data.tool_name && {
        [SemanticConventions.TOOL_NAME]: data.tool_name,
      }),
      ...(data.tool_input && {
        [SemanticConventions.TOOL_PARAMETERS]: JSON.stringify(data.tool_input),
      }),
      ...(output && {
        [SemanticConventions.OUTPUT_MIME_TYPE]: MimeType.JSON,
        [SemanticConventions.OUTPUT_VALUE]: output,
      }),
    };
  }

  // tool events (from agent)
  if (
    [toolErrorEventName, toolStartEventName, toolSuccessEventName].includes(
      meta.name as keyof BeeCallbacks,
    )
  ) {
    const { data } = dataObject as InferCallbackValue<
      | BeeCallbacks["toolError"]
      | BeeCallbacks["toolStart"]
      | BeeCallbacks["toolSuccess"]
    >;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const output: any = data.result && data.result.createSnapshot();

    return {
      [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.TOOL,
      thought: data.iteration.thought,
      ...(data.input
        ? { [SemanticConventions.TOOL_PARAMETERS]: JSON.stringify(data.input) }
        : {}),
      ...(data.tool.description && {
        [SemanticConventions.TOOL_DESCRIPTION]: data.tool.description,
      }),
      ...(data.tool.name && {
        [SemanticConventions.TOOL_NAME]: data.tool.name,
      }),
      ...(data.error && {
        "exception.message": data.error.message,
        "exception.stacktrace": data.error.stack,
        "exception.type": data.error.name,
      }),
      ...(output && {
        [SemanticConventions.OUTPUT_VALUE]: JSON.stringify(output),
      }),
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
    ].includes(meta.name as keyof ToolEvents) &&
    meta.creator instanceof Tool
  ) {
    if (!dataObject) {
      return {
        [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
          OpenInferenceSpanKind.TOOL,
      };
    }
    const { input, output, error } = dataObject as InferCallbackValue<
      | ToolEvents["start"]
      | ToolEvents["success"]
      | ToolEvents["retry"]
      | ToolEvents["error"]
    >;

    return {
      [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.TOOL,
      ...(input
        ? { [SemanticConventions.TOOL_PARAMETERS]: JSON.stringify(input) }
        : {}),
      ...(output && {
        [SemanticConventions.OUTPUT_VALUE]: JSON.stringify(output),
      }),
      ...(error && {
        "exception.message": error.message,
        "exception.stacktrace": error.stack,
        "exception.type": error.name,
      }),
    };
  }

  // llm events
  if (
    [
      successLLMEventName,
      startLLMEventName,
      errorLLMEventName,
      newTokenLLMEventName,
    ].includes(meta.name as keyof ChatModelEvents) &&
    meta.creator instanceof ChatModel
  ) {
    const { value, input, error } = dataObject as InferCallbackValue<
      | ChatModelEvents["success"]
      | ChatModelEvents["start"]
      | ChatModelEvents["error"]
      | ChatModelEvents["newToken"]
    >;

    const creator = meta.creator.createSnapshot();
    return {
      [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.LLM,
      ...(value?.usage?.completionTokens && {
        [SemanticConventions.LLM_TOKEN_COUNT_COMPLETION]:
          value?.usage?.completionTokens,
      }),
      ...(value?.usage?.promptTokens && {
        [SemanticConventions.LLM_TOKEN_COUNT_PROMPT]:
          value?.usage?.promptTokens,
      }),
      ...(value?.usage?.totalTokens && {
        [SemanticConventions.LLM_TOKEN_COUNT_TOTAL]: value?.usage?.totalTokens,
      }),
      ...(input?.messages.length > 0 && {
        [SemanticConventions.INPUT_MIME_TYPE]: MimeType.JSON,
        [SemanticConventions.INPUT_VALUE]: JSON.stringify(input.messages),
        ...parserLLMInputMessages(input.messages),
      }),
      ...(value?.messages.length > 0 && {
        [SemanticConventions.OUTPUT_MIME_TYPE]: MimeType.JSON,
        [SemanticConventions.OUTPUT_VALUE]: JSON.stringify(value.messages),
        ...parseLLMOutputMessages(value.messages),
      }),
      ...(error && {
        "exception.message": error.message,
        "exception.stacktrace": error.stack,
        "exception.type": error.name,
      }),
      ...("providerId" in creator && {
        [SemanticConventions.LLM_PROVIDER]: creator.providerId,
        [`${SemanticAttributePrefixes.metadata}.${LLMAttributePostfixes.provider}`]:
          creator.providerId,
      }),
      ...("modelId" in creator && {
        [SemanticConventions.LLM_MODEL_NAME]: creator.modelId,
        [`${SemanticAttributePrefixes.metadata}.${LLMAttributePostfixes.model_name}`]:
          creator.modelId,
      }),
      ...(creator.parameters && {
        [SemanticConventions.LLM_INVOCATION_PARAMETERS]: JSON.stringify(
          creator.parameters,
        ),
      }),
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
    try {
      return { data: JSON.stringify(data.createSnapshot()), test: "hallo" };
    } catch (e) {
      instrumentationLogger.warn(
        e,
        "Invalid createSnapshot method in the Serializable class",
      );
      return null;
    }
  }

  return { data: JSON.stringify(data) };
}

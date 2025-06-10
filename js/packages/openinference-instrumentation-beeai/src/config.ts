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

import type { ReActAgentCallbacks } from "beeai-framework/agents/react/types";
import { ChatModelEvents } from "beeai-framework/backend/chat";
import { ToolEvents } from "beeai-framework/tools/base";

export const INSTRUMENTATION_IGNORED_KEYS = process.env
  .BEE_FRAMEWORK_INSTRUMENTATION_IGNORED_KEYS
  ? process.env.BEE_FRAMEWORK_INSTRUMENTATION_IGNORED_KEYS.split(",").filter(
      Boolean,
    )
  : [];

export const partialUpdateEventName: keyof ReActAgentCallbacks =
  "partialUpdate";
export const updateEventName: keyof ReActAgentCallbacks = "update";
export const toolErrorEventName: keyof ReActAgentCallbacks = "toolError";
export const toolStartEventName: keyof ReActAgentCallbacks = "toolStart";
export const toolSuccessEventName: keyof ReActAgentCallbacks = "toolSuccess";
export const startEventName: keyof ReActAgentCallbacks = "start";
export const successEventName: keyof ReActAgentCallbacks = "success";
export const errorEventName: keyof ReActAgentCallbacks = "error";
export const retryEventName: keyof ReActAgentCallbacks = "retry";

export const newTokenLLMEventName: keyof ChatModelEvents = `newToken`;
export const successLLMEventName: keyof ChatModelEvents = `success`;
export const startLLMEventName: keyof ChatModelEvents = `start`;
export const errorLLMEventName: keyof ChatModelEvents = `error`;
export const finishLLMEventName: keyof ChatModelEvents = `finish`;

export const startToolEventName: keyof ToolEvents = "start";
export const successToolEventName: keyof ToolEvents = "success";
export const finishToolEventName: keyof ToolEvents = "finish";
export const retryToolEventName: keyof ToolEvents = "retry";
export const errorToolEventName: keyof ToolEvents = "error";

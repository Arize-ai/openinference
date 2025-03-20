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

import { OITracer } from "@arizeai/openinference-core";
import { getSerializedObjectSafe } from "./helpers/getSerializedObjectSafe";
import { createSpan } from "./helpers/create-span";
import { IdNameManager } from "./helpers/idNameManager";
import { getErrorSafe } from "./helpers/getErrorSafe";
import { findLast, isEmpty } from "remeda";
import type { ReActAgentCallbacks } from "beeai-framework/agents/react/types";
import type { InferCallbackValue } from "beeai-framework/emitter/types";
import { FrameworkError } from "beeai-framework/errors";
import { Version } from "beeai-framework/version";
import { Role } from "beeai-framework/backend/message";
import type { GetRunContext, RunInstance } from "beeai-framework/context";
import { traceSerializer } from "./helpers/traceSerializer";
import {
  errorLLMEventName,
  finishLLMEventName,
  INSTRUMENTATION_IGNORED_KEYS,
  newTokenLLMEventName,
  partialUpdateEventName,
  successLLMEventName,
} from "./config";
import { createFullPath } from "beeai-framework/emitter/utils";
import type { ReActAgent } from "beeai-framework/agents/react/agent";
import { BaseAgent } from "beeai-framework/agents/base";
import { diag } from "@opentelemetry/api";
import { FrameworkSpan, GeneratedResponse } from "./types";
import { buildTraceTree } from "./helpers/buildTraceTree";
import {
  OpenInferenceSpanKind,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";

export const activeTracesMap = new Map<string, string>();

/**
 * First we collect, filter and transform events from the beeai emitter to our internal framework spans in the `emitter.match("*.*"` block
 * Some special data are collected in other `emitter.match` blocks like (history, response, rawPromt)
 *
 * Then we create the open telemetry spans when all data are collected. We are retroactively deleting some unnecessary internal spans
 * see "emitter.match((event) => event.path === `${basePath}.run.${finishEventName}`" section
 */
export function createTelemetryMiddleware(
  tracer: OITracer,
  mainSpanKind: OpenInferenceSpanKind,
) {
  return (context: GetRunContext<RunInstance, unknown>) => {
    if (!context.emitter?.trace?.id) {
      throw new FrameworkError(`Fatal error. Missing traceId`, [], { context });
    }

    const traceId = context.emitter?.trace?.id;
    if (activeTracesMap.has(traceId)) {
      return;
    }
    activeTracesMap.set(traceId, context.instance.constructor.name);

    diag.debug("createTelemetryMiddleware", {
      source: context.instance.constructor.name,
      traceId: traceId,
    });
    const { emitter, runParams, instance } = context;
    const basePath = createFullPath(emitter.namespace, "");

    let prompt: string | undefined | null = null;
    if (instance instanceof BaseAgent) {
      prompt = (runParams as Parameters<ReActAgent["run"]>)[0].prompt;
    }

    const spansMap = new Map<string, FrameworkSpan>();
    const parentIdsMap = new Map<string, number>();
    const spansToDeleteMap = new Map<string, undefined>();

    let generatedMessage: GeneratedResponse | undefined = undefined;
    let history: GeneratedResponse[] | undefined = undefined;
    const groupIterations: string[] = [];

    const idNameManager = new IdNameManager();

    const eventsIterationsMap = new Map<string, Map<string, string>>();

    const startTimeDate = new Date().getTime();
    const startTimePerf = performance.now();
    function convertDateToPerformance(date: Date) {
      return date.getTime() - startTimeDate + startTimePerf;
    }

    const serializer = traceSerializer({
      ignored_keys: INSTRUMENTATION_IGNORED_KEYS,
    });

    function cleanSpanSources({ spanId }: { spanId: string }) {
      const parentId = spansMap.get(spanId)?.parent_id;
      if (!parentId) {
        return;
      }

      const spanCount = parentIdsMap.get(parentId);
      if (!spanCount) {
        return;
      }

      if (spanCount > 1) {
        // increase the span count for the parentId
        parentIdsMap.set(parentId, spanCount - 1);
      } else if (spanCount === 1) {
        parentIdsMap.delete(parentId);
        // check the `spansToDelete` if the span should be deleted when it has no children's
        if (spansToDeleteMap.has(parentId)) {
          spansMap.delete(parentId);
          spansToDeleteMap.delete(parentId);
        }
      }
    }

    /**
     * Create OpenTelemetry spans from collected data
     */
    emitter.match(
      (event) => event.path === `${basePath}.run.${finishLLMEventName}`,
      async (_, meta) => {
        try {
          diag.debug("run finish event", { path: meta.path, traceId: traceId });

          if (!prompt && instance instanceof BaseAgent) {
            prompt = findLast(
              instance.memory.messages,
              (message) => message.role === Role.USER,
            )?.text;

            if (!prompt) {
              throw new FrameworkError(
                "The prompt must be defined for the Agent's run",
                [],
                {
                  context,
                },
              );
            }
          }

          // create tracer spans from collected data
          buildTraceTree({
            tracer,
            mainSpanKind,
            data: {
              prompt: prompt,
              history,
              generatedMessage,
              spans: JSON.parse(serializer(Array.from(spansMap.values()))),
              traceId,
              version: Version,
              runErrorSpanKey: `${basePath}.run.${errorLLMEventName}`,
              startTime: startTimePerf,
              endTime: performance.now(),
              source: activeTracesMap.get(traceId)!,
            },
          });
        } catch (e) {
          diag.warn("Instrumentation error", e);
        } finally {
          activeTracesMap.delete(traceId);
        }
      },
    );

    /**
     * This block collects all "not run category" events with their data and prepares spans for the OpenTelemetry.
     * The huge number of `newToken` events are skipped and only the last one for each parent event is saved because of `generated_token_count` information
     * The framework event tree structure is different from the open-telemetry tree structure and must be transformed from groupId and parentGroupId pattern via idNameManager
     * The artificial "iteration" main tree level is computed from the `meta.groupId`
     */
    emitter.match("*.*", (data, meta) => {
      // allow `run.error` event due to the runtime error information
      if (
        meta.path.includes(".run.") &&
        meta.path !== `${basePath}.run.${errorLLMEventName}`
      ) {
        return;
      }
      // skip all new token events
      if (meta.name === newTokenLLMEventName) {
        return;
      }
      if (!meta.trace?.runId) {
        throw new FrameworkError(
          `Fatal error. Missing runId for event: ${meta.path}`,
          [],
          {
            context,
          },
        );
      }

      /**
       * create groupId span level (id does not exist)
       * I use only the top-level groups like iterations other nested groups like tokens would introduce unuseful complexity
       */
      if (
        meta.groupId &&
        !meta.trace.parentRunId &&
        !groupIterations.includes(meta.groupId)
      ) {
        spansMap.set(
          meta.groupId,
          createSpan({
            id: meta.groupId,
            name: meta.groupId,
            target: "groupId",
            data: {
              [SemanticConventions.OPENINFERENCE_SPAN_KIND]: "Chain",
            },
            startedAt: convertDateToPerformance(meta.createdAt),
          }),
        );
        groupIterations.push(meta.groupId);
      }

      const { spanId, parentSpanId } = idNameManager.getIds({
        path: meta.path,
        id: meta.id,
        runId: meta.trace.runId,
        parentRunId: meta.trace.parentRunId,
        groupId: meta.groupId,
      });

      const serializedData = getSerializedObjectSafe(data, meta);

      // skip partialUpdate events with no data
      if (meta.name === partialUpdateEventName && isEmpty(serializedData)) {
        return;
      }

      const span = createSpan({
        id: spanId,
        name: meta.name,
        target: meta.path,
        ...(parentSpanId && { parent: { id: parentSpanId } }),
        ctx: meta.context,
        data: serializedData,
        error: getErrorSafe(data),
        startedAt: convertDateToPerformance(meta.createdAt),
      });

      const lastIteration = groupIterations[groupIterations.length - 1];

      // delete the `partialUpdate` event if does not have nested spans
      const lastIterationEventSpanId = eventsIterationsMap
        .get(lastIteration)
        ?.get(meta.name);
      if (
        lastIterationEventSpanId &&
        partialUpdateEventName === meta.name &&
        spansMap.has(lastIterationEventSpanId)
      ) {
        const { context } = spansMap.get(lastIterationEventSpanId)!;
        if (parentIdsMap.has(context.span_id)) {
          spansToDeleteMap.set(lastIterationEventSpanId, undefined);
        } else {
          // delete span
          cleanSpanSources({ spanId: lastIterationEventSpanId });
          spansMap.delete(lastIterationEventSpanId);
        }
      }

      // create new span
      spansMap.set(span.context.span_id, span);
      // update number of nested spans for parent_id if exists
      if (span.parent_id) {
        parentIdsMap.set(
          span.parent_id,
          (parentIdsMap.get(span.parent_id) || 0) + 1,
        );
      }

      // save the last event for each iteration
      if (groupIterations.length > 0) {
        if (eventsIterationsMap.has(lastIteration)) {
          eventsIterationsMap
            .get(lastIteration)!
            .set(meta.name, span.context.span_id);
        } else {
          eventsIterationsMap.set(
            lastIteration,
            new Map().set(meta.name, span.context.span_id),
          );
        }
      }
    });

    // The generated response and message history are collected from the `success` agent's event
    emitter.match(
      (event) =>
        event.name === successLLMEventName &&
        event.creator instanceof BaseAgent,
      (data: InferCallbackValue<ReActAgentCallbacks["success"]>) => {
        const { data: dataObject, memory } = data;

        generatedMessage = {
          role: dataObject.role,
          text: dataObject.text,
        };
        history = memory.messages.map((msg) => ({
          text: msg.text,
          role: msg.role,
        }));
      },
    );
  };
}

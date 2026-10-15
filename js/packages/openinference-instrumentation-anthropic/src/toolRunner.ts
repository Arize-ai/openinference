import type { Span } from "@opentelemetry/api";
import { context, SpanStatusCode, trace } from "@opentelemetry/api";

import type { TraceConfig } from "@arizeai/openinference-core";
import { getOutputAttributes } from "@arizeai/openinference-core";

import { getAnthropicOutputMessagesAttributes } from "./attributes/content";
import { getExecContext, recordSpanError } from "./spanLifecycle";
import type { ToolRunner } from "./types";
import { safelyGetAttributes, toJSONOutput } from "./utils";

/**
 * Runs each run of a tool runner under a CHAIN span of its own, so the LLM
 * span of every request and the TOOL span of every tool it runs are children
 * of it, and ends the span when the run ends:
 *
 * - run to the end (`for await`, `runUntilDone()` or `await runner`): status
 *   OK, with the final message as the output;
 * - left early (`break`): with the status unset;
 * - failed: status ERROR, with the error recorded.
 *
 * Every way of running the runner iterates it, so wrapping its async iterator
 * covers them all, and a run retried after a failure gets a new span.
 */
export function instrumentToolRunner({
  runner,
  startRunSpan,
  traceConfig,
}: {
  runner: ToolRunner;
  startRunSpan: () => Span;
  traceConfig: TraceConfig;
}) {
  const iterate = runner[Symbol.asyncIterator].bind(runner);
  type Item = Awaited<ReturnType<ReturnType<typeof iterate>["next"]>>;
  Object.defineProperty(runner, Symbol.asyncIterator, {
    configurable: true,
    writable: true,
    value: () => {
      const span = startRunSpan();
      const runContext = trace.setSpan(getExecContext(span), span);
      let spanEnded = false;
      const endSpan = () => {
        if (!spanEnded) {
          spanEnded = true;
          span.end();
        }
      };
      const onError = (error: unknown) => {
        if (!spanEnded) {
          recordSpanError(span, error);
          endSpan();
        }
      };
      const onDone = async () => {
        if (spanEnded) {
          return;
        }
        const message = await runner.done().catch(() => undefined);
        if (message != null) {
          span.setAttributes(
            safelyGetAttributes(() => ({
              ...getAnthropicOutputMessagesAttributes(message, traceConfig),
              ...getOutputAttributes(toJSONOutput(message)),
            })),
          );
        }
        span.setStatus({ code: SpanStatusCode.OK });
        endSpan();
      };

      const iterator = context.with(runContext, iterate);
      const tracedIterator = {
        async next(): Promise<Item> {
          let result: Item;
          try {
            result = await context.with(runContext, () => iterator.next());
          } catch (error) {
            onError(error);
            throw error;
          }
          if (result.done) {
            await onDone();
          }
          return result;
        },
        async return(value?: unknown): Promise<Item> {
          try {
            const returned = await context.with(runContext, () => iterator.return?.(value));
            const result: Item = returned ?? { done: true, value: undefined };
            endSpan();
            return result;
          } catch (error) {
            onError(error);
            throw error;
          }
        },
        [Symbol.asyncIterator]() {
          return tracedIterator;
        },
      };
      return tracedIterator;
    },
  });
}

import type { Stream } from "@anthropic-ai/sdk/streaming";
import type { Span } from "@opentelemetry/api";
import { diag, SpanStatusCode } from "@opentelemetry/api";

import type { TraceConfig } from "@arizeai/openinference-core";

import { MODULE_NAME } from "./constants";
import {
  getAnthropicStreamAttributes,
  createAnthropicStreamState,
  applyAnthropicStreamEvent,
} from "./streamState";
import type { RawMessageStreamEvent, MetadataSources } from "./types";
import { safelyGetAttributes } from "./utils";

/**
 * Observes an Anthropic stream through the iterator its caller reads, and ends
 * the span when the caller is done with it:
 *
 * - read to the end: the span ends with status OK;
 * - left early (`break`, `return()`, `toReadableStream()` cancel) or aborted
 *   through `stream.controller`: the span ends with what was read so far and
 *   the status left unset, as the request did not run to completion;
 * - failed (a network error or an `error` event): the error is recorded on
 *   the span, which ends with status ERROR, and the error still reaches the
 *   caller.
 *
 * The SDK `Stream` builds every iterator it hands out (`for await`, `tee()`,
 * `toReadableStream()`) from its `iterator` factory, so wrapping that factory
 * observes all of them while leaving the stream's own controller, early-exit
 * abort and helpers untouched.
 */
export function instrumentAnthropicStream({
  stream,
  span,
  endSpan,
  recordError,
  metadataSources,
  traceConfig,
}: {
  stream: Stream<RawMessageStreamEvent>;
  span: Span;
  endSpan: () => void;
  recordError: (error: unknown) => void;
  metadataSources: MetadataSources;
  traceConfig: TraceConfig;
}) {
  const state = createAnthropicStreamState();
  const recordStreamAttributes = () => {
    span.setAttributes(
      safelyGetAttributes(() => getAnthropicStreamAttributes(state, metadataSources, traceConfig)),
    );
  };
  const observer: AnthropicStreamObserver = {
    onEvent: (event) => {
      try {
        applyAnthropicStreamEvent({ chunk: event, state });
      } catch (error) {
        diag.warn(`Failed to record a ${MODULE_NAME} stream event`, error);
      }
    },
    onDone: () => {
      recordStreamAttributes();
      if (!stream.controller.signal.aborted) {
        span.setStatus({ code: SpanStatusCode.OK });
      }
      endSpan();
    },
    onReturn: () => {
      recordStreamAttributes();
      endSpan();
    },
    onError: (error) => {
      recordStreamAttributes();
      recordError(error);
    },
  };

  // Only the first iterator is observed: the response can be read once, so a
  // later one only gets the SDK's "consumed stream" error, which belongs to
  // that caller and must not end the span while the first is still reading.
  let observing = false;
  const observeFirst = (iterator: AsyncIterator<RawMessageStreamEvent>) => {
    if (observing) {
      return iterator;
    }
    observing = true;
    return wrapAnthropicStreamIterator({ iterator, observer });
  };

  const iteratorFactory: unknown = Reflect.get(stream, "iterator");
  if (typeof iteratorFactory === "function") {
    Reflect.set(stream, "iterator", () =>
      observeFirst(getAsyncIterator(Reflect.apply(iteratorFactory, stream, []))),
    );
    return;
  }
  // Fall back to the async iterator protocol should the SDK stop building its
  // iterators from `iterator`.
  const asyncIterator = stream[Symbol.asyncIterator].bind(stream);
  Object.defineProperty(stream, Symbol.asyncIterator, {
    configurable: true,
    writable: true,
    value: () => observeFirst(asyncIterator()),
  });
}

/**
 * Narrows the value returned by the SDK stream's iterator factory.
 */
function getAsyncIterator(value: unknown): AsyncIterator<RawMessageStreamEvent> {
  if (isAsyncIterator(value)) {
    return value;
  }
  throw new TypeError(`${MODULE_NAME} stream iterator factory did not return an iterator`);
}

function isAsyncIterator(value: unknown): value is AsyncIterator<RawMessageStreamEvent> {
  return (
    value != null &&
    typeof value === "object" &&
    "next" in value &&
    typeof value.next === "function"
  );
}

interface AnthropicStreamObserver {
  onEvent: (event: RawMessageStreamEvent) => void;
  onDone: () => void;
  onReturn: () => void;
  onError: (error: unknown) => void;
}

/**
 * Wraps a stream iterator so that each event, the end of the stream, an early
 * return and a failure are reported to the observer. Calls are forwarded to the
 * SDK iterator unchanged, so its `return()` still aborts the HTTP request when
 * the caller leaves the loop early.
 */
function wrapAnthropicStreamIterator({
  iterator,
  observer,
}: {
  iterator: AsyncIterator<RawMessageStreamEvent>;
  observer: AnthropicStreamObserver;
}): AsyncIterator<RawMessageStreamEvent> {
  return {
    async next() {
      let result: IteratorResult<RawMessageStreamEvent>;
      try {
        result = await iterator.next();
      } catch (error) {
        observer.onError(error);
        throw error;
      }
      if (result.done) {
        observer.onDone();
      } else {
        observer.onEvent(result.value);
      }
      return result;
    },
    async return(value?: unknown) {
      try {
        const result: IteratorResult<RawMessageStreamEvent> = iterator.return
          ? await iterator.return(value)
          : { done: true, value };
        observer.onReturn();
        return result;
      } catch (error) {
        observer.onError(error);
        throw error;
      }
    },
  };
}

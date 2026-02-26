import type { RetrieveAndGenerateStreamResponseOutput } from "@aws-sdk/client-bedrock-agent-runtime";
import { diag } from "@opentelemetry/api";

import type { CallbackHandler, RagCallbackHandler } from "./callbackHandler";
import { getObjectDataFromUnknown } from "./utils/jsonUtils";

export function interceptAgentResponse<
  T extends { chunk?: { bytes?: Uint8Array }; trace?: object },
>(originalStream: AsyncIterable<T>, callback: CallbackHandler): AsyncIterable<T> {
  return {
    async *[Symbol.asyncIterator]() {
      try {
        for await (const item of originalStream) {
          try {
            if (item.chunk?.bytes) {
              callback.consumeResponse(item.chunk.bytes);
            } else if (item.trace) {
              callback.consumeTrace(item.trace as Record<string, unknown>);
            }
          } catch (err: unknown) {
            diag.debug("Error in interceptAgentResponse Stream:", err);
          }
          yield item;
        }
        try {
          callback.onComplete();
        } catch (err: unknown) {
          diag.debug("Error in interceptAgentResponse:", err);
        }
      } catch (err) {
        callback.onError(err);
        throw err;
      }
    },
  };
}

/**
 * Intercepts a streaming RAG (Retrieve and Generate) response and invokes the provided RagCallbackHandler.
 *
 * This utility wraps an AsyncIterable stream of RetrieveAndGenerateStreamResponseOutput items,
 * forwarding output text and citation events to the callback as they are received.
 *
 * - For each streamed item, if output text is present, it is passed to callback.handleOutput.
 * - If a citation is present, it is extracted and passed to callback.handleCitation.
 * - After the stream completes, callback.onComplete is called.
 * - If an error occurs, callback.onError is called and the error is re-thrown.
 *
 * @template T Extends RetrieveAndGenerateStreamResponseOutput
 * @param originalStream The original async iterable stream of RAG response items.
 * @param callback The RagCallbackHandler instance to receive output and citation events.
 * @returns An async iterable that yields the same items as the original stream, while invoking the callback.
 */
export function interceptRagResponse<T extends RetrieveAndGenerateStreamResponseOutput>(
  originalStream: AsyncIterable<T>,
  callback: RagCallbackHandler,
): AsyncIterable<T> {
  return {
    async *[Symbol.asyncIterator]() {
      try {
        for await (const item of originalStream) {
          try {
            if (item.output?.text) {
              callback.handleOutput(item.output?.text);
            }
            if (item?.citation) {
              callback.handleCitation(
                getObjectDataFromUnknown({ data: item, key: "citation" }) || {},
              );
            }
          } catch (err: unknown) {
            diag.debug("Error in interceptRagResponse stream", err);
          }
          yield item;
        }
        try {
          callback.onComplete();
        } catch (err: unknown) {
          diag.debug("Error in interceptRagResponse:", err);
        }
      } catch (err) {
        callback.onError(err);
        throw err;
      }
    },
  };
}

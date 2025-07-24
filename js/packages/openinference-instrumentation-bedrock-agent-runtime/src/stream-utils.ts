import { CallbackHandler } from "./callback-handler";

export function interceptAgentResponse<
  T extends { chunk?: { bytes?: Uint8Array }; trace?: object },
>(
  originalStream: AsyncIterable<T>,
  callback: CallbackHandler,
): AsyncIterable<T> {
  return {
    async *[Symbol.asyncIterator]() {
      try {
        for await (const item of originalStream) {
          if (item.chunk?.bytes) {
            callback.consumeResponse(item.chunk.bytes);
          } else if (item.trace) {
            callback.consumeTrace(item.trace);
          }
          yield item;
        }
        callback.onComplete();
      } catch (err) {
        callback.onError(err);
        throw err;
      }
    },
  };
}

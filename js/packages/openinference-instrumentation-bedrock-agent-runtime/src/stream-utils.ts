import { ResponseHandler } from "./response-handler";

export function interceptAgentResponse<
  T extends { chunk?: { bytes?: Uint8Array } },
>(
  originalStream: AsyncIterable<T>,
  callback: ResponseHandler,
): AsyncIterable<T> {
  return {
    async *[Symbol.asyncIterator]() {
      try {
        for await (const item of originalStream) {
          if (item.chunk?.bytes) {
            callback.consumeResponse(item.chunk.bytes);
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

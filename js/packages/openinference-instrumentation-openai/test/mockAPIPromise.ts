import type OpenAI from "openai";
import { APIPromise } from "openai";

/**
 * Builds a `client.post` mock implementation whose parsed value is `fn()`.
 *
 * The SDK's `create` methods call `._thenUnwrap()` on whatever `post` returns
 * before handing it to the caller, so a mock that resolves to a plain object
 * through an ordinary Promise breaks them. This returns a real `APIPromise`
 * over a stub HTTP response instead, so both the SDK and the instrumentation
 * see the shape they expect.
 *
 * @param client - The client whose `post` is being mocked.
 * @param fn - Produces the parsed response body, synchronously or not.
 * @returns A `post` implementation for `vi.spyOn(client, "post").mockImplementation`.
 */
export function mockAPIPromise<T>(client: OpenAI, fn: () => T | Promise<T>): () => APIPromise<T> {
  return () => {
    // One body per `post` call: the APIPromise may be parsed more than once
    // (vitest's spy also awaits the returned promise to record its result), and
    // every parse must see the same response, as it would over a real request.
    const parsed = fn();
    return new APIPromise<T>(
      client,
      Promise.resolve({
        response: new Response(),
        options: { method: "post", path: "/mock" },
        controller: new AbortController(),
        requestLogID: "mock",
        retryOfRequestLogID: undefined,
        startTime: Date.now(),
      }),
      // The SDK types the parsed value as WithRequestID<T>, which only adds an
      // optional `_request_id` that the stub has no request id to fill.
      () => parsed as never,
    );
  };
}

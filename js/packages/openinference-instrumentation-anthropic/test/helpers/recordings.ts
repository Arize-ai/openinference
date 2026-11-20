/**
 * Network recordings for tests, built on `nock.back`.
 *
 * A test makes real SDK calls. The first time, they reach the live API and
 * the request and response pairs are saved to `test/recordings/<name>.json`.
 * After that the file is replayed, so the test needs no API key or network.
 *
 * ```ts
 * afterEach(() => resetRecordings());
 *
 * it("should trace a message", async () => {
 *   const client = new Anthropic({ apiKey: getAPIKey("ANTHROPIC_API_KEY"), maxRetries: 0 });
 *   const recording = await startRecording("message");
 *   await client.messages.create(request);
 *   recording.stop();
 *   // assert on the exported spans
 * });
 * ```
 *
 * To capture recordings, run the test with a real API key and one of:
 *
 * - `NOCK_BACK_MODE=record` captures the recordings that do not exist yet
 * - `NOCK_BACK_MODE=update` captures every recording again
 *
 * This file adds four things to what `nock.back` does on its own:
 *
 * 1. Replay is strict by default. nock's own default lets a request without a
 *    recording through to the live API; here it fails the test.
 * 2. Recordings are readable. nock saves a compressed response as hex, so
 *    bodies are decompressed before they are written.
 * 3. Recordings are anonymous. Only the `content-type` response header is
 *    kept, and request headers (with the API key) are never recorded.
 * 4. A recording that was started but never stopped, because the test failed
 *    half way, is cleaned up by `resetRecordings`.
 */
import path from "node:path";
import { brotliDecompressSync, gunzipSync, inflateSync } from "node:zlib";

import nock from "nock";

import { isArray, isRecord, isString, isStringArray } from "@arizeai/openinference-core";

/** The directory recordings are saved to and replayed from. */
const RECORDINGS_DIRECTORY = path.join(__dirname, "..", "recordings");

/**
 * The mode used when `NOCK_BACK_MODE` is not set: replay from disk and refuse
 * every request that has no recording.
 */
const DEFAULT_MODE: nock.BackMode = "lockdown";

/** The `NOCK_BACK_MODE` values that send requests to the live API. */
const LIVE_MODES: nock.BackMode[] = ["record", "update", "wild"];

/** The API key a client is given when its requests are replayed from disk. */
const REPLAY_API_KEY = "fake-api-key";

/**
 * The response headers kept in a recording. The rest are not needed to replay
 * a response, and some identify the account that recorded it.
 */
const RECORDED_RESPONSE_HEADERS = ["content-type"];

/**
 * How to decompress a response body, by `content-encoding`. A body with an
 * encoding that is not listed is saved as nock recorded it.
 */
const DECOMPRESSORS: Partial<Record<string, (body: Buffer) => Buffer>> = {
  br: brotliDecompressSync,
  gzip: gunzipSync,
  deflate: inflateSync,
};

/** A recording that has been started, either capturing or replaying. */
export interface Recording {
  /**
   * Ends the recording. When capturing, writes the recording to disk. When
   * replaying, fails if a recorded request was never made.
   */
  stop: () => void;
}

/**
 * Finishes the recording that is in progress, or `undefined` when there is
 * none. Kept so `resetRecordings` can finish a recording a failed test left
 * open.
 */
let finishActiveRecording: (() => void) | undefined;

/**
 * Starts capturing or replaying the network requests of a test.
 *
 * On replay, each request must match a recorded one by method, path and body.
 * Call `stop` on the returned recording once the requests have been made.
 *
 * @param name - The file name of the recording, without its extension. Tests
 * that send the same request can share a name.
 * @returns The recording, to be stopped when the requests have been made
 */
export async function startRecording(name: string): Promise<Recording> {
  if (!nock.isActive()) {
    nock.activate();
  }
  nock.back.fixtures = RECORDINGS_DIRECTORY;
  nock.back.setMode(getMode());
  const { nockDone, context } = await nock.back(`${name}.json`, {
    afterRecord: sanitizeRecordings,
  });
  finishActiveRecording = nockDone;
  return {
    stop: () => {
      finishActiveRecording = undefined;
      nockDone();
      context.assertScopesFinished();
    },
  };
}

/**
 * Cleans up after a test. Call it from `afterEach`.
 *
 * Finishes a recording the test left open and removes the recorded responses
 * that were not used, so neither affects the tests that follow.
 */
export function resetRecordings(): void {
  finishActiveRecording?.();
  finishActiveRecording = undefined;
  nock.cleanAll();
}

/**
 * Picks the API key for a test client: the real key when requests reach the
 * live API, and a placeholder when they are replayed.
 * @param environmentVariable - The environment variable that holds the real key
 * @returns The API key to construct the client with
 */
export function getAPIKey(environmentVariable: string): string {
  if (!isLiveMode()) {
    return REPLAY_API_KEY;
  }
  const apiKey = process.env[environmentVariable];
  if (!isString(apiKey) || apiKey.length === 0) {
    throw new Error(`${environmentVariable} must be set to record network requests`);
  }
  return apiKey;
}

/**
 * Whether requests reach the live API instead of being replayed.
 * @returns `true` when `NOCK_BACK_MODE` is `record`, `update` or `wild`
 */
export function isLiveMode(): boolean {
  return LIVE_MODES.includes(getMode());
}

/**
 * Reads the nock back mode.
 * @param mode - The requested mode, read from `NOCK_BACK_MODE` by default
 * @returns The requested mode, or strict replay when none is requested
 */
function getMode(mode: string | undefined = process.env.NOCK_BACK_MODE): nock.BackMode {
  return isString(mode) && mode.length > 0 ? (mode as nock.BackMode) : DEFAULT_MODE;
}

/**
 * Prepares captured recordings for saving: decompresses the response bodies
 * and drops every response header that is not needed for replay.
 * @param definitions - The request and response pairs nock captured
 * @returns The pairs to write to disk
 */
function sanitizeRecordings(definitions: nock.Definition[]): nock.Definition[] {
  return definitions.map((definition) => {
    const headers = getResponseHeaders(definition.rawHeaders);
    const recordedHeaders: Record<string, string> = {};
    for (const headerName of RECORDED_RESPONSE_HEADERS) {
      const value = headers[headerName];
      if (isString(value)) {
        recordedHeaders[headerName] = value;
      }
    }
    return {
      ...definition,
      // The body is JSON, text or untouched hex chunks, all of which nock
      // replays, but its type for `response` does not cover a parsed JSON value.
      response: decodeResponseBody({
        response: definition.response,
        contentEncoding: headers["content-encoding"],
        contentType: headers["content-type"],
      }) as nock.Definition["response"],
      rawHeaders: recordedHeaders,
    };
  });
}

/**
 * Reads the response headers of a captured recording, which nock stores as
 * either a flat `[name, value, ...]` list or an object.
 * @param rawHeaders - The response headers nock captured
 * @returns The headers, keyed by lower case header name
 */
function getResponseHeaders(rawHeaders: unknown): Record<string, string> {
  const entries: [unknown, unknown][] = [];
  if (isArray(rawHeaders)) {
    for (let index = 0; index + 1 < rawHeaders.length; index += 2) {
      entries.push([rawHeaders[index], rawHeaders[index + 1]]);
    }
  } else if (isRecord(rawHeaders)) {
    entries.push(...Object.entries(rawHeaders));
  }
  const headers: Record<string, string> = {};
  for (const [name, value] of entries) {
    if (isString(name) && isString(value)) {
      headers[name.toLowerCase()] = value;
    }
  }
  return headers;
}

/**
 * Decompresses a captured response body, which nock stores as hex encoded
 * chunks when the response was compressed.
 * @param params - The captured body and how it was encoded
 * @param params.response - The response body nock captured
 * @param params.contentEncoding - The `content-encoding` header of the response
 * @param params.contentType - The `content-type` header of the response
 * @returns The decompressed body, parsed when it is JSON. A body that was not
 * compressed is returned as captured.
 */
function decodeResponseBody({
  response,
  contentEncoding,
  contentType,
}: {
  response: unknown;
  contentEncoding: string | undefined;
  contentType: string | undefined;
}): unknown {
  const decompress = isString(contentEncoding) ? DECOMPRESSORS[contentEncoding] : undefined;
  if (!isStringArray(response) || !decompress) {
    return response;
  }
  const body = decompress(Buffer.from(response.join(""), "hex")).toString("utf8");
  const isJSON = isString(contentType) && contentType.includes("application/json");
  return isJSON ? JSON.parse(body) : body;
}

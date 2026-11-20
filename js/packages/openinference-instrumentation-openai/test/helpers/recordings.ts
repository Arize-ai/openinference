import path from "node:path";
import { brotliDecompressSync, gunzipSync, inflateSync } from "node:zlib";

import nock from "nock";

import { isArray, isRecord, isString, isStringArray } from "@arizeai/openinference-core";

/** The directory the network recordings are saved to and replayed from. */
const RECORDINGS_DIRECTORY = path.join(__dirname, "..", "recordings");

/**
 * The mode recordings run in unless `NOCK_BACK_MODE` says otherwise: replay
 * from disk and refuse every request that has no recording, so the tests need
 * neither network access nor an API key.
 */
const DEFAULT_MODE: nock.BackMode = "lockdown";

/** The `NOCK_BACK_MODE` values that send requests to the live API. */
const LIVE_MODES: nock.BackMode[] = ["record", "update", "wild"];

/** The API key the client uses when requests are replayed from disk. */
const REPLAY_API_KEY = "fake-api-key";

/**
 * The response headers kept in a recording. The rest identify the account
 * that recorded it (organization, project, cookies, request ids) and are not
 * needed to replay the response.
 */
const RECORDED_RESPONSE_HEADERS = ["content-type"];

/**
 * Decompression by `content-encoding`. An encoding that is not listed leaves
 * the body as nock recorded it.
 */
const DECOMPRESSORS: Partial<Record<string, (body: Buffer) => Buffer>> = {
  br: brotliDecompressSync,
  gzip: gunzipSync,
  deflate: inflateSync,
};

/** A network recording in progress, either being captured or being replayed. */
export interface Recording {
  /**
   * Ends the recording. Saves the captured requests when recording, and
   * otherwise fails if a recorded request was never made.
   */
  stop: () => void;
}

/**
 * Reads the nock back mode from `NOCK_BACK_MODE`.
 * @param mode - The requested mode, read from `NOCK_BACK_MODE` by default
 * @returns The requested mode, or the replay only default when it is not set
 */
function getMode(mode: string | undefined = process.env.NOCK_BACK_MODE): nock.BackMode {
  return isString(mode) && mode.length > 0 ? (mode as nock.BackMode) : DEFAULT_MODE;
}

/**
 * Whether requests reach the live API, which is the case when
 * `NOCK_BACK_MODE` is `record`, `update` or `wild`.
 * @returns `true` when requests are sent to the live API
 */
export function isLiveMode(): boolean {
  return LIVE_MODES.includes(getMode());
}

/**
 * Picks the API key for a test client: the real key from the environment when
 * requests reach the live API, and a placeholder when they are replayed.
 * @param environmentVariable - The environment variable that holds the real key
 * @returns The API key to construct the client with
 */
export function getAPIKey(environmentVariable: string): string {
  const apiKey = process.env[environmentVariable];
  if (!isLiveMode()) {
    return REPLAY_API_KEY;
  }
  if (!isString(apiKey) || apiKey.length === 0) {
    throw new Error(`${environmentVariable} must be set to record network requests`);
  }
  return apiKey;
}

/**
 * Reads the response headers nock recorded, which it stores either as a flat
 * `[name, value, ...]` list or as an object.
 * @param rawHeaders - The response headers nock recorded
 * @returns The headers, keyed by lower case header name
 */
function getResponseHeaders(rawHeaders: unknown): Record<string, string> {
  const headers: Record<string, string> = {};
  const entries: [unknown, unknown][] = [];
  if (isArray(rawHeaders)) {
    for (let index = 0; index + 1 < rawHeaders.length; index += 2) {
      entries.push([rawHeaders[index], rawHeaders[index + 1]]);
    }
  } else if (isRecord(rawHeaders)) {
    entries.push(...Object.entries(rawHeaders));
  }
  for (const [name, value] of entries) {
    if (isString(name) && isString(value)) {
      headers[name.toLowerCase()] = value;
    }
  }
  return headers;
}

/**
 * Decompresses a response body. nock records a compressed body as a list of
 * hex encoded chunks, which would otherwise be unreadable in the recording.
 * @param params - The recorded body and how it was encoded
 * @param params.response - The response body nock recorded
 * @param params.contentEncoding - The `content-encoding` header of the response
 * @param params.contentType - The `content-type` header of the response
 * @returns The decompressed body, parsed when it is JSON, or the body as
 * recorded when it was not compressed
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

/**
 * Makes freshly captured recordings readable and anonymous before they are
 * written to disk: compressed bodies are decompressed and account specific
 * response headers are dropped. Request headers, and with them the API key,
 * are never recorded.
 * @param definitions - The request and response pairs nock captured
 * @returns The same pairs with readable bodies and only the replayable headers
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
 * Starts recording or replaying the network requests of a test.
 *
 * Recordings are replayed from `test/recordings` by default. Run the tests
 * with `NOCK_BACK_MODE=record` and a real API key to capture the recordings
 * that are missing, or with `NOCK_BACK_MODE=update` to capture all of them
 * again. Replayed requests must match the recorded method, path and body.
 *
 * @param name - The file name of the recording, without its extension
 * @returns A handle that ends the recording
 *
 * @example
 * ```ts
 * const recording = await startRecording("chat-completion");
 * await client.chat.completions.create(request);
 * recording.stop();
 * ```
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
  return {
    stop: () => {
      nockDone();
      context.assertScopesFinished();
    },
  };
}

/**
 * Removes the interceptors a recording left behind, so they do not answer the
 * requests of the tests that follow.
 */
export function resetRecordings(): void {
  nock.cleanAll();
}

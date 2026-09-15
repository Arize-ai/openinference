import * as fs from "fs";
import * as path from "path";

/**
 * Set ANTHROPIC_VCR_MODE=record and a real ANTHROPIC_API_KEY to (re)record
 * cassettes against the live Anthropic API. By default, cassettes are
 * replayed from disk so tests run without network access or an API key.
 */
const isRecordingMode = process.env.ANTHROPIC_VCR_MODE === "record";

const CASSETTES_DIR = path.join(__dirname, "..", "cassettes");

interface Cassette {
  status: number;
  headers: Record<string, string>;
  body: string;
}

const PASSTHROUGH_HEADERS = new Set(["content-type"]);

function getCassettePath(name: string): string {
  return path.join(CASSETTES_DIR, `${name}.json`);
}

function readCassette(cassettePath: string): Response {
  const raw = fs.readFileSync(cassettePath, "utf-8");
  const cassette: Cassette = JSON.parse(raw);
  return new Response(cassette.body, {
    status: cassette.status,
    headers: cassette.headers,
  });
}

/**
 * Returns a custom `fetch` implementation for the Anthropic SDK that either
 * records a real response to a cassette file or replays a previously
 * recorded cassette.
 *
 * In recording mode, if the cassette already exists it is replayed as-is
 * (no live request is made); only missing cassettes are recorded.
 */
export function vcrFetch(cassetteName: string): typeof fetch {
  const cassettePath = getCassettePath(cassetteName);

  if (isRecordingMode) {
    return (async (input, init) => {
      if (fs.existsSync(cassettePath)) {
        return readCassette(cassettePath);
      }
      const response = await fetch(input, init);
      const body = await response.clone().text();
      const headers: Record<string, string> = {};
      response.headers.forEach((value, key) => {
        if (PASSTHROUGH_HEADERS.has(key.toLowerCase())) {
          headers[key] = value;
        }
      });
      const cassette: Cassette = {
        status: response.status,
        headers,
        body,
      };
      fs.mkdirSync(CASSETTES_DIR, { recursive: true });
      fs.writeFileSync(cassettePath, JSON.stringify(cassette, null, 2));
      return response;
    }) as typeof fetch;
  }

  return (async () => readCassette(cassettePath)) as typeof fetch;
}

/**
 * Returns a custom `fetch` implementation that records/replays a sequence of
 * cassettes, one per call, named `${baseName}-${callIndex}`. Useful for
 * multi-turn conversations that issue more than one request per test.
 */
export function vcrFetchSequence(baseName: string): typeof fetch {
  let callIndex = 0;
  return (async (input, init) => {
    const fetchImpl = vcrFetch(`${baseName}-${callIndex}`);
    callIndex += 1;
    return fetchImpl(input, init);
  }) as typeof fetch;
}

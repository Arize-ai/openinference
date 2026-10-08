# OpenAI cache writes and reads in Phoenix

OpenInference records the API's cache counts without estimating them or changing
its prompt/completion/total counts. Both Python and JavaScript OpenAI
instrumentation support Chat Completions and Responses, including streaming.
OpenAI Agents instrumentation also records cache writes from the usage exposed by
its SDK (Python Responses usage; JavaScript Responses, Chat Completions, and
generation usage details).

| API usage field | Span attribute |
| --- | --- |
| `prompt_tokens_details.cached_tokens` (Chat) / `input_tokens_details.cached_tokens` (Responses) | `llm.token_count.prompt_details.cache_read` |
| `prompt_tokens_details.cache_write_tokens` (Chat) / `input_tokens_details.cache_write_tokens` (Responses) | `llm.token_count.prompt_details.cache_write` |

Explicit zeros are retained. Missing fields remain absent, including with older
models or SDK responses that do not report cache writes. A cache miss does **not**
imply a write: do not derive writes as `prompt - cache_read`.

Cache reads and writes are already included in OpenAI's prompt/input count.
Ordinary input is `prompt - cache_read - cache_write`; adding the cache counts to
prompt or total would double-count them. These are token counts, not prices.
Consumers should apply the model's ordinary-input, cache-read, and cache-write
rates separately. See OpenAI's [prompt caching guide](https://developers.openai.com/api/docs/guides/prompt-caching)
and [Astra pricing](https://developers.openai.com/api/docs/models/gpt-6-astra).

## Run against this checkout

Start Phoenix at `http://localhost:6006` and set `OPENAI_API_KEY`. From the repository
root, install the local instrumentor in an isolated environment:

```sh
uv venv /tmp/oi-cache-example
uv pip install --python /tmp/oi-cache-example/bin/python \
  -e python/openinference-semantic-conventions \
  -e python/openinference-instrumentation \
  -e python/instrumentation/openinference-instrumentation-openai \
  openai opentelemetry-exporter-otlp-proto-http

/tmp/oi-cache-example/bin/python \
  python/instrumentation/openinference-instrumentation-openai/examples/cache_tokens.py \
  > /tmp/cache-usage.jsonl
```

The example defaults to `gpt-6-astra` and makes two billable calls. It uses a unique
prefix longer than the model's cache minimum, holds that prefix in a separate
message, and changes only the final question. It checks each raw usage count
against the instrumented span, checks for a cold write and a warm read, and exports
the spans to the `openai-cache-tokens` Phoenix project. Cache availability is a
service behavior: if the warm call misses, the example fails visibly rather than
manufacturing a cache hit.

Use `--api chat` for Chat Completions, `--stream` for streaming, or both. Chat
streaming requests `stream_options.include_usage`; Responses streaming obtains
usage from the final response event. Streams must be fully consumed. `--model`,
`--project`, and `--endpoint` override the defaults.

## Verify the Phoenix round trip

With the Phoenix CLI installed, retrieve spans and compare the persisted counts
by span ID with the example's raw-usage assertions:

```sh
px span list --endpoint http://localhost:6006 --project openai-cache-tokens \
  --limit 100 --format raw --no-progress > /tmp/cache-spans.json

python3 - <<'PY'
import json
spans = {
    span["context"]["span_id"]: span
    for span in json.load(open("/tmp/cache-spans.json"))
}
for line in open("/tmp/cache-usage.jsonl"):
    expected = json.loads(line)
    actual = spans[expected["span_id"]]["attributes"]
    for key, value in expected["attributes"].items():
        assert actual[key] == value, (key, actual[key], value)
    print("Verified Phoenix span", expected["span_id"])
PY
```

If ingestion has not completed, repeat the readback. Export completion alone does
not verify the stored attributes. Open the project in the Phoenix UI to inspect
`llm.token_count.prompt_details.cache_read` and `cache_write` alongside prompt and
total counts. Accurate ingestion of counts does not by itself verify Phoenix's
model pricing configuration.

## Observed Astra run

On 2026-09-21, using Python OpenAI SDK 3.16.2 and this checkout, all eight
spans below were retrieved from local Phoenix and their prompt, completion,
total, cache-read, and cache-write counts matched the API usage exactly.
Counts vary with the unique prefix and service behavior.

| API | Streaming | Cold read / write | Warm read / write |
| --- | --- | --- | --- |
| Responses | No | 0 / 1,843 | 1,834 / 9 |
| Responses | Yes | 0 / 1,836 | 1,827 / 9 |
| Chat Completions | No | 0 / 1,837 | 1,828 / 9 |
| Chat Completions | Yes | 0 / 1,842 | 1,833 / 9 |

For example, the first Responses request used 1,846 prompt tokens: 1,843 written
to cache and 3 ordinary input tokens. The second also used 1,846 prompt tokens:
1,834 read, 9 written, and 3 ordinary input tokens. Neither cache count was added
to the prompt total.

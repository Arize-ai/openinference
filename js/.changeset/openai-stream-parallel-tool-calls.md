---
"@arizeai/openinference-instrumentation-openai": patch
---

Fix streamed chat completion and Responses API spans:

- Record each parallel tool call of a streamed chat completion as its own tool call. Stream deltas were keyed by their position in the chunk instead of their `index`, so calls streamed in separate chunks were merged into one with concatenated ids, names and arguments. Providers that leave out the `index`, such as Gemini's OpenAI-compatible endpoint, start a new tool call on each new tool call id.
- Set the status of streamed chat completion spans: OK when the stream is read to the end, ERROR with the error recorded when reading it fails, and unset when the request is aborted. The span keeps what was streamed before it ended.
- End the span with status ERROR when a chat completion or Responses API stream fails part way. Previously the span was left unended and the instrumentation raised an unhandled promise rejection, which ended the process even when the caller handled the error. An aborted Responses API stream now leaves the status unset instead of setting ERROR.

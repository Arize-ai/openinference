/* eslint-disable no-console */
import { shutdownTracing } from "./instrumentation";

import { randomUUID } from "node:crypto";

import Anthropic from "@anthropic-ai/sdk";

import { SemanticConventions } from "@arizeai/openinference-semantic-conventions";

const model = "claude-opus-5";

/**
 * Builds a system prompt long enough to cache on every current model (the
 * minimum cacheable prefix is 512 to 4096 tokens depending on the model). The
 * run id makes the prefix unique, so the first request always writes a new
 * cache entry instead of reading one left over from an earlier run.
 */
function buildSystemPrompt(runId: string): string {
  const notes = Array.from(
    { length: 400 },
    (_, i) => `Note ${i + 1}: box ${i + 1} of the archive is on shelf ${(i % 20) + 1}.`,
  );
  return [`Run ${runId}. Answer questions about the archive using these notes.`, ...notes].join(
    "\n",
  );
}

/**
 * The token count attributes the span should carry for this usage. Anthropic's
 * input_tokens excludes cached tokens, so the prompt count adds them back.
 */
function expectedTokenCounts(usage: Anthropic.Messages.Usage) {
  const cacheWrite = usage.cache_creation_input_tokens ?? 0;
  const cacheRead = usage.cache_read_input_tokens ?? 0;
  const prompt = usage.input_tokens + cacheWrite + cacheRead;
  return {
    [SemanticConventions.LLM_TOKEN_COUNT_PROMPT]: prompt,
    [SemanticConventions.LLM_TOKEN_COUNT_COMPLETION]: usage.output_tokens,
    [SemanticConventions.LLM_TOKEN_COUNT_TOTAL]: prompt + usage.output_tokens,
    ...(cacheWrite > 0 && {
      [SemanticConventions.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE]: cacheWrite,
    }),
    ...(cacheRead > 0 && {
      [SemanticConventions.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ]: cacheRead,
    }),
  };
}

async function main() {
  try {
    const anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
    const systemPrompt = buildSystemPrompt(randomUUID());

    const requests = [
      // Writes the system prompt to the cache.
      { name: "cache write", cache: true, question: "Which shelf holds box 7?" },
      // Same prefix, so it is read back from the cache.
      { name: "cache read", cache: true, question: "Which shelf holds box 42?" },
      // No cache_control: both cache counts are 0 and left off the span.
      { name: "no caching", cache: false, question: "Which shelf holds box 99?" },
    ];

    for (const { name, cache, question } of requests) {
      const message = await anthropic.messages.create({
        model,
        max_tokens: 1024,
        system: [
          {
            type: "text",
            text: systemPrompt,
            ...(cache && { cache_control: { type: "ephemeral" as const } }),
          },
        ],
        messages: [{ role: "user", content: question }],
      });

      console.log(
        JSON.stringify(
          { request: name, usage: message.usage, expectedSpan: expectedTokenCounts(message.usage) },
          null,
          2,
        ),
      );
    }
  } finally {
    await shutdownTracing();
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

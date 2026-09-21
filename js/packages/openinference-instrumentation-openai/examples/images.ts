/* eslint-disable no-console */
import "./instrumentation";

import { isPatched } from "../src";

import OpenAI from "openai";

if (!isPatched()) {
  throw new Error("OpenAI instrumentation failed");
}

const openai = new OpenAI();

async function main() {
  const response = await openai.images.generate({
    model: "gpt-image-1",
    prompt: "A small red dot with 1px image.",
    size: "1024x1024",
    // Compressed WebP keeps the recorded data URL small enough for Phoenix to render it.
    quality: "low",
    output_format: "webp",
    output_compression: 0,
  });

  const image = response.data?.[0];
  console.log(`Generated image response: b64_json=${image?.b64_json != null}`);
  console.log(`Recorded data URL length: ${(image?.b64_json?.length ?? 0) + 23}`);
}

main().catch(console.error);

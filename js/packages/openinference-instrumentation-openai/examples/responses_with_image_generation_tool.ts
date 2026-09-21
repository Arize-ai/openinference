/* eslint-disable no-console */
import "./instrumentation";

import { isPatched } from "../src";

import OpenAI from "openai";

if (!isPatched()) {
  throw new Error("OpenAI instrumentation failed");
}

const openai = new OpenAI();

async function main() {
  const response = await openai.responses.create({
    model: "gpt-4.1-mini",
    tools: [
      {
        type: "image_generation",
        size: "1024x1024",
        // Compressed WebP keeps the recorded data URL small enough for Phoenix to render it.
        quality: "low",
        output_format: "webp",
        output_compression: 0,
      },
    ],
    input: "Generate a small 1px white color dot",
  });

  const images = response.output
    .filter((item) => item.type === "image_generation_call")
    .map((item) => item.result);
  console.log(`Generated ${images.length} image(s): result=${Boolean(images[0])}`);
  console.log(`Recorded data URL length: ${(images[0]?.length ?? 0) + 24}`);
}

main().catch(console.error);

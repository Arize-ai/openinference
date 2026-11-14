// @ts-check
import { defineConfig } from "astro/config";

import rehypeRepoLinks from "./src/plugins/rehype-repo-links.mjs";

const base = "/openinference";

// https://astro.build/config
// Deployed to GitHub Pages: https://docs.astro.build/en/guides/deploy/github/
export default defineConfig({
  site: "https://arize-ai.github.io",
  base,
  // Emit `spec/llm_spans.html` rather than `spec/llm_spans/index.html` so the
  // URLs the Jekyll build published keep working.
  build: { format: "preserve" },
  markdown: {
    rehypePlugins: [[rehypeRepoLinks, { base }]],
  },
});

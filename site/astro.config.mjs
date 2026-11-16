// @ts-check
import { defineConfig } from "astro/config";

import { shikiConfig } from "./src/lib/shiki.mjs";
import rehypeReadmeIntro from "./src/plugins/rehype-readme-intro.mjs";
import rehypeRepoLinks from "./src/plugins/rehype-repo-links.mjs";
import rehypeTableScroll from "./src/plugins/rehype-table-scroll.mjs";

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
    // The Pierre theme, in light and dark: see src/lib/shiki.mjs.
    shikiConfig,
    rehypePlugins: [
      [rehypeRepoLinks, { base }],
      rehypeReadmeIntro,
      rehypeTableScroll,
    ],
  },
});

// @ts-check
import { existsSync } from "node:fs";
import { copyFile, readdir } from "node:fs/promises";

import { defineConfig } from "astro/config";

import { shikiConfig } from "./src/lib/shiki.mjs";
import rehypeReadmeIntro from "./src/plugins/rehype-readme-intro.mjs";
import rehypeRepoLinks from "./src/plugins/rehype-repo-links.mjs";
import rehypeTableScroll from "./src/plugins/rehype-table-scroll.mjs";

const base = "/openinference";

/**
 * Astro treats `spec/configuration.html` (the old configuration page, which
 * redirects) and `spec/configuration/index.html` (the configuration section's
 * directory URL) as one route and builds only the first. Copy such a page into
 * its directory so both URLs resolve; the old page already redirects to the
 * section's first page.
 * @type {import("astro").AstroIntegration}
 */
const specSectionIndexes = {
  name: "spec-section-indexes",
  hooks: {
    "astro:build:done": async ({ dir }) => {
      const spec = new URL("spec/", dir);
      for (const entry of await readdir(spec, { withFileTypes: true })) {
        if (!entry.isDirectory()) continue;
        const index = new URL(`${entry.name}/index.html`, spec);
        const page = new URL(`${entry.name}.html`, spec);
        if (!existsSync(index) && existsSync(page)) await copyFile(page, index);
      }
    },
  },
};

// https://astro.build/config
// Deployed to GitHub Pages: https://docs.astro.build/en/guides/deploy/github/
export default defineConfig({
  site: "https://arize-ai.github.io",
  base,
  // Emit `spec/spans/llm.html` rather than `spec/spans/llm/index.html`, the
  // URL style the Jekyll build published. The old flat spec pages redirect
  // from src/pages/spec/[moved].astro.
  build: { format: "preserve" },
  integrations: [specSectionIndexes],
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

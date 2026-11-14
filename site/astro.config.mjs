// @ts-check
import { defineConfig } from "astro/config";
import pierreDark from "@pierre/theme/pierre-dark";
import pierreLight from "@pierre/theme/pierre-light";

import rehypeReadmeIntro from "./src/plugins/rehype-readme-intro.mjs";
import rehypeRepoLinks from "./src/plugins/rehype-repo-links.mjs";
import rehypeTableScroll from "./src/plugins/rehype-table-scroll.mjs";

const base = "/openinference";

/**
 * Pierre's themes are typed as readonly objects, which Shiki's mutable theme
 * type doesn't accept; the data is a standard TextMate theme, so pass it as one.
 * @param {unknown} theme
 */
const shikiTheme = (theme) =>
  /** @type {NonNullable<import("astro").ShikiConfig["theme"]>} */ (theme);

// https://astro.build/config
// Deployed to GitHub Pages: https://docs.astro.build/en/guides/deploy/github/
export default defineConfig({
  site: "https://arize-ai.github.io",
  base,
  // Emit `spec/llm_spans.html` rather than `spec/llm_spans/index.html` so the
  // URLs the Jekyll build published keep working.
  build: { format: "preserve" },
  markdown: {
    // Code blocks use the Pierre theme (https://diffs.com/theme), as on the
    // Arize Labs site. Shiki emits the light colours inline and the dark ones
    // as --shiki-dark variables; global.css switches to them with the theme.
    shikiConfig: {
      themes: { light: shikiTheme(pierreLight), dark: shikiTheme(pierreDark) },
    },
    rehypePlugins: [
      [rehypeRepoLinks, { base }],
      rehypeReadmeIntro,
      rehypeTableScroll,
    ],
  },
});

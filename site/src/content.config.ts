import { defineCollection } from "astro:content";
import { glob } from "astro/loaders";

/** Keep the file name (minus `.md`) as the entry id, e.g. `README.zh-TW`. */
const fileNameId = ({ entry }: { entry: string }) =>
  entry.replace(/\.md$/, "");

/** The repository's root README, rendered as the landing page. */
const root = defineCollection({
  loader: glob({ pattern: "README.md", base: "..", generateId: fileNameId }),
});

/** The OpenInference specification. */
const spec = defineCollection({
  loader: glob({ pattern: "*.md", base: "../spec", generateId: fileNameId }),
});

/** Translations of the root README. */
const i18n = defineCollection({
  loader: glob({
    pattern: "README.*.md",
    base: "../docs/i18n",
    generateId: fileNameId,
  }),
});

export const collections = { root, spec, i18n };

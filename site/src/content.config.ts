import { defineCollection } from "astro:content";
import { glob } from "astro/loaders";

/**
 * Keep the path (minus `.md`) as the entry id, e.g. `README.zh-TW` or, in the
 * spec's subdirectories, `spans/llm`.
 */
const fileNameId = ({ entry }: { entry: string }) =>
  entry.replace(/\.md$/, "");

/** The repository's root README, rendered as the landing page. */
const root = defineCollection({
  loader: glob({ pattern: "README.md", base: "..", generateId: fileNameId }),
});

/** The project's mission statement, rendered at /mission.html. */
const mission = defineCollection({
  loader: glob({ pattern: "MISSION.md", base: "..", generateId: fileNameId }),
});

/** The OpenInference specification, including its section directories. */
const spec = defineCollection({
  loader: glob({ pattern: "**/*.md", base: "../spec", generateId: fileNameId }),
});

/** Translations of the root README. */
const i18n = defineCollection({
  loader: glob({
    pattern: "README.*.md",
    base: "../docs/i18n",
    generateId: fileNameId,
  }),
});

export const collections = { root, mission, spec, i18n };

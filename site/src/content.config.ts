import { defineCollection } from "astro:content";
import { glob } from "astro/loaders";
import { z } from "astro/zod";

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

/** A GitHub handle: letters, digits and single hyphens, at most 39 characters. */
const githubHandle = z
  .string()
  .regex(/^[a-z\d](?:[a-z\d]|-(?=[a-z\d])){0,38}$/i, "a GitHub handle, without the @");

/**
 * News: one markdown file per post in `docs/news/`, its file name the URL
 * (`docs/news/site-launch.md` → `/news/site-launch.html`). Release notes stay
 * on GitHub; a post links to them with `releases`. See docs/news/README.md.
 */
const news = defineCollection({
  loader: glob({
    pattern: ["*.md", "!README.md"],
    base: "../docs/news",
    generateId: fileNameId,
  }),
  schema: z.object({
    title: z.string(),
    /** One or two sentences: the timeline, the lede, the meta description and the social card. */
    description: z.string(),
    pubDate: z.coerce.date(),
    updatedDate: z.coerce.date().optional(),
    /** GitHub handles, without the @. */
    authors: z.array(githubHandle).min(1),
    tags: z.array(z.string()).default([]),
    /** Release tags on GitHub (`python-openinference-instrumentation-v0.1.40`) the post announces. */
    releases: z.array(z.string()).default([]),
    /** Shown by `pnpm dev`, left out of the build. */
    draft: z.boolean().default(false),
  }),
});

export const collections = { root, mission, spec, i18n, news };

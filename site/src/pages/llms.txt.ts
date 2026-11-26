/**
 * `llms.txt` (https://llmstxt.org): what OpenInference is, then a link to
 * the markdown copy of each page an agent would read, with a line on what
 * it covers. The copies, the pages' links to them and `llms-full.txt` are
 * written after the build by `src/integrations/llms-txt.mjs`, which reads
 * this file for the pages `llms-full.txt` includes.
 */
import type { APIContext } from "astro";

import { FULL_SECTIONS, markdownRouteOf } from "../integrations/llms-txt.mjs";
import * as js from "../lib/js-api";
import { getNews, newsUrl } from "../lib/news";
import * as python from "../lib/python-api";
import { base, pages, routeOf, site } from "../lib/site";
import { getSpecSections, ledeOf, titleOf } from "../lib/spec";

interface Link {
  title: string;
  /** The page's route, e.g. `/spec/`. */
  route: string;
  notes?: string;
}

interface Section {
  title: string;
  links: Link[];
}

/** One line of the description: whitespace collapsed. */
const oneLine = (text: string) => text.replace(/\s+/g, " ").trim();

/** Packages in their groups' order: core first. */
function byGroup<Package extends { group: string }>({
  packages,
  groups,
}: {
  packages: Package[];
  groups: readonly { id: string }[];
}): Package[] {
  return groups.flatMap((group) => packages.filter((pkg) => pkg.group === group.id));
}

async function getSections(): Promise<Section[]> {
  const specPages = (await getSpecSections()).flatMap((section) => section.pages);
  const news = await getNews();
  return [
    {
      title: "Project",
      links: [
        {
          title: "README",
          route: base,
          notes:
            "What OpenInference is, and every instrumentation package for Python, JavaScript, Java and Go, with what it instruments",
        },
      ],
    },
    {
      title: "Specification",
      links: specPages.map((entry) => ({
        title: entry.id === "README" ? "Specification overview" : titleOf(entry),
        route: routeOf(`spec/${entry.id}.md`),
        notes: ledeOf(entry.body),
      })),
    },
    {
      title: "Python API",
      links: [
        { title: "Python API overview", route: python.pythonRoot, notes: pages.python.description },
        ...byGroup({ packages: python.packages, groups: python.GROUPS }).map((pkg) => ({
          title: pkg.name,
          route: python.routeOfPackage(pkg.name),
          notes: pkg.description,
        })),
      ],
    },
    {
      title: "JavaScript API",
      links: [
        { title: "JavaScript API overview", route: js.jsRoot, notes: pages.js.description },
        ...byGroup({ packages: js.packages, groups: js.GROUPS }).map((pkg) => ({
          title: pkg.name,
          route: js.routeOfPackage(pkg),
          notes: pkg.description,
        })),
      ],
    },
    {
      title: "Optional",
      links: [
        { title: pages.mission.title, route: routeOf("MISSION.md"), notes: pages.mission.description },
        ...news.map((post) => ({
          title: `News: ${post.data.title}`,
          route: newsUrl(post),
          notes: post.data.description,
        })),
      ],
    },
  ];
}

export async function GET(context: APIContext) {
  const absolute = (route: string) => new URL(route, context.site).href;
  const link = ({ title, route, notes }: Link) =>
    `- [${title}](${absolute(markdownRouteOf(route))})${notes ? `: ${oneLine(notes)}` : ""}`;
  const sections = await getSections();
  const text = [
    `# ${site.name}`,
    "> OpenInference is a set of semantic conventions and instrumentation libraries, built on OpenTelemetry, for tracing AI applications: their LLM calls, agents, tool use, retrieval and embeddings. Its traces work with Arize Phoenix, Arize AX and any OpenTelemetry backend.",
    [
      `The specification defines the span kinds and attributes every OpenInference trace uses. The instrumentation packages follow it for Python, JavaScript, Java and Go; this site documents the Python and JavaScript APIs. The source is at ${site.repo}.`,
      `Every page here is also markdown, at its URL plus \`.md\`; the links below are to those. ${absolute(`${base}llms-full.txt`)} is the ${FULL_SECTIONS.join(" and ")} pages in one file.`,
    ].join("\n\n"),
    ...sections.map((section) => [`## ${section.title}`, ...section.links.map(link)].join("\n")),
  ].join("\n\n");
  return new Response(`${text}\n`, { headers: { "Content-Type": "text/plain; charset=utf-8" } });
}

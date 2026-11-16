import { getCollection, type CollectionEntry } from "astro:content";

import { BRANCH, GITHUB_URL } from "../plugins/rehype-repo-links.mjs";
import { pagerOf, type DocsNav } from "./docs";
import { routeOf } from "./site";

type SpecEntry = CollectionEntry<"spec">;

/** The first `# Heading` of a markdown document, falling back to its id. */
export function titleOf(entry: { id: string; body?: string }): string {
  const match = entry.body?.match(/^#\s+(.+)$/m);
  return match ? match[1].trim() : entry.id;
}

/** The section directory of a spec page, e.g. `spans` for `spans/llm`. */
const sectionOf = (id: string) => (id.includes("/") ? id.slice(0, id.indexOf("/")) : "");

/** Links from `spec/README.md` to its pages, capturing the page id. */
const PAGE_LINK = /\]\(\.\/([^)#]+)\.md[)#]/g;

/**
 * Spec pages ordered the way `spec/README.md` links to them, followed by any
 * pages the README does not link to.
 */
async function getSpecPages(): Promise<SpecEntry[]> {
  const entries = await getCollection("spec");
  const readme = entries.find((entry) => entry.id === "README");
  const linked = [...(readme?.body ?? "").matchAll(PAGE_LINK)].map((match) => match[1]);
  const rank = (id: string) => {
    if (id === "README") return -1;
    const index = linked.indexOf(id);
    return index === -1 ? linked.length : index;
  };
  return entries.sort((a, b) => rank(a.id) - rank(b.id) || a.id.localeCompare(b.id));
}

/**
 * Section labels from `spec/README.md`: each `### Heading` labels the
 * directory of the first page linked below it.
 */
function sectionLabelsOf(readme: string): Map<string, string> {
  const labels = new Map<string, string>();
  let heading: string | undefined;
  for (const line of readme.split("\n")) {
    const match = line.match(/^###\s+(.+)$/);
    if (match) {
      heading = match[1].trim();
      continue;
    }
    for (const [, id] of line.matchAll(PAGE_LINK)) {
      const dir = sectionOf(id);
      if (heading && dir && !labels.has(dir)) labels.set(dir, heading);
    }
  }
  return labels;
}

export interface SpecSection {
  /** The directory under `spec/`, e.g. `spans`; empty for top-level pages. */
  dir: string;
  /** The README heading over the section's links, or the directory name. */
  label?: string;
  pages: SpecEntry[];
}

/**
 * The spec's sections, one per directory under `spec/`, in the order
 * `spec/README.md` first links to them. Top-level pages (the README) come
 * first, unlabelled.
 */
export async function getSpecSections(): Promise<SpecSection[]> {
  const pages = await getSpecPages();
  const labels = sectionLabelsOf(pages.find((entry) => entry.id === "README")?.body ?? "");
  const sections: SpecSection[] = [];
  for (const entry of pages) {
    const dir = sectionOf(entry.id);
    let section = sections.find((candidate) => candidate.dir === dir);
    if (!section) {
      section = { dir, label: dir === "" ? undefined : (labels.get(dir) ?? dir), pages: [] };
      sections.push(section);
    }
    section.pages.push(entry);
  }
  return sections;
}

/** The sidebar, the previous and next pages, and the source link for a spec page. */
export async function getSpecNav(currentId: string): Promise<DocsNav> {
  const sections = await getSpecSections();
  const groups = sections.map((section) => ({
    label: section.label,
    items: section.pages.map((entry) => ({
      href: routeOf(`spec/${entry.id}.md`),
      label: entry.id === "README" ? "Overview" : titleOf(entry),
      current: entry.id === currentId,
    })),
  }));
  return {
    label: "Specification",
    groups,
    ...pagerOf(groups),
    source: `${GITHUB_URL}/blob/${BRANCH}/spec/${currentId}.md`,
  };
}

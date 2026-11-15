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

/**
 * Spec pages ordered the way `spec/README.md` links to them, followed by any
 * pages the README does not link to.
 */
async function getSpecPages(): Promise<SpecEntry[]> {
  const entries = await getCollection("spec");
  const readme = entries.find((entry) => entry.id === "README");
  const linked = [...(readme?.body ?? "").matchAll(/\]\(\.\/([^)#]+)\.md[)#]/g)].map(
    (match) => match[1],
  );
  const rank = (id: string) => {
    if (id === "README") return -1;
    const index = linked.indexOf(id);
    return index === -1 ? linked.length : index;
  };
  return entries.sort((a, b) => rank(a.id) - rank(b.id) || a.id.localeCompare(b.id));
}

/** The sidebar, the previous and next pages, and the source link for a spec page. */
export async function getSpecNav(currentId: string): Promise<DocsNav> {
  const pages = await getSpecPages();
  const items = pages.map((entry) => ({
    href: routeOf(`spec/${entry.id}.md`),
    label: entry.id === "README" ? "Overview" : titleOf(entry),
    current: entry.id === currentId,
  }));
  const groups = [{ items }];
  return {
    label: "Specification",
    groups,
    ...pagerOf(groups),
    source: `${GITHUB_URL}/blob/${BRANCH}/spec/${currentId}.md`,
  };
}

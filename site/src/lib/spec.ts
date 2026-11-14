import { getCollection, type CollectionEntry } from "astro:content";

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
export async function getSpecPages(): Promise<SpecEntry[]> {
  const entries = await getCollection("spec");
  const readme = entries.find((entry) => entry.id === "README");
  const linked = [
    ...(readme?.body ?? "").matchAll(/\]\(\.\/([^)#]+)\.md[)#]/g),
  ].map((match) => match[1]);
  const rank = (id: string) => {
    if (id === "README") return -1;
    const index = linked.indexOf(id);
    return index === -1 ? linked.length : index;
  };
  return entries.sort(
    (a, b) => rank(a.id) - rank(b.id) || a.id.localeCompare(b.id),
  );
}

export async function getSpecSidebar(currentId: string) {
  const base = import.meta.env.BASE_URL.replace(/\/?$/, "/");
  const pages = await getSpecPages();
  return {
    heading: "Specification",
    items: pages.map((entry) => ({
      href:
        entry.id === "README"
          ? `${base}spec/`
          : `${base}spec/${entry.id}.html`,
      label: entry.id === "README" ? "Overview" : titleOf(entry),
      current: entry.id === currentId,
    })),
  };
}

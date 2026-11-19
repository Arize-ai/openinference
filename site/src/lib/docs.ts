/**
 * The navigation DocsLayout draws around a page: the rail of pages (in
 * groups), the previous and next pages, and the source link. The spec
 * (`lib/spec.ts`) and the API references (`lib/js-api.ts`,
 * `lib/python-api.ts`) each build one.
 */

export interface DocsLink {
  href: string;
  label: string;
}

export interface DocsGroup {
  /** A label over the group in the rail; omit it for a single, flat list. */
  label?: string;
  items: (DocsLink & { current: boolean })[];
}

export interface DocsNav {
  /** The rail's eyebrow and the narrow disclosure's label, e.g. `Specification`. */
  label: string;
  groups: DocsGroup[];
  previous?: DocsLink;
  next?: DocsLink;
  /** The page's source on GitHub. */
  source: string;
  /** The source link's text. */
  sourceLabel?: string;
}

/** The previous and next pages around the current one, across groups. */
export function pagerOf(groups: DocsGroup[]): Pick<DocsNav, "previous" | "next"> {
  const items = groups.flatMap((group) => group.items);
  const index = items.findIndex((item) => item.current);
  const link = (item: DocsLink | undefined) =>
    item ? { href: item.href, label: item.label } : undefined;
  return {
    previous: index > 0 ? link(items[index - 1]) : undefined,
    next: index >= 0 ? link(items[index + 1]) : undefined,
  };
}

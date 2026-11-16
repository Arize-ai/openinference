// @ts-check
/**
 * Wraps every markdown table in `<div class="table-scroll">`, so a table
 * wider than its column scrolls inside itself instead of pushing the page
 * sideways. See "Tables" in DESIGN.md.
 */
export default function rehypeTableScroll() {
  /** @param {any} tree */
  return (tree) => {
    /** @param {any} node */
    const visit = (node) => {
      if (!node.children) return;
      node.children = node.children.map((/** @type {any} */ child) => {
        if (child.type === "element" && child.tagName === "table") {
          return {
            type: "element",
            tagName: "div",
            properties: { className: ["table-scroll"], tabIndex: 0, role: "region", ariaLabel: "Table" },
            children: [child],
          };
        }
        visit(child);
        return child;
      });
    };
    visit(tree);
  };
}

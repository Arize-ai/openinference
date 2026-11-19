// @ts-check
/**
 * Drops the README's GitHub opening from the landing page.
 *
 * The root README and its translations open with a GitHub banner (the logo,
 * the language switcher and two shields.io badges, as centred raw HTML)
 * followed by a one-paragraph introduction. The site draws its own hero
 * (the tagline, the calls to action and the language picker), so this
 * plugin drops the banner and the introduction; they stay in the README
 * for GitHub. Other markdown (the spec) is left alone.
 */
import { repoPathOf } from "./rehype-repo-links.mjs";

const README = /^(?:README|docs\/i18n\/README\.[\w-]+)\.md$/;
const BANNER = /^\s*<p\s+align=["']center["']/i;

/** @param {any} node */
const isBlank = (node) => node.type === "text" && !node.value.trim();

export default function rehypeReadmeIntro() {
  /**
   * @param {any} tree
   * @param {{ path?: string, history?: string[], data: any }} file
   */
  return (tree, file) => {
    const repoPath = repoPathOf(file);
    if (!repoPath || !README.test(repoPath)) return;

    const children = tree.children;
    let i = 0;
    while (
      i < children.length &&
      (isBlank(children[i]) || (children[i].type === "raw" && BANNER.test(children[i].value)))
    ) {
      i += 1;
    }
    const intro = children[i];
    if (intro?.type === "element" && intro.tagName === "p") i += 1;
    children.splice(0, i);
  };
}

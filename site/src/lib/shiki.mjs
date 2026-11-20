// @ts-check
import pierreDark from "@pierre/theme/pierre-dark";
import pierreLight from "@pierre/theme/pierre-light";

/**
 * Pierre's themes are typed as readonly objects, which Shiki's mutable theme
 * type doesn't accept; the data is a standard TextMate theme, so pass it as one.
 * @param {unknown} theme
 */
const shikiTheme = (theme) =>
  /** @type {NonNullable<import("astro").ShikiConfig["theme"]>} */ (theme);

/**
 * Code blocks use the Pierre theme (https://diffs.com/theme), as on the
 * Arize Labs site. Shiki emits the light colours inline and the dark ones
 * as --shiki-dark variables; global.css switches to them with the theme.
 *
 * Shared by the markdown pipeline (astro.config.mjs) and the API
 * references, which render doc comments themselves (src/lib/api.ts).
 * @type {import("astro").ShikiConfig}
 */
export const shikiConfig = {
  themes: { light: shikiTheme(pierreLight), dark: shikiTheme(pierreDark) },
};

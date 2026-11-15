# Design guidelines

How the OpenInference site looks, and the rules that keep it consistent. The
site should read as clean, quiet and precise: paper, ink, one accent, generous
space, and nothing that isn't doing a job.

Values live in [`src/styles/tokens.css`](./src/styles/tokens.css); this
document says which ones to use where. When code and this document disagree,
fix one of them in the same change.

The palette, type and component vocabulary come from the
[Arize Labs site](https://github.com/arize-labs/website) (its `DESIGN.md` and
`src/styles/tokens.css`). Keep the two in step: if a brand value changes
there, change it here.

## Principles

- **The markdown is the content.** Every page renders markdown that lives
  elsewhere in the repository and is written for GitHub. Style the elements
  it produces; never add classes to it or fork it for the site. If a page
  needs chrome (a hero, a sidebar), the layout draws it.
- **Centre the column, not the text.** Columns are centred in the page; text
  inside them is left-aligned, titles included. The one exception is the
  landing hero, which centres its tagline and actions.
- **Two measures, no others.** The 880px page (`--page-max`), whose text
  and tables both fill it, on the landing page and every spec page; and the
  1400px wide frame (`--frame-max`, as on the Arize Labs site) that holds the
  header, the footer and the docs rails. Don't invent a third.
- **One left edge.** Within a page, text and tables start at the same left
  edge.
- **Balance.** Pages are symmetric about the centre line. When something
  sits beside the page, something of the same width sits on the other side,
  or neither does.
- **Tokens only.** Colours, type sizes, spacing and widths come from
  `tokens.css`. A literal pixel value in a component is a bug unless it is a
  breakpoint (CSS can't read custom properties in media queries) or a
  hairline (1px rules, 2px accent bars).
- **One accent.** Signal orange (`--accent`) marks one thing at a time: a
  hover, the current page, a quote rule, the focus ring. It is never a fill
  behind text, and never a second brand colour.
- **Texture once.** Texture (the dot ground and the gutter rulers)
  appears only on the landing page.
  Everywhere else the page is flat paper or ink.

## Page types

Every page but one is one of two types. The layout decides everything
below the header.

| Type | Layout | Width | Pages |
|---|---|---|---|
| Landing | `ReadmeLayout` | Hero and README on the 880px page, centred | `/`, `/docs/i18n/README.*.html` |
| Docs | `DocsLayout` | Wide frame: pages rail, the page, an "On this page" rail | `/spec/`, `/spec/*.html`, `/js/`, `/js/*.html`, `/python/`, `/python/*.html` |

Both wrap [`BaseLayout`](./src/layouts/BaseLayout.astro), which owns
`<head>`, the theme script, the header and the footer. Pages never use
`BaseLayout` directly, except the brand page (`/brand.html`), which is one
of a kind: see [Brand](#brand).

The API references at `/js/` and `/python/` are docs pages too: see
[API reference](#api-reference).

### Landing

The root README and its translations. Top to bottom:

The layout and page markings are ported from the Arize Labs home page
(github.com/arize-labs/website: `components/sections/Hero.astro`,
`components/PageMarkings.astro`), without its glass mark or its art band.

1. **Hero.** Centred, on the page column, and all in the first screen:
   the tagline (`site.tagline`, "The Open Standard for Agent
   Observability") as the `h1`, `--space-9` under the header, at
   `--text-display`, weight 600, `--tracking-display`, balanced, at most
   18ch; then, `--space-7` under it, two buttons (**Read the spec**, solid;
   **View on GitHub**, outline). No wordmark or mark: the header carries
   the wordmark, and the hero doesn't repeat it. No lede and no art: the
   tagline says it. The language picker is off the landing page for now.
2. **Pillars.** Three `.tick-card`s on the page column, one per part of
   the project: Specification, Instrumentation, Any backend. Each has a
   title, a sentence in `--soft` and a mono caps link to where to read
   more (the spec, the README's `#instrumentation`,
   `#supported-destinations`). One column on phones.
3. **README.** The rest of the README on the 880px page, centred under
   the hero. Each `##` is a section, ruled from the one before with a 1px
   `--rule` and `--space-8` above and below:
   - the `##` heading at `--text-feature`, weight 600, `--tracking-tight`;
   - the paragraph under it in `--soft`;
   - `###` (Libraries, Examples) as a mono caps label in `--dim`;
   - lists as ruled rows (a 1px `--rule` between items), without bullets;
   - a note after a table (such as the Go version) at `--text-small`.

   Text and tables both fill the page, as on spec pages, so there is one
   left edge, one right edge, and the column sits on the hero's centre
   line. Don't narrow the text inside it: a 720px measure pinned to the
   left of a wider column leaves the right side empty and pulls the page
   off centre.
4. **Page markings.** A dot ground in the margins beyond the frame's edges
   (`--frame-max`, holding `--frame-edge` from a narrower window), and
   gutter rulers on those edges from under the hero's tagline to the README, and
   alongside the packages, with a registration cross at each `##`
   (`--marking-opacity`). None on phones.

Third-party work (the typefaces) is listed in
`src/lib/credits.ts` and shown on `/credits.html`, linked from the footer.

The README's GitHub opening (banner, language links, badges and the
introduction paragraph) is for GitHub: `rehype-readme-intro` drops it, so
the README stays the single source and the site draws its own hero. The
layout styles the README's own `##` and `###` elements; nothing else is
added to the markdown. A translation's body gets its `lang` and `dir`; the
header, the hero and the footer stay English.

### Docs

A spec page. The markdown's `# Title` is the title block, so don't add
another. The page is symmetric: whatever surrounds it, it sits on the
centre line, under the header's.

- **The page.** `--page-max` (880px) wide, centred. Text, headings, lists,
  tables and the page footer all fill it, so there is one left edge and one
  right edge. Never centre narrower text over a wider table: two left edges
  look like a mistake.
- **Wide (1360px and up).** Three columns on the wide frame: the spec's pages
  on the left, the page in the middle, and **On this page** (the page's `##`
  sections) on the right. Both rails are `--sidebar-width`, so the middle
  column is centred. Rails are sticky under the header, set at
  `--text-small` in `--dim`, with one 1px `--rule` down their left side.
- **Narrower.** The rails don't take space from the page. Both go; the spec's
  pages fold into a disclosure above the title (`Specification` and the
  current page's name). The page stays 880px, centred.
- **Current page and section.** The current page in the left rail is `--fg`,
  weight 600, with a 2px `--accent` bar over the rule. The section being
  read in the right rail gets `--fg` and an ink bar, not the accent, so
  there is still one accent on screen.
- **Under the page.** **Edit this page on GitHub**, then the previous and
  next pages: mono caps direction labels over the page titles at
  `--text-h4`, which take the accent on hover.

The header and footer sit on the wide frame on every page, so they don't
shift as you move between pages, and on docs pages the logo lines up with the
left rail and the nav with the right one.

### API reference

The JavaScript (`/js/`) and Python (`/python/`) references are one design,
drawn by the shared components in `src/components/api/` and styled by
`src/styles/api.css`; only how a language writes a signature differs. They
are docs pages (`DocsLayout`) drawn from the data the scripts extract, not
from markdown, so the page draws its own `h1`. The left rail lists the
packages in labelled groups: **Core** and **Instrumentation**, and for
JavaScript **Integrations** (instrumentations by what they instrument:
`openai`, not the full name). "On this page" goes a level deeper than on
spec pages: each module, then its members, set in; a member's name is
clipped rather than wrapped.

- **Package head.** A `JavaScript · Core` or `Python · Instrumentation`
  eyebrow, the package name as the title, its description in `--soft` at
  `--text-lede`, then a ruled row of facts (version, runtime, what it
  instruments) as eyebrow labels over `--text-small` values, then mono caps
  links (npm or PyPI, README, Source).
- **Install, Usage.** An install block; for an instrumentation, a short
  usage block.
- **Modules.** Each module (a Python module, or an entry in a JavaScript
  package's `exports`) is an `h2`, its import path in mono.
- **Objects.** Each member is an `h3` in mono with its kind (mono caps,
  `--dim`) and a **Source ↗** link at the right edge, ruled from the one
  before with a 1px `--rule`. Under it the signature, mono on the `--card`
  fill in a code block's frame, highlighted with the code blocks' Pierre
  theme (names linked to where they're documented), one per overload; a long one puts a parameter per line, and an options
  object a property per line. Then the docs. A class's constructor, methods
  and properties are `h4` objects, set in behind a 1px `--rule`; its
  attributes or properties, and an enum's members, are one table, as are a
  module's constants.
- **Docs.** Prose as `.prose`; Parameters, Returns and Raises as tables
  under mono caps labels; examples as highlighted code; other notes (See
  also, Deprecated) with a 2px `--rule` (not the accent) on the left.
- **Overview.** The packages as a table per group, and a search over every
  documented object: a `--surface` input, results as ruled rows.

## Header and footer

**Header.** As on the Arize Labs site, on the wide frame on every page.
Sticky, `--header-height` tall, on solid `--bg` with no rule underneath.
The wordmark alone (22px tall) on the left; the nav on the right in mono caps,
`--dim`, `--fg` on hover and for the current section. Links off the site
carry `↗`. Under 720px the nav folds into a **Menu** disclosure that opens
a panel under the bar, so it works without JavaScript.

**Footer.** A 1px `--rule`, then the Arize logo in `--fg` with a one-line
note on who maintains the project, two columns of mono caps links (the
first ends with **Brand**), and a
last row with the copyright and the theme switch (Auto, Light, Dark).

## Long-form content (`.prose`)

All rendered markdown sits in `.prose`, styled in
[`global.css`](./src/styles/global.css).

- **Title.** A spec page's `#` is `--text-h2`, weight 600,
  `--tracking-tight`, balanced wrapping.
- **Headings.** `##` at `--text-section` (28px) with room above it; `###` at
  `--text-h3`; `####` at `--text-h4`. All weight 600, no rules under them.
- **Body.** `--text-body-lg` (18px; 16px on phones), line height 1.65.
- **Links.** `--fg` with a faint underline; `--accent` on hover.
- **Inline code.** Geist Mono at 0.84em on a `--tint` chip. Package names in
  the first column of a table drop the chip: a column of chips is noise.
- **Code blocks.** Fenced blocks are highlighted with the Pierre theme, in
  light and dark, framed with a 1px `--rule`.
- **Tables.** Ruled rows, mono caps headers over a `--fg` rule, figures in
  tabular numerals. `rehype-table-scroll` puts every table in a
  `.table-scroll` box, so a wide table scrolls inside itself and the page
  never scrolls sideways.
- **Badges.** shields.io badges are greyscale, since their blue isn't in the
  palette, and regain their colour on hover.
- **Quotes.** `>` blockquotes: a 2px accent rule on the left, `--soft` text.

## Type

| Token | Size | Use |
|---|---|---|
| `--text-h2` | 36 to 56px | Spec page titles |
| `--text-display` | 42 to 88px | The landing hero's tagline |
| `--text-feature` | 32 to 48px | Landing section headings (README `##`) |
| `--text-section` | 28px | Prose `##` on spec pages |
| `--text-lede` | 19 to 24px | Page intros (the brand and credits pages) |
| `--text-h3` | 22px | Prose `###` |
| `--text-body-lg` | 18px | Prose |
| `--text-h4` | 17px | Prose `####`, pager titles |
| `--text-body` | 16px | UI text, prose on phones |
| `--text-small` | 14px | Sidebar links, tables, footer note |
| `--text-micro` | 11px | Eyebrows, nav, table headers, buttons (mono caps) |

Urbanist (`--font-sans`) for everything people read; Geist Mono
(`--font-mono`) for code, and in uppercase, tracked `--tracking-micro`, for
labels and navigation (the `.eyebrow` class). Both are self-hosted through
Fontsource, so the site makes no third-party font requests.

## Colour

| Token | Light | Dark | Use |
|---|---|---|---|
| `--bg` | paper `#f6f5f2` | `#141213` | Page ground |
| `--fg` | ink `#231f20` | paper `#f6f5f2` | Text, solid buttons, the logo |
| `--surface` | `#ffffff` | `#1c1a1b` | Panels that sit on the ground (the language picker) |
| `--soft` | `#4b4745` | `#bdb8b0` | Secondary text: the lede, the footer note |
| `--dim` | `#6f6b66` | `#9b958d` | Labels, nav, metadata |
| `--rule` | `#d8d5ce` | `#3a3637` | Every 1px line |
| `--tint` | `--fg` at 6% | same | Inline code chips |
| `--card` | `--fg` at 3% | same | The fill of a `.tick-card` |
| `--tick` | `--fg` at 24% | same | A `.tick-card`'s corner ticks |
| `--accent` | signal `#ff5a1f` | same | The one highlighted thing |

Use the semantic tokens, never the palette (`--color-*`) directly. Every
token has a dark value; if something needs a colour that has no token, add
the token in both themes rather than hard-coding it.

**Theme.** Auto (the default) follows the system; Light and Dark are saved
per browser. `BaseLayout` sets `data-theme` before first paint, and without
JavaScript the `prefers-color-scheme` block in `tokens.css` applies.

## Brand

The OpenInference mark, Keyway, is six sides around a nut and bolt, with
one keyway cut at the bottom: from a distance the O of
OpenInference, closer the bolt over its slot is the i. The wordmark is
Urbanist SemiBold, outlined. The two are used apart, never
locked up side by side: the header carries the wordmark; the mark is the
icon, the favicon and the avatar. The canonical artwork is in
[`src/assets/brand/`](./src/assets/brand/), as single-colour outlines.
Never redraw the mark or retype the wordmark.

Six rules fix the mark, in Labs units (Labs side = 1000): the Labs height
(757.77), a ring 135.5 thick (outer flat to window flat), a bolt of r 70.5,
outer corners of r 123, window corners of r 70.5 (the keyway's mouth too)
and the Labs outer corner, r 25, at the keyway's shoulders. The keyway is as
wide as the bolt (141): its walls run straight down from the bolt's sides,
so the bolt sits over its slot edge on edge, the dot of the i. The hexagon
is solved so the rounded crown and shoulders meet the height exactly, and
the window is the hexagon brought in by the ring's thickness. The rules live in
[`src/lib/mark.mjs`](./src/lib/mark.mjs);
`pnpm brand` ([`scripts/build-brand.mjs`](./scripts/build-brand.mjs)) draws
the mark, the wordmark in paper (for the README banner's dark mode), the
icon, the SVG favicon and its PNG fallbacks (`favicon-32.png`,
`apple-touch-icon.png`, rasterised with `sharp`) from them, so change the rules there
and rerun it, never edit the SVGs by hand. The wordmark is the one drawn
file the script recolours but doesn't draw.

The mark is one colour, always: the bolt is never picked out in another.
It is never rotated (the keyway stays at the bottom), its keyway is never
closed, and its bolt is never moved or resized. Clear space is one bolt
diameter (141) on every side, and the mark is never drawn under 16px tall.

[`src/lib/brand.ts`](./src/lib/brand.ts) reads the artwork and quotes the
same geometry (the build fails if the files are out of date);
[`src/pages/brand.astro`](./src/pages/brand.astro), the brand guidelines
(Mark, Anatomy, Wordmark, Clear space, Colour, Typography), draws from it,
so the page can't drift from the files. Like the
[OpenAI brand page](https://openai.com/brand/), it is mostly artwork: a
heading and a download per section, captions only where a rule needs one.
Anatomy is the one exception: the construction drawing beside a table of
every dimension and where it comes from, both computed from the geometry,
with the accent on the keyway's walls running up the bolt's sides.
It sits on the 880px page, like the README and the spec; its panels keep a
fixed white or ink ground in both themes (the `--color-*` palette tokens),
so the artwork always shows as specified.

[`BrandArt`](./src/components/brand/BrandArt.astro) inlines a piece of
artwork in `currentColor`, so it follows the theme: the header uses it for
the wordmark. The favicon (`public/favicon.svg`) is
the mark, ink on light browser chrome and paper on dark, as tall as its
square so that at 16px it is the mark's minimum size. The icon (the PNG
favicon, the iOS home screen icon, and the avatar) is the mark in paper on
an ink square, one bolt diameter clear on every side.

## Spacing and breakpoints

The spacing scale runs from `--space-1` (4px) to `--space-9` (96px). Use the
scale; don't add in-between values.

Breakpoints, as literals in media queries:

- **1100px:** gutters shrink from 48px to 20px.
- **1360px:** below it, the docs rails go and the spec's pages fold into a
  disclosure.
- **720px:** prose drops to 16px, the hero tightens, the footer and pager
  stack.

## Checklist for a change

- [ ] Content changes go in the markdown (`README.md`, `spec/`,
      `docs/i18n/`), not in the site.
- [ ] A new page uses `ReadmeLayout` or `DocsLayout`; no hand-built title
      or header.
- [ ] Text left-aligned (outside the landing hero); only the column is
      centred.
- [ ] Only tokens for colour, type and spacing; one accent on screen at a
      time.
- [ ] Checked at 1440px and 390px wide, in light and dark: no sideways
      scroll, and no table wider than its column.
- [ ] Checked a right-to-left translation (`README.ar.html`) if the change
      touches the landing page.

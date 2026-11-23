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
  inside them is left-aligned, titles included.
- **Two measures, no others.** The 880px page (`--page-max`), whose text and
  tables both fill it, on the landing page and every spec page; and the
  1400px wide frame (`--frame-max`, as on the Arize Labs site) that holds the
  header, the footer, the docs rails and the landing page's split hero. Don't
  invent a third.
- **One left edge.** Within a page, text and tables start at the same left
  edge.
- **Balance.** Pages are symmetric about the centre line. When something
  sits beside the page, something of the same width sits on the other side,
  or neither does.
- **Tokens only.** Colours, type sizes, spacing and widths come from
  `tokens.css`. A literal pixel value in a component is a bug unless it is a
  breakpoint (CSS can't read custom properties in media queries), a
  hairline (1px rules, 2px accent bars), or part of a component's own
  geometry block: one set of named custom properties at the top of the
  component (the trace hero's `.stage`), which its layouts change and its
  rules only read.
- **One accent.** Signal orange (`--accent`) marks one thing at a time: a
  hover, the current page, a quote rule, the focus ring. It is never a fill
  behind text, and never a second brand colour.
- **Texture once.** Texture (the gutter rulers) appears only on the
  landing page, beside the packages; there is no dot ground.
  Everywhere else the page is flat paper or ink.

## Page types

Every page but one is one of two types. The layout decides everything
below the header.

| Type | Layout | Width | Pages |
|---|---|---|---|
| Landing | `ReadmeLayout` | A split hero on the wide frame, then the README on the 880px page, centred | `/`, `/docs/i18n/README.*.html` |
| Docs | `DocsLayout` | Wide frame: pages rail, the page, an "On this page" rail | `/spec/`, `/spec/*.html`, `/js/`, `/js/*.html`, `/python/`, `/python/*.html` |

Both wrap [`BaseLayout`](./src/layouts/BaseLayout.astro), which owns
`<head>`, the theme script, the header and the footer. Pages never use
`BaseLayout` directly, except the brand page (`/brand.html`), which is one
of a kind: see [Brand](#brand).

The API references at `/js/` and `/python/` are docs pages too: see
[API reference](#api-reference). The news (`/news/`) is a third, small
family on the 880px page: see [News](#news).

### Landing

The root README and its translations. Top to bottom:

The layout and page markings are ported from the Arize Labs home page
(github.com/arize-labs/website: `components/sections/Hero.astro`,
`components/PageMarkings.astro`), without its glass mark or its art band.

1. **Hero.** On the wide frame like the header (the tagline starts under
   the wordmark, the demo ends under the nav), all in the first screen at
   1280 × 800 and up, `--space-8` under the header. The words sit beside the
   trace hero (6 : 11, 5 : 12 under 1360px; the same at every width past the frame's 1400px, so a wider window never shrinks the demo), centred on it: the tagline
   (`site.tagline`) as the `h1`, weight 600, `--tracking-display`, in two
   whole lines, "Telemetry" over "for the AI Frontier", sized
   to its column (up to 64px) so neither ever breaks; one short sentence of
   lede in `--soft` at `--text-hero-lede`; two buttons (**Read the spec**, solid; **View on
   GitHub**, outline). Under both, the three pillars in a row, one per part
   of the project (Specification, Instrumentation, Vendor agnostic), set
   off by type and space alone: each title is the link (to the spec, the
   README's `#instrumentation`, `#supported-destinations`), with an arrow
   that moves on hover, over one short sentence in `--dim`. No wordmark or
   mark: the header carries the wordmark. The language picker is off the
   landing page for now. Under 1240px the hero stacks on the 880px page,
   one left edge with the README: the words, the pillars, then the demo;
   while the page markings show (721 to 1240px) the landing page keeps a
   gutter and `--space-6` from the window's edge; on a phone the pillars
   stack.
2. **Trace hero.** The hero's right half: one support-agent run replayed on
   a single clock. A lens at the top shows what the person sees (the
   session) or what runs (the agent's code, scrolled to the running line);
   under it, the OpenInference trace it records and the span in view. It is a component, not markdown
   ([`TraceHero.astro`](./src/components/hero/TraceHero.astro)); every value
   it shows is in [`model.ts`](./src/components/hero/model.ts).
   - **The run is real-shaped.** The code is the agent, instrumented with
     the OpenAI and requests instrumentors and the `@tracer` decorators, and
     the spans are the tree it emits: names, kinds, nesting and attribute
     keys as the instrumentors write them. Change the code and the spans
     together, and check them by running the sample.
   - **The stage.** No card or frame: one column whose structure is
     spacing and one hairline, under the tabs. No
     chrome the story doesn't need: no status readout, no file header, no
     composer, no message actions, no speed control.
   - **The session.** Quiet, like a real chat: the question, then one
     line that glimmers with what the agent is doing now ("Searching the
     help center…"), its steps collapsed under it (a click opens them), then
     the answer streaming in token by token, as a model writes it: plain
     text, no fade or caret. When the
     answer starts, the line settles into "Worked for 6.0s".
   - **The trace.** Drawn like the rest of the site, as a wireframe, and as
     plainly as it can be: a span tree beside a timeline. The tree is the
     kind's glyph and the span's name, set in by depth, with hairline
     hierarchy lines that grow with the trace; nothing frames it. Each span
     is a solid square rectangle in its kind's full colour, with its
     duration beside it; a span still running is only
     its dashed outline, and fills in when it ends. Rows ease in as their
     spans start. The agent is no exception: its dashed outline is the
     envelope around the work while it runs, and it fills in when the run
     ends. The failed request is hatched in the accent; the retry is simply
     the next row. Gridlines are dotted. No
     boxes, gradients or glows.
   - **The detail.** What OpenInference recorded on the span in view, and
     nothing it hasn't: inputs from the start, outputs, token counts and the
     status (OK, ERROR, or UNSET for a plain HTTP success) only once the
     span ends. It follows the agent's own steps, not every short call inside
     them, holding each at least two seconds, and holds the failed request
     just after it fails; hovering a row shows any span.
   - **Colour.** Span kinds take the `--kind-*` tokens on their glyph and
     line, never on text; the accent marks only the request that failed, the
     one thing the person never saw.
   - **Glyphs.** [`glyphs.ts`](./src/components/hero/glyphs.ts): Phoenix's
     span-kind shapes redrawn as bare strokes on a 16px grid, in the kind's
     colour, so the hero and Phoenix read as one system.
   - **Motion.** Everything is a function of trace time; one frame loop,
     paused off screen and in a hidden tab. With reduced motion it shows the
     finished trace, still, until someone presses play. Without JavaScript
     the server's finished trace stands.
   - **Its geometry is named once**, at the top of `.stage`: the lens's
     height, the tree's width and indent, the row pitch, the glyph, bar,
     icon, control and attribute sizes. The wide and phone layouts change
     only these, never the rules that read them, and every offset derives
     from them (a tree line sits at half a glyph).
   - **It lays itself out by its own width** (a container query), wherever
     the page puts it. 760px and wider, the trace takes the right 60% and
     the full height, with larger names and bars: it is the hero's
     highlight. Beside it, the session (or the code) and, at its foot, the
     span in view, level with the trace's last row. The code rests just past
     its 13 lines of setup (a scroll up shows them) and never scrolls back
     into them while it follows the running line; the agent's lines are
     written short enough never to wrap (a long setup import wraps under
     itself, out of view), and spans find their code lines by content
     (`lineOf` in model.ts), so the code can change freely. Switching
     between Session and Code never moves anything: the trace sets the
     height and the lens fills it. Narrower, the lens sits over the trace,
     and the span in view under it. A short timeline labels every other
     second.
   - **The span in view** sits in a faint well (`--well`), so it
     reads as that span's record.
   - **Narrow.** On a phone everything fits the screen, never scrolling
     sideways: a narrower tree in a 10px face, every other tick, and each
     attribute's value under its key.
3. **Statement.** The mission in one paragraph (`site.statement`), on the
   880px page under the hero, one left edge with the README: display type
   at `--text-h2`, weight 500, `--tracking-tight`. Its words light from
   `--unlit` to `--fg` in reading order as it scrolls through the window,
   as on the Vercel Labs home page; without JavaScript, or with reduced
   motion, every word is lit. Under it, **Read the mission →**, set like a
   pillar's link, to `/mission.html`. No heading: the statement is the
   heading ([`Statement.astro`](./src/components/Statement.astro)).
4. **README.** The rest of the README on the 880px page, centred under
   the hero. Each `##` is a section, set apart by type and space alone (no
   rules), `--space-9` plus `--space-6` above it:
   - the `##` heading at `--text-feature`, weight 600, `--tracking-tight`;
   - the paragraph under it in `--soft`;
   - `###` (Libraries, Examples) as a mono caps label in `--dim`;
   - lists as rows without bullets or rules;
   - a note after a table (such as the Go version) at `--text-small`.

   Text and tables both fill the page, as on spec pages, so there is one
   left edge, one right edge, and the column sits on the hero's centre
   line. Don't narrow the text inside it: a 720px measure pinned to the
   left of a wider column leaves the right side empty and pulls the page
   off centre.
5. **Page markings.** Gutter rulers on the frame's edges (`--frame-max`,
   holding `--frame-edge` from a narrower window), alongside the packages (never beside the hero,
   where they would compete with the demo), with a registration cross at each `##`
   (`--marking-opacity`). None on phones.

Third-party work (the typefaces) is listed in
`src/lib/credits.ts` and shown on `/credits.html`, linked from the footer.

The mission is the repository's [`MISSION.md`](../MISSION.md), rendered on
`/mission.html` as plain `.prose` on the 880px page, linked from the
landing statement and the footer.
The line under its title is the page's lede (`--text-lede`, in `--soft`).

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
- **Copy page.** At the top of the page, floated to its end edge so the
  title's first line (or an API page's eyebrow) wraps short of it: a
  **Copy page** button that copies the page's markdown copy, joined to a
  toggle for a menu (view the markdown; open it in ChatGPT, Claude or
  Perplexity). Both are the code blocks' Copy button: mono caps on
  `--surface` in a 1px `--rule`, `--fg` on hover, `--accent` once copied.
  The menu is a `--surface` panel in a 1px `--rule`; an item's label takes
  the accent on hover. It sits outside the converted article, so it isn't
  in the markdown, and it is hidden without JavaScript.
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

### News

The posts in `docs/news/`, a timeline at `/news/` and one page per post,
on the 880px page with `BaseLayout`, like the mission. Release notes are
GitHub's: the site links to them and never copies them.

- **Timeline.** The page head (`News`, the lede in `--soft`), then two mono
  caps links: **Release notes on GitHub ↗** and **RSS**. The posts by year,
  newest first: the year as an eyebrow, then each post's day (mono caps,
  `--dim`) beside one 1px `--rule` that runs down the year, with a square
  marker on it (ink outline on the ground) level with the title. The title
  at `--text-h3`, the description in `--soft` under it, the tags as an
  eyebrow. The whole entry is the link; on hover the title and the marker
  take the accent. On a phone the day sits over the title, both beside the
  line.
- **Post.** The frontmatter is the title block, so the markdown starts at
  `##`: an eyebrow (**News**, linked, the date, the tags), the title as the
  `h1`, the description as the lede, then a byline of GitHub handles in
  `--text-small`. The body is `.prose`. If the post announces releases, a ruled
  **Release notes** list of their tags in mono chips, each to its notes on
  GitHub. Under it, **Edit this post on GitHub** and a pager like the docs
  pager: **← Newer** and **Older →** (the oldest post points back to the
  timeline).
- **Landing.** The newest post, for 60 days after its date, is one quiet
  line over the tagline: a **News** eyebrow, its title in `--soft` at
  `--text-small` (clipped, never wrapped), and an arrow that takes the
  accent on hover.

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
  light and dark, framed with a 1px `--rule`. Each has a Copy button in its top
  corner ([`CodeCopy`](./src/components/CodeCopy.astro)), shown
  on hover or focus and always on touch screens. API signatures don't.
- **Tables.** Ruled rows, mono caps headers over a `--fg` rule, figures in
  tabular numerals. `rehype-table-scroll` puts every table in a
  `.table-scroll` box, so a wide table scrolls inside itself and the page
  never scrolls sideways.
- **Badges.** shields.io badges are greyscale, since their blue isn't in the
  palette, and regain their colour on hover. The exception is the stability
  badges in the spec's tables (development, stable, deprecated): their
  colour is the information, so they keep it.
- **Quotes.** `>` blockquotes: a 2px accent rule on the left, `--soft` text.

## Type

| Token | Size | Use |
|---|---|---|
| `--text-h2` | 36 to 56px | Spec page titles |
| `--text-hero` | 18 to 64px, by its column | The landing hero's tagline: a share of its own column (a container query), so its two lines hold whole at any width |
| `--text-feature` | 32 to 48px | Landing section headings (README `##`) |
| `--text-section` | 28px | Prose `##` on spec pages |
| `--text-lede` | 19 to 24px | Page intros (the brand and credits pages) |
| `--text-h3` | 22px | Prose `###` |
| `--text-body-lg` | 18px | Prose |
| `--text-hero-lede` | 15 to 24px, by its column | The sentence under the tagline, from the same column, so the two keep their ratio |
| `--text-title` | 20px | Short titles in a row: the landing pillars |
| `--text-h4` | 17px | Prose `####`, pager titles |
| `--text-body` | 16px | UI text, prose on phones |
| `--text-small` | 14px | Sidebar links, tables, footer note |
| `--text-label` | 12px | Span names in the trace hero's wide layout |
| `--text-micro` | 11px | Eyebrows, nav, table headers, buttons (mono caps) |
| `--text-nano` | 10px | The trace hero's code, ticks and time chip; its span names on phones |

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
| `--card` | `--fg` at 3% | same | The fill behind an API signature |
| `--well` | `--fg` at 2.5% | same | The well that holds the trace hero's span in view |
| `--accent` | signal `#ff5a1f` | same | The one highlighted thing |
| `--kind-*` | Phoenix's kind hue, full chroma, L 56 (the LLM's amber L 72, the tool's gold L 80) | L 74 | Span kinds in the trace hero (agent, llm, tool, retriever, embedding, reranker, guardrail, http) |

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
every dimension, both computed from the geometry. The accent marks the
i: the bolt (its dot) and the keyway (its stem, drawn on up through the
window to just under the dot, as a lowercase i leaves a gap), filled and
called out. The keyway's walls run on up the bolt's sides as plain
construction lines. The page says nothing of
the Arize Labs mark the rules descend from (its "Labs units"): to a reader
they are the drawing's units, `u`.
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

## Social cards

Every page has a social card: a 1200 × 630 PNG at `og/<page>.png` (the
page's route, `index` for a directory), named in the page's `og:image` and
`twitter:image` (`summary_large_image`) by `BaseLayout`. The cards are drawn
at build time, from the same data as the pages, so a card can't disagree
with its page:

- [`src/lib/og/cards.ts`](./src/lib/og/cards.ts) lists the cards: the
  landing page, each spec page (its `# Title`, and the first sentence of
  its introduction, which is also the page's meta description), each API
  reference overview and package, the news timeline and each post (its
  date as the eyebrow), the brand page and credits. A page
  without a card of its own (a README translation, whose chrome is English)
  shares the landing card.
- [`src/lib/og/render.ts`](./src/lib/og/render.ts) draws them: satori lays
  out the type in Urbanist and Geist Mono as outlines (Fontsource's static
  WOFF files, since satori reads neither WOFF2 nor variable fonts), and
  sharp rasterises it.
- [`src/pages/og/[...card].png.ts`](./src/pages/og/[...card].png.ts)
  publishes them.

The cards are the site in miniature: flat paper, ink, one accent, and type
and space doing the work. Always the light theme, since a card can't follow
the reader's.

- **Frame.** The wordmark in ink, 40px tall, top left, and the mark in
  ink, 56px tall, top right, on the same centre line: the width of the
  card apart, never locked up. At the foot, a hairline `--rule` and the
  page's address in Geist Mono, `--dim`. 72px either side, 64px top and
  bottom.
- **Page card.** Centred between the two: the eyebrow (mono caps, `--dim`);
  the page title in
  Urbanist SemiBold, `--tracking-display`, as large as two lines allow (92px
  down to 48px); the description in `--soft`, at most two lines. A package
  name's shared start (`@arizeai/openinference-instrumentation-`) is set back
  in a fainter tone, so the part that names the package reads first.
- **Landing card.** The tagline in its two whole lines and the lede, beside
  the trace hero's run, finished, drawn as the hero draws it: kind glyphs and
  hairline tree lines, solid bars in each kind's colour (`--kind-*`, light),
  the failed request hatched in the accent, dotted
  gridlines a second apart. The trace ends on the foot's hairline. The
  failed request is the card's one accent.

To see a card, run `pnpm dev` and open `/openinference/og/<page>.png`.

## Spacing and breakpoints

The spacing scale runs from `--space-1` (4px) to `--space-9` (96px). Use the
scale; don't add in-between values.

Breakpoints, as literals in media queries:

- **1100px:** gutters shrink from 48px to 20px.
- **1360px:** below it, the docs rails go and the spec's pages fold into a
  disclosure.
- **720px:** prose drops to 16px, the hero tightens, the footer and pager
  stack.
- **1360px:** below it, the landing hero gives the demo more of the frame
  (5 : 12 instead of 6 : 11), so it keeps the session beside the trace.
- **1240px:** below it, the landing hero stacks: the words, the pillars,
  then the demo.
- **560px:** the trace hero fits a phone (a narrower tree, values under
  their keys).

The trace hero also lays itself out by its own width, a container query
rather than a breakpoint: from 760px wide it sets the session beside the
trace, and a trace narrower than 520px labels every other second.

## Checklist for a change

- [ ] Content changes go in the markdown (`README.md`, `spec/`,
      `docs/i18n/`, `docs/news/`), not in the site.
- [ ] A new page uses `ReadmeLayout` or `DocsLayout`; no hand-built title
      or header.
- [ ] Text left-aligned (outside the landing hero); only the column is
      centred.
- [ ] Only tokens for colour, type and spacing; one accent on screen at a
      time.
- [ ] Checked at 1440px and 390px wide, in light and dark: no sideways
      scroll, and no table wider than its column.
- [ ] A new page has a social card (`src/lib/og/cards.ts`); open it at
      `/openinference/og/<page>.png`.
- [ ] Checked a right-to-left translation (`README.ar.html`) if the change
      touches the landing page.

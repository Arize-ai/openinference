# OpenInference site

The [Astro](https://astro.build) project behind
<https://arize-ai.github.io/openinference/>.

The site renders markdown that lives elsewhere in the repository; nothing is
duplicated here. The content collections in `src/content.config.ts` load:

| Source               | Route                                |
| -------------------- | ------------------------------------ |
| `README.md`          | `/openinference/`                    |
| `spec/README.md`     | `/openinference/spec/`               |
| `spec/*.md`          | `/openinference/spec/<name>.html`    |
| `docs/i18n/*.md`     | `/openinference/docs/i18n/<name>.html` |
| `docs/news/*.md`     | `/openinference/news/<name>.html`    |

Relative links in that markdown are rewritten by
`src/plugins/rehype-repo-links.mjs`: links to published pages point at their
route, and links to anything else (package directories, source files) point at
the file on GitHub.

## News

`/openinference/news/` is a timeline of the project's news, newest first,
with an RSS feed at `/openinference/news/rss.xml`. Each post is a markdown
file in [`docs/news/`](../docs/news/README.md) with frontmatter (`title`,
`description`, `pubDate`, `authors` as GitHub handles, and optional `tags`,
`releases` and `draft`), checked by the `news` collection's schema in
`src/content.config.ts`, as posts are on the Arize Labs site. Release notes
are not written here: they stay on GitHub, and a post links to the releases
it announces with `releases`. [`docs/news/README.md`](../docs/news/README.md)
is the guide for writing one.

## llms.txt

The site publishes [`llms.txt`](https://llmstxt.org) for agents, at
`/openinference/llms.txt`, with a markdown copy of every page.

| File                         | What it is                                                     | Made by                                                           |
| ---------------------------- | -------------------------------------------------------------- | ----------------------------------------------------------------- |
| `llms.txt`                   | The index: what OpenInference is, then a link to each page's markdown with a line on what it covers | [`src/pages/llms.txt.ts`](./src/pages/llms.txt.ts), from the same data as the pages |
| `<page>.md`                  | A page as markdown, at its URL plus `.md` (`spec/spans/llm.html.md`, `spec/index.html.md`) | [`src/integrations/llms-txt.mjs`](./src/integrations/llms-txt.mjs), after the build |
| `llms-full.txt`              | The README and the spec in one file (`FULL_SECTIONS`)          | `src/integrations/llms-txt.mjs`, from the pages `llms.txt` lists  |

The markdown is converted from each page's built HTML, so it says what the
page says, the API references included. The element converted is the one
marked `data-markdown` (each layout's `<article>`); anything inside it that
only works in a browser is marked `data-markdown-ignore`. Each converted
page's `<head>` links to its copy (`rel="alternate"`) and to `llms.txt`
(`rel="describedby"`). A new page with that marker gets a copy; add it to
`llms.txt.ts` to list it. The build fails if `llms.txt` links to a page with
no copy.

`pnpm dev` serves `llms.txt`; the copies and `llms-full.txt` exist only after
`pnpm build` (see them with `pnpm preview`).

## API references

The JavaScript and Python API references are part of the site, built the
same way: a script extracts each package's public API to JSON in
`src/generated/` (not committed), and the pages render it with the shared
components in `src/components/api/`.

| Source                                                 | Route                                  |
| ------------------------------------------------------ | -------------------------------------- |
| `js/packages/<package>`                                | `/openinference/js/<package>.html`     |
| `python/<package>`, `python/instrumentation/<package>` | `/openinference/python/<package>.html` |

`/openinference/js/` and `/openinference/python/` are overviews: every
package by group, and a search over every documented object.

### JavaScript

[`scripts/js-api.mjs`](./scripts/js-api.mjs) (`pnpm js-api`) reads each
package in `js/packages/` with [TypeDoc](https://typedoc.org), using the
`js/` workspace's install, and writes `src/generated/js-api.json`;
[`src/lib/js-api.ts`](./src/lib/js-api.ts) and the pages in `src/pages/js/`
render it. A package's types import its siblings' built declarations, so
install and build `js/` first.

What's documented: each entry in a package's `exports` map (`.`,
`./utils`) is a module on its page, with everything it exports, less what's
marked `@internal`, `private` or `protected`. Doc comments are TSDoc:
`@param` (including `@param options.name` for an options object),
`@returns`, `@example`, `@remarks`, `@deprecated`, `@see`. `{@link Name}`
links to `Name` wherever this reference documents it, in any package;
OpenTelemetry types link to their docs through `externalSymbolLinkMappings`
in `js/typedoc.jsonc`. Constants with literal values are one table per
module. An instrumentation's page shows its `registerInstrumentations`
usage, and the library versions from its `InstrumentationNodeModuleDefinition`.

### Python

[`scripts/python-api.py`](./scripts/python-api.py) (`pnpm python-api`)
reads the packages' source with
[griffe](https://mkdocstrings.github.io/griffe/) and writes
`src/generated/python-api.json`; [`src/lib/python-api.ts`](./src/lib/python-api.ts)
and the pages in `src/pages/python/` render it. griffe parses the code
statically, so nothing is imported, and none of the libraries the
instrumentors patch need installing: the script's only dependency is griffe,
which `uv` installs from the script's inline metadata.

What's documented, per module: the names in `__all__`; without one, the
classes and functions defined in it whose names don't start with `_`. A
package is read from its top-level module down through its public
submodules, and an object re-exported higher up is documented there only.
Docstrings are Google style; reStructuredText roles (`` :func:`name` ``) link
to the object when the reference documents it. Each instrumentor's page also
documents `instrument()` and `uninstrument()`, which it inherits from
OpenTelemetry's `BaseInstrumentor`.

To change what a package's page shows, in either language, change the
package: its exports, its doc comments, its `package.json` or
`pyproject.toml` description.

## Design

Follow [`DESIGN.md`](./DESIGN.md) when changing the site's look or adding a
page: pick the layout (`ReadmeLayout` or `DocsLayout`) and use the tokens in
`src/styles/tokens.css` only.

## Development

Requires Node.js 22.23.4 or newer.

The API references need their data first (rerun after changing `js/` or
`python/`): the JavaScript one needs the `js/` workspace installed and built,
and the Python one needs [uv](https://docs.astral.sh/uv/).

```sh
# From the repository root
(cd js && pnpm install && pnpm run -r prebuild && pnpm run -r build)
cd site
pnpm install
pnpm js-api      # writes src/generated/js-api.json
pnpm python-api  # writes src/generated/python-api.json
pnpm dev      # http://localhost:4321/openinference/
pnpm build    # outputs to dist/
pnpm preview
```

Astro caches rendered markdown and doesn't notice when a plugin in
`src/plugins/` changes. `pnpm astro sync --force` refreshes the cache that
`pnpm build` reads (`node_modules/.astro/`), but the dev server reads
`.astro/data-store.json`: delete that file and restart `pnpm dev`.

## Deployment

Pushes to `main` run the workflow, which extracts the JavaScript API (in a
job that installs and builds `js/`) and the Python API, then builds the site with
[`withastro/action`](https://github.com/withastro/action) and deploys it with
`actions/deploy-pages`, following
[Astro's GitHub Pages guide](https://docs.astro.build/en/guides/deploy/github/).
The repository's Pages source must be set to **GitHub Actions**.

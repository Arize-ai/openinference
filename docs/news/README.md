# News

Posts for the [news timeline](https://arize-ai.github.io/openinference/news/)
on the OpenInference site: new instrumentation, changes to the spec, and the
releases worth knowing about. Release notes themselves stay on
[GitHub](https://github.com/Arize-ai/openinference/releases); a post links to
them.

## Writing a post

Add one markdown file here. Its name is the post's address:
`docs/news/langgraph-instrumentation.md` is published at
`/openinference/news/langgraph-instrumentation.html`, so pick a short,
lowercase, hyphenated name and don't rename it once it's published.

```markdown
---
title: LangGraph instrumentation for JavaScript
description: One or two sentences. They show on the timeline, under the title, in search results and on the social card.
pubDate: 2026-11-18
authors: [your-github-handle]
tags: [javascript, instrumentation]
releases: ["@arizeai/openinference-instrumentation-langchain@4.2.1"]
---

The body, in GitHub-flavored markdown. Don't repeat the title as a `#`
heading: the site draws it from the frontmatter. Start sections at `##`.
```

| Field         | Required | What it is                                                                                         |
| ------------- | -------- | -------------------------------------------------------------------------------------------------- |
| `title`       | yes      | The post's title.                                                                                  |
| `description` | yes      | One or two sentences: the timeline, the lede, the meta description and the social card.           |
| `pubDate`     | yes      | `YYYY-MM-DD`. The timeline is newest first.                                                        |
| `authors`     | yes      | GitHub handles, without the `@`. Each links to the profile.                                        |
| `tags`        | no       | Short lowercase labels, such as `python`, `javascript`, `spec`.                                    |
| `releases`    | no       | Git tags of the releases the post announces. Each links to its release notes on GitHub.            |
| `updatedDate` | no       | `YYYY-MM-DD`, when a post is corrected after it's published.                                       |
| `draft`       | no       | `true` keeps the post out of the published site; `pnpm dev` still shows it.                        |

The build checks every post against this list (`site/src/content.config.ts`):
a missing field, a bad date or a handle with an `@` fails it.

Link to other files in the repository with relative paths, as anywhere else
in it (`../../spec/README.md`): the site turns links to pages it publishes
into links to those pages, and everything else into links to the file on
GitHub.

The newest post also shows over the landing page's tagline for 60 days.

## Previewing

```sh
cd site
pnpm install
pnpm dev   # http://localhost:4321/openinference/news/
```

/** The news as an RSS feed, at `news/rss.xml`. */
import rss from "@astrojs/rss";
import type { APIContext } from "astro";

import { getNews, newsIndex, newsUrl } from "../../lib/news";
import { pages, site } from "../../lib/site";

export async function GET(context: APIContext) {
  const posts = await getNews();
  return rss({
    title: `${site.name} news`,
    description: pages.news.description,
    // The channel links to the timeline; the items' links carry the base, so
    // they resolve against the origin.
    site: new URL(newsIndex, context.site),
    items: posts.map((post) => ({
      title: post.data.title,
      description: post.data.description,
      pubDate: post.data.pubDate,
      link: newsUrl(post),
      categories: post.data.tags,
    })),
  });
}

/**
 * The social cards, one PNG per page, at `og/<page>.png`. See
 * src/lib/og/cards.ts and DESIGN.md ("Social cards").
 */
import type { APIRoute, GetStaticPaths } from "astro";

import { getCards, type Card } from "../../lib/og/cards";
import { renderCard } from "../../lib/og/render";

export const getStaticPaths = (async () =>
  (await getCards()).map((card) => ({ params: { card: card.id }, props: { card } }))) satisfies GetStaticPaths;

export const GET: APIRoute = async ({ props }) =>
  new Response(new Uint8Array(await renderCard((props as { card: Card }).card)), {
    headers: { "Content-Type": "image/png" },
  });

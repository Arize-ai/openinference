/**
 * The files of a published registry version, served as written:
 * `/schemas/<version>/manifest.yaml` and `/schemas/<version>/registry.yaml`.
 */
import type { APIContext } from "astro";

import { getSchemaVersions, SCHEMA_FILES, type SchemaFile } from "../../../lib/schemas";

export function getStaticPaths() {
  return getSchemaVersions().flatMap((schema) =>
    SCHEMA_FILES.map((file) => ({
      params: { version: schema.version, file },
      props: { content: schema.files[file] },
    })),
  );
}

export function GET({ props }: APIContext<{ content: string }, { version: string; file: SchemaFile }>) {
  return new Response(props.content, {
    headers: { "Content-Type": "application/yaml; charset=utf-8" },
  });
}

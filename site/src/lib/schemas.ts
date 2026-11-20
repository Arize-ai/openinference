/**
 * The published semantic convention registry: for each version, the
 * publication manifest and the resolved registry that the site serves under
 * `/schemas/<version>/`, which is where the registry's schema URL points.
 *
 * The files are written to src/generated/schemas/ by
 * `make -C semantic-conventions site-schemas` (the version in the repository)
 * and, in the deploy workflow, downloaded from each release. They are not
 * committed. Without them the site still builds; the schemas page says how to
 * generate them.
 */

/** The two files of a published registry version. */
export const SCHEMA_FILES = ["manifest.yaml", "registry.yaml"] as const;
export type SchemaFile = (typeof SCHEMA_FILES)[number];

export interface SchemaVersion {
  /** e.g. `0.1.0` */
  version: string;
  /** The `schema_url` in the manifest. */
  schemaUrl: string;
  /** The registry's own stability, e.g. `development`. */
  stability: string;
  /** Number of distinct attributes the resolved registry defines or references. */
  attributeCount: number;
  /**
   * False when the files were built from the model in the repository because
   * the version has no release yet. Such a version can still change, and its
   * tag does not exist.
   */
  released: boolean;
  files: Record<SchemaFile, string>;
}

const sources = import.meta.glob<string>("../generated/schemas/*/*.yaml", {
  query: "?raw",
  import: "default",
  eager: true,
});

/** Written next to the files by `make site-schemas`; a release has none. */
const unreleased = new Set(
  Object.keys(import.meta.glob("../generated/schemas/*/unreleased", { query: "?raw" })).map(
    (path) => path.split("/").slice(-2)[0],
  ),
);

/** The value of a top-level `key: value` line of a YAML document. */
const field = (yaml: string, key: string) =>
  yaml.match(new RegExp(`^${key}:\\s*(.+)$`, "m"))?.[1].trim() ?? "";

/** Compares `1.10.0` after `1.9.0`. */
const byVersion = (a: string, b: string) => a.localeCompare(b, undefined, { numeric: true });

/** Every published version, newest first. */
export function getSchemaVersions(): SchemaVersion[] {
  const byName = new Map<string, Partial<Record<SchemaFile, string>>>();
  for (const [path, content] of Object.entries(sources)) {
    const [version, file] = path.split("/").slice(-2);
    byName.set(version, { ...byName.get(version), [file as SchemaFile]: content });
  }
  return [...byName.entries()]
    .filter((entry): entry is [string, Record<SchemaFile, string>] =>
      SCHEMA_FILES.every((file) => entry[1][file] !== undefined),
    )
    .map(([version, files]) => ({
      version,
      schemaUrl: field(files["manifest.yaml"], "schema_url"),
      stability: field(files["manifest.yaml"], "stability"),
      released: !unreleased.has(version),
      // The catalog lists an attribute once per description a span gives it.
      attributeCount: new Set(files["registry.yaml"].match(/^- key: .+$/gm) ?? []).size,
      files,
    }))
    .sort((a, b) => byVersion(b.version, a.version));
}

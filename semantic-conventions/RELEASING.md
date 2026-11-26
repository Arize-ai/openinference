# Releasing the Semantic Conventions Registry

The registry is versioned independently of the language packages, with the same
[release-please](https://github.com/googleapis/release-please) flow as the rest of the repository.

## Version

The version is the last path segment of `schema_url` in [`model/manifest.yaml`](model/manifest.yaml):

```yaml
schema_url: https://openinf.dev/schemas/0.0.1 # x-release-please-version
```

release-please rewrites that line, [`version.txt`](version.txt), and `CHANGELOG.md` in its release pull request. Do not
edit them by hand. `make print-version` prints the current version.

Between releases, `main` still carries the schema URL of the latest release while containing unreleased changes.
Consume a tag, not `main`.

The schema URL is the registry's identifier, and the docs site serves the registry there:
`https://openinf.dev/schemas/X.Y.Z/` has a page for the version, with `manifest.yaml` and `registry.yaml` next to it.
The deploy workflow serves a released version from its release assets, so it never changes once published; a version
that has no release yet is built from the model on `main`. The release workflow starts a deploy once it has
attached the assets, so a new version switches to them without waiting for the next push to `main`.

## What moves the version

Conventional commits that touch `semantic-conventions/` (the repository bumps the patch version for `feat` while
below 1.0):

| Change | Commit |
| --- | --- |
| New attribute, well-known value, or span | `feat(semantic-conventions): ...` |
| Correction to a brief, note, type, or requirement level | `fix(semantic-conventions): ...` |
| Deprecation | `feat(semantic-conventions): deprecate ...` |
| Tooling, templates, or policies only | `chore(semantic-conventions): ...` (no release) |

Attributes are deprecated, never removed. CI fails a pull request that removes or incompatibly changes anything in the
model on the base branch.

## Flow

1. Changes merge to `main`. release-please opens or updates the release pull request.
2. Merging the release pull request tags `semantic-conventions-vX.Y.Z` and publishes a GitHub release.
3. The [Semantic Conventions Release](../.github/workflows/semantic-conventions-release.yaml) workflow runs on that
   release. It validates the model at the tag and attaches two assets:
   - `manifest.yaml`: the publication manifest (schema URL, dependencies, and where the resolved registry is).
   - `registry.yaml`: the resolved registry, with every reference and dependency inlined.

If the workflow did not run or failed, rerun it from the Actions tab with the tag as input.

Other registries depend on a release by its tag, not by these assets; see
[Using the registry](README.md#using-the-registry) for why.

The workflow's work is one target, `make release-assets TAG=semantic-conventions-vX.Y.Z`: it checks the tag
against the schema URL version, validates the model, and builds the assets into `.build/package/`. CI runs the same
target on every change with the tag the next release would have, so the only step a release adds is the upload.

## First release

The registry starts at `0.0.1`. release-please would otherwise make the first release of a package `1.0.0`, so
`release-please-config.json` sets `"initial-version": "0.0.1"` for this package. The setting only applies while the
package has no release, so nothing has to be removed afterwards. `model/manifest.yaml` already carries that version,
so the first release changes only `CHANGELOG.md` and `version.txt`.

A `release-please release-pr --dry-run` against the branch that introduced the registry planned the release and
listed `CHANGELOG.md`, `version.txt`, and `model/manifest.yaml` as the files it updates, and release-please's updater
rewrites the `schema_url` line when the version changes. The first real release is the confirmation.

After it, pin `BASELINE_REGISTRY` in local runs to the released tag when you want to compare against it (see the
[`Makefile`](Makefile)).

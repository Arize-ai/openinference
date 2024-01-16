## JavaScript Development

The development guide for the JavaScript packages in this repo.

### Setup

This project is built using the following tools:

- [pnpm](https://pnpm.io/) for managing packages across the repo. Note, this project uses pnpm workspaces, so you must use pnpm to install packages at the root of the repo.
- [TypeScript](https://www.typescriptlang.org/) for type checking and transpiling.

## Changesets

The changes to the packages managed by this repo are tracked via [changesets](https://pnpm.io/using-changesets). Changesets are similar to semantic commits in that they describe the changes made to the codebase. However, changesets track changes to all the packages by committing `changesets` to the `.changeset` directory. If you make a change to a package, you should create a changeset for it via:

```shell
pnpm changeset
```

A changeset is an intent to release a set of packages at particular [semver bump types](https://semver.org/) with a summary of the changes made.

For a detailed explanation of changesets, consult [this documentation])(https://github.com/changesets/changesets/blob/main/docs/detailed-explanation.md)

## Publishing

```shell
npx changeset # create a changeset
pnpm -r publish # publish to npm
```

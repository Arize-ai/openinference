## JavaScript Development

The development guide for the JavaScript packages in this repo.

This project and its packages are built using the following tools:

- [pnpm](https://pnpm.io/) for managing packages across the repo. Note, this project uses pnpm workspaces, so you must use pnpm to install packages at the root of the repo.
- [TypeScript](https://www.typescriptlang.org/) for type checking and transpiling.
- [Jest](https://jestjs.io/) for unit testing.
- [Eslint](https://eslint.org/) for linting and best practices.
- [Prettier](https://prettier.io/) for code formatting.

### Setup

To get started, you will first need to install [Node.js](https://nodejs.org/en/). This project uses Node.js v20. We recommend using [nvm](https://github.com/nvm-sh/nvm?tab=readme-ov-file#installing-and-updating) to keep track of your Node.js versions.

Once NVM is installed, install Node.js v20 via:

```shell
nvm install 20
nvm use 20
```

Next, you will need to install [pnpm](https://pnpm.io/installation). pnpm is a package manager that is similar to npm, but is much faster and is optimized for workspaces with multiple packages. Once PNPM is installed, you can install the packages in this repo via:

The pnpm version used in this repo is managed in the package.json file. This will allow you to run all commands below with the proper version. In order to take advantage of that make sure your global pnpm version is `>=9.7.0`.

```shell
pnpm install --frozen-lockfile -r
```

This will install all the packages in the repo and their dependencies.

After the dependencies are installed, you can build the packages via the following two commands:

```shell
pnpm run -r prebuild
pnpm run -r build
```

Note that there is a prebuild script that is run before the build script. This script will run the `prebuild` script in each package. This generates all the files needed for the build script to run (e.g. adds instrumentation version numbers and symlinks cross-package dependencies). You will have to run these scripts whenever you make changes to packages in the repo so that the cross-package dependencies are updated.

### Testing

To run the tests for all the packages in the repo, run:

```shell
pnpm run -r test
```

> [!NOTE]
> The tests in this repo use `jest` but it's auto-mocking feature can cause issues since instrumentation relies on it running first before the package is imported in user-code. For the tests you may have to manually set the instrumented module manually (e.x.`instrumentation._modules[0].moduleExports = module`)

## Changesets

The changes to the packages managed by this repo are tracked via [changesets](https://pnpm.io/using-changesets). Changesets are similar to semantic commits in that they describe the changes made to the codebase. However, changesets track changes to all the packages by committing `changesets` to the `.changeset` directory. If you make a change to a package, you should create a changeset for it via:

```shell
pnpm changeset
```

A changeset is an intent to release a set of packages at particular [semver bump types](https://semver.org/) with a summary of the changes made.

For a detailed explanation of changesets, consult [this documentation])(https://github.com/changesets/changesets/blob/main/docs/detailed-explanation.md)

## Publishing

In most cases, changes to the packages in this repo will be published automatically via Github Actions and the changesets workflow. However, if you need to publish manually, you can do so via:

```shell
pnpm changeset # create a changeset
pnpm changeset version # bump the version of the packages
pnpm -r prebuild # generate the files needed for the build
pnpm -r build # build the packages
pnpm -r publish # publish to npm
```

Note that the packages are published to the `@arizeai` npm organization. You will need to be added to this organization to publish packages.

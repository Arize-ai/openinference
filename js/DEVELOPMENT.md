## JavaScript Development

- [Setup](#setup)
- [Testing](#testing)
- [Creating an Instrumentor](#creating-an-instrumentor)
  - [Minimum Feature Set](#minimum-feature-set)
    - [Suppress Tracing](#suppress-tracing)
    - [Context Attribute Propagation](#context-attribute-propagation)
    - [Trace Configuration](#trace-configuration)
    - [Testing](#testing-1)
- [Changesets](#changesets)
- [Publishing](#publishing)

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

## Creating an Instrumentor

To keep our instrumentors up to date and in sync there is a set of features that each instrumentor must contain. Most of these features are implemented in our [openinference-core package](./packages/openinference-core/) and will be handled by the [OITracer](./packages/openinference-core/src/trace/trace-config/OITracer.ts) and underlying [OISpan](./packages/openinference-core/src/trace/trace-config/OISpan.ts) so it's important to use the OITracer in your instrumentations.

To use the our OITracer in your instrumentor's make sure to add it as a private property on your instrumentor class. Then be sure to use it anytime you need to create a span (or do anything else with the `tracer`). Example:

```typescript
export class MyInstrumentation extends InstrumentationBase<
  typeof moduleToInstrument
> {
  private oiTracer: OITracer;
  constructor({
    instrumentationConfig,
    traceConfig,
  }: {
    instrumentationConfig?: InstrumentationConfig;
    traceConfig?: TraceConfigOptions;
  } = {}) {
    super(
      "@arizeai/openinference-instrumentation-example-module",
      VERSION,
      Object.assign({}, instrumentationConfig),
    );
    this.oiTracer = new OITracer({ tracer: this.tracer, traceConfig });
  }
  // ...
  // Use this.oiTracer not this.tracer in your code to create spans
  const span = this.oiTracer.startSpan("my-span");
  span.end()
}
```

### Minimum Feature Set

Each instrumentation must contain the following features:

#### Suppress Tracing

Every instrumentor must allow tracing to be suppressed or disabled.

In JS/TS tracing suppression is controlled by a context attribute see [suppress-tracing.ts](https://github.com/open-telemetry/opentelemetry-js/blob/55a1fc88d84b22c08e6a19eff71875e15377b781/packages/opentelemetry-core/src/trace/suppress-tracing.ts#L23) from opentelemetry-js. This context key must be respected in each instrumentation. To check for this key and block tracing see our [openai-instrumentation](./packages/openinference-instrumentation-openai/src/instrumentation.ts#69).

Every instrumentation must also be able to be disabled. The `disable` method is inherited from the `InstrumentationBase` class and does not have to be implemented. To ensure that your instrumentation can be properly disabled you just need to properly implement the `unpatch` method on your instrumentation.

#### Context Attribute Propagation

There are a number of situations in which a user may want to attach a specific attribute to every span that gets created within a particular block or scope. For example a user may want to ensure that every span created has a user or session ID attached to it. We achieve this by allowing users to set attributes on [context](https://opentelemetry.io/docs/specs/otel/context/). Our instrumentors must respect these attributes and correctly propagate them to each span.

This fetching and propagation is controlled by our [OITracer](./packages/openinference-core/src/trace/trace-config/OITracer.ts#117) and [context attributes](./packages/openinference-core/src/trace/contextAttributes.ts) from our core package. See the example above to properly use the OITracer in your instrumentor to ensure context attributes are repected.

#### Trace Configuration

In some situations, users may want to control what data gets added to a span. We allow them to do this via [trace config](./packages/openinference-core/src/trace/trace-config/). Our trace config allows users to mask certain fields on a span to prevent sensitive information from leaving their system. 

As with context attribute propagation, this is controlled by our OITracer and [OISpan](./packages/openinference-core/src/trace/trace-config/OISpan.ts#21). See the example above to properly use the OITracer in your instrumentor to ensure the trace config is respected.

#### Testing

In addition to any additional testing you do for your instrumentor, it's important to write tests specifically for the features above. This ensures that all of our instrumentors have the same core set of functionality and can help to catch up-stream bugs in our core package.

## Changesets

The changes to the packages managed by this repo are tracked via [changesets](https://pnpm.io/using-changesets). Changesets are similar to semantic commits in that they describe the changes made to the codebase. However, changesets track changes to all the packages by committing `changesets` to the `.changeset` directory. If you make a change to a package, you should create a changeset for it via:

```shell
pnpm changeset
```

and commit it in your pr.

A changeset is an intent to release a set of packages at particular [semver bump types](https://semver.org/) with a summary of the changes made.

Once your pr is merged, Github Actions will create a release PR like [this](https://github.com/Arize-ai/openinference/pull/994). Once the release pr is merged, new versions of any changed packages will be published to npm.

For a detailed explanation of changesets, consult [this documentation](https://github.com/changesets/changesets/blob/main/docs/detailed-explanation.md)

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

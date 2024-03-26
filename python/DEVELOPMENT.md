# Development

This project uses [ruff](https://github.com/astral-sh/ruff) for formatting and linting, 
[mypy](https://github.com/python/mypy) for type checking, and [tox](https://github.com/tox-dev/tox) for 
automation. To start, install `tox`.

```console
$ pip install tox
```

## Example commands

- `tox run-parallel` to run all CI checks in parallel, including unit tests for all packages
under multiple Python versions.
  - This runs all the environments defined under `envlist=` in `tox.ini`.
- `tox run -e ruff-openai` to run formatting and linting the openai instrumentation package.
- `tox run -e mypy-openai` to run type checking on the openai instrumentation package.
- `tox run -e test-openai` to run type testing on the openai instrumentation package.
- `tox run -e ruff-mypy-test-openai` to run all three at the same time on the openai
instrumentation package.

### Notes
- When you run these commands, `tox` will print out all the steps it is taking in the command line.
- The `openai` substring in the commands above can be replaced by other package names defined in `tox.ini`, e.g.
`semconv`.
- `-e` specifies the environment string(s), which is a hyphen-delimited list of "factors" defined in `tox.ini`.
  - Multiple environment strings can be specified by separating them with commas, e.g. `-e ruff-openai,ruff-semconv`.
- Without `-e`, `tox` runs all the environments defined under `envlist=` in `tox.ini`.

## Introduction to `tox`

The first thing to understand about `tox` is that an environment string such as `ruff-openai` is a
hyphen-delimited _list_ of "factors", i.e. `ruff` and `openai`. In `tox.ini`, these factors are defined separately
and each instructs `tox` to carry out different actions. When we call `tox run -e ruff-openai`, `tox` will first 
create a virtual environment called "ruff-openai", and then inside that virtual environment carry out the 
actions defined under `ruff` and `openai`. In short, the `-` (hyphen) signifies a conjunction of predefined 
actions to performed in a virtual environment created under a name that is a concatenation of the factor strings.
That is why you don't see "ruff-mypy-test-openai" defined in `tox.ini` -- it is just a conjunction of the four 
factors: `ruff`, `mypy`, `test`, and `openai`.


# Publishing

## Conda-Forge

In addition to distributing our software via `pypi`, we also publish to Conda.

### Creating a New Conda Feedstock

Once a new package is published to PyPI, a few manual steps are needed to create a corresponding Conda feedstock, which is a repository containing all the necessary files and configurations to build the Conda package.

1. Fork and clone `conda-forge`'s [`staged-recipes`](https://github.com/conda-forge/staged-recipes) repo.
1. Change directories into the `recipes` folder.
1. Install `grayskull`, a tool to generate your conda recipe, with `pip install grayskull`. Run `grayskull pypi <name-of-your-package-on-pypi>`.
1. Create a PR targeting upstream.
1. Modify the contents of `meta.yaml` as needed. In particular:
  - Make sure your GitHub username is correct under `extra.recipe-maintainers`.
  - Make sure the imports under `test.imports` are correct.
  - Set `about.home` to the appropriate URL.
1. Once CI passes, alert the Conda Forge admins that the PR is ready for review with a comment such as: `@conda-forge/help-python, ready for review!`. Once the admin has reviewed and merged your PR, your feedstock repo will be created under the `conda-forge` organization on GitHub by appending `-feedstock` to your package name, e.g., `https://github.com/conda-forge/arize-phoenix-feedstock`.
1. In the feedstock repo, add a new issue entitled `@conda-forge-admin, please add bot automerge` to allow the bot to merge PRs automatically.
1. For each member of the OSS team, create an issue entitled `@conda-forge-admin, please add user @<user-name>`. The bot will automatically create PRs to add maintainers, which you must merge manually.

### Updating the Conda Feedstock

After the Conda feedstock has been created and the Conda package has been published, Conda will handle most subsequent releases automatically. In some cases (e.g., when a new dependency is added to the package), it's necessary to manually edit the Conda feedstock. This can be accomplished by committing directly to `main` in the feedstock repo.

### Notes

Conda Forge has a delay of several hours when detecting new releases on PyPI. If multiple PyPI releases for the same package are made in a single day, Conda Forge usually only catches the last one, so we sometimes wind up with gaps in our release train for Conda.

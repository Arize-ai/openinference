# Development

This project uses [ruff](https://github.com/astral-sh/ruff) for formatting and linting, 
[mypy](https://github.com/python/mypy) for type checking, and [tox](https://github.com/tox-dev/tox) for 
automation. To install these tools, run:

```console
$ pip install -r dev-requirements.txt
```

## Example commands

- `tox run-parallel` to run all CI checks in parallel, including unit tests for all packages
under multiple Python versions.
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
Multiple environment strings can be specified by separating them with commas, e.g. `-e ruff-openai,ruff-semconv`.
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

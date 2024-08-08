# OpenInference Development Guide <!-- omit in toc -->

- [Development](#development)
  - [Testing](#testing)
    - [Introduction to `tox`](#introduction-to-tox)
    - [`tox` Example Commands](#tox-example-commands)
    - [`tox` Notes](#tox-notes)
  - [Creating an instrumentor](#creating-an-instrumentor)
    - [Minimal Feature Set](#minimal-feature-set)
      - [Suppress Tracing](#suppress-tracing)
      - [Customizing Spans](#customizing-spans)
      - [Tracing Configuration](#tracing-configuration)
    - [Setup Testing](#setup-testing)
      - [Setup Dependencies](#setup-dependencies)
      - [Setup `tox`](#setup-tox)
- [Publishing](#publishing)
  - [Conda-Forge](#conda-forge)
    - [Creating a New Conda Feedstock](#creating-a-new-conda-feedstock)
    - [Updating the Conda Feedstock](#updating-the-conda-feedstock)
    - [Conda Notes](#conda-notes)


## Development

This project uses [ruff](https://github.com/astral-sh/ruff) for formatting and linting, [mypy](https://github.com/python/mypy) for type checking, and [tox](https://github.com/tox-dev/tox) for automation. To start, install `tox`:
```console
pip install tox
```

### Testing

#### Introduction to `tox`

The first thing to understand about `tox` is that an environment string such as `ruff-openai` is a hyphen-delimited _list_ of "factors", i.e. `ruff` and `openai`. In `tox.ini`, these factors are defined separately and each instructs `tox` to carry out different actions. When we call `tox run -e ruff-openai`, `tox` will first create a virtual environment called "ruff-openai", and then inside that virtual environment carry out the actions defined under `ruff` and `openai`. In short, the `-` (hyphen) signifies a conjunction of predefined actions to performed in a virtual environment created under a name that is a concatenation of the factor strings. 
That is why you don't see "ruff-mypy-test-openai" defined in `tox.ini` -- it is just a conjunction of the four
factors: `ruff`, `mypy`, `test`, and `openai`.

#### `tox` Example Commands

- `tox run-parallel` to run all CI checks in parallel, including unit tests for all packages
under multiple Python versions.
  - This runs all the environments defined under `envlist=` in `tox.ini`.
- `tox run -e ruff-openai` to run formatting and linting the openai instrumentation package.
- `tox run -e mypy-openai` to run type checking on the openai instrumentation package.
- `tox run -e test-openai` to run type testing on the openai instrumentation package.
- `tox run -e ruff-mypy-test-openai` to run all three at the same time on the openai
instrumentation package.

#### `tox` Notes

- When you run these commands, `tox` will print out all the steps it is taking in the command line.
- The `openai` substring in the commands above can be replaced by other package names defined in `tox.ini`, e.g.
`semconv`.
- `-e` specifies the environment string(s), which is a hyphen-delimited list of "factors" defined in `tox.ini`.
  - Multiple environment strings can be specified by separating them with commas, e.g. `-e ruff-openai,ruff-semconv`.
- Without `-e`, `tox` runs all the environments defined under `envlist=` in `tox.ini`.

### Creating an instrumentor

#### Minimal Feature Set

Each instrumentor created must contain the following features:

##### Suppress Tracing

Every instrumentor must allow tracing to be paused temporarily or disabled permanently:

To **disable tracing temporarily**, we use the Python context manager `suprress_tracing` imported from the core `openinference-instrumentation` package:

```python
from phoenix.trace import suppress_tracing

with suppress_tracing():
    # Code running inside this block doesn't generate traces.
    # For example, running LLM evals here won't generate additional traces.
    ...
# Tracing will resume outside the block.
...
```

The instrumentor you are developing must be reactive to this and in fact stop tracing inside the `with` block. What the context manager `suppress_tracing` does is to attach `_SUPPRESS_INSTRUMENTATION_KEY=True` to the OTEL `Context` as we enter the `with` block. Once we exit the block, we attach `_SUPPRESS_INSTRUMENTATION_KEY=False` to the OTEL `Context`. 

Hence, in order for your instrumentor to be reactive to this, you need to check for the value of `_SUPPRESS_INSTRUMENTATION_KEY` from the OTEL `Context`. Here is an example:

```python
from opentelemetry import context as context_api
if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
    # skip tracing
```

To **disable tracing permanently**, we call the `.uninstrument()` method that must be present on every instrumentor. Below is the examples for LangChain, LlamaIndex and OpenAI, respectively.

```python
LangChainInstrumentor().uninstrument()
LlamaIndexInstrumentor().uninstrument()
OpenAIInstrumentor().uninstrument()
```

For the instrumentor to have this option, you need to implement the private method `_uninstrument()`, where you will include the logic necessary to stop tracing indefinitely. In general, this method cancels the monkey patching that takes place in the `_instrument()` method and returns the logic of the package that you are instrumenting (OpenAI, Llama-Index, MistralAI, ...) to its original state. Check our any of our current instrumentors for specific examples.

```python
def _uninstrument(self, **kwargs: Any) -> None:
    # Return package logic to original state
```

##### Customizing Spans

Our core `openinference-instrumentation` package offers other Python context managers that serve to customize every span created when a call is placed inside the `with` block. At the time of this writing, these context managers are (can be found [here](https://github.com/Arize-ai/openinference/blob/main/python/openinference-instrumentation/src/openinference/instrumentation/context_attributes.py)):

- `using_session`: to specify a session ID to track and group a multi-turn conversation with a user
- `using_user`: to specify a user ID to track different conversations with a given user
- `using_metadata`: to add custom metadata, that can provide extra information that supports a wide range of operational needs
- `using_tag`: to add tags, to help filter on specific keywords
- `using_prompt_template`: to reflect the prompt template used, with its version and variables. This is useful for prompt template management
- `using_attributes`: it helps handling multiple of the previous options at once in a concise manner

In short, they attach extra key value pairs to the OTEL `Context` (because of this we often can refer to them as "context attributes") as we enter the `with` block and detach them as we exit. Please look at the implementation for more details.

Hence, for any instrumentor to capture these extra attributes, we need to read from the OTEL `Context`. Our core `openinference-instrumentation` package offers a helper function to do this with ease: `get_attributes_from_context`. Thus, it suffices with doing the following when you create a span:

```python
from opentelemetry.trace import Tracer
from openinference.instrumentation import get_attributes_from_context

tracer: Tracer = # ...
span = tracer.start_span(
    name=# span name...
    start_time=# start time...
    attributes=dict(get_attributes_from_context()),
    #Other possible options...
)
```

or set the attributes after you've created the span:

```python
from opentelemetry.trace import Tracer
from openinference.instrumentation import get_attributes_from_context

span_name:str = #...
with tracer.start_as_current_span(name=span_name) as span:
    span.set_attributes(dict(get_attributes_from_context()))
```

Check our current instrumentors for more examples and details.

##### Tracing Configuration

#### Setup Testing

##### Setup Dependencies

##### Setup `tox`

## Publishing

### Conda-Forge

In addition to distributing our software via `pypi`, we also publish to Conda.

#### Creating a New Conda Feedstock

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

#### Updating the Conda Feedstock

After the Conda feedstock has been created and the Conda package has been published, Conda will handle most subsequent releases automatically. In some cases (e.g., when a new dependency is added to the package), it's necessary to manually edit the Conda feedstock. This can be accomplished by committing directly to `main` in the feedstock repo.

#### Conda Notes

Conda Forge has a delay of several hours when detecting new releases on PyPI. If multiple PyPI releases for the same package are made in a single day, Conda Forge usually only catches the last one, so we sometimes wind up with gaps in our release train for Conda.

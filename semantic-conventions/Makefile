# OpenInference Semantic Conventions - Makefile
#
# Weaver warns about every file of the OpenTelemetry dependency on each run, so
# targets send its stderr to $(BUILD)/*.log and print the log only on failure.
#
# Requires either a local weaver >= $(WEAVER_VERSION) or docker/podman. The
# weaver version is pinned in versions.env and, when no suitable local weaver is
# found, weaver runs from the otel/weaver container image.
#
# Run from this directory:  make -C semantic-conventions <target>

SHELL := /bin/bash
.DEFAULT_GOAL := help

# Shared external version pins. Override on the command line when needed, e.g.
# `make check WEAVER_VERSION=v0.29.0`.
include versions.env

LOCAL_RAW_VERSION := $(shell weaver --version 2>/dev/null)

ifeq ($(LOCAL_RAW_VERSION),)
    USE_CONTAINER := 1
else
    LOCAL_VERSION := $(shell echo "$(LOCAL_RAW_VERSION)" | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -n1)
    PINNED_VERSION := $(shell echo "$(WEAVER_VERSION)" | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -n1)
    IS_LOWER := $(shell awk -v v1="$(LOCAL_VERSION)" -v v2="$(PINNED_VERSION)" 'BEGIN { split(v1, a, "."); split(v2, b, "."); for (i = 1; i <= 3; i++) { if (a[i]+0 < b[i]+0) { print "yes"; exit }; if (a[i]+0 > b[i]+0) { print "no"; exit } }; print "no" }')
    ifeq ($(IS_LOWER),yes)
        $(warning local weaver $(LOCAL_VERSION) is older than the pinned $(PINNED_VERSION); using the container image.)
        USE_CONTAINER := 1
    else
        USE_CONTAINER := 0
    endif
endif

# The repository root is mounted, not just this directory, because generate-spec
# rewrites tables in ../spec.
ifeq ($(USE_CONTAINER),1)
CONTAINER_RUNTIME ?= $(shell command -v docker 2>/dev/null || command -v podman 2>/dev/null)
WEAVER := $(CONTAINER_RUNTIME) run --rm \
	-u $(shell id -u):$(shell id -g) \
	-v "$(abspath $(CURDIR)/..):/workspace" \
	-w /workspace/semantic-conventions \
	-e HOME=/tmp \
	docker.io/otel/weaver:$(WEAVER_VERSION)
else
WEAVER := weaver
endif

PYTHON ?= python3
MODEL := ./model
BUILD := .build
DOCS_OUTPUT := docs/registry
# Prose specification whose tables are generated from the model.
SPEC_DIR := ../spec
# Language packages whose constants are generated from the model, as
# <template directory under templates/registry>:<output directory>.
CODE_TARGETS := \
	python:../python/openinference-semantic-conventions/src/openinference/semconv/trace \
	go:../go/openinference-semantic-conventions \
	typescript:../js/packages/openinference-semantic-conventions/src/trace \
	java:../java/openinference-semantic-conventions/src/main/java/com/arize/semconv/trace
# The files those templates write, for the staleness check.
CODE_OUTPUTS := \
	../python/openinference-semantic-conventions/src/openinference/semconv/trace/__init__.py \
	../go/openinference-semantic-conventions/attributes.go \
	../go/openinference-semantic-conventions/enums.go \
	../js/packages/openinference-semantic-conventions/src/trace/SemanticConventions.ts \
	../java/openinference-semantic-conventions/src/main/java/com/arize/semconv/trace/SemanticConventions.java
PACKAGE_OUTPUT := $(BUILD)/package

# Registry to compare against for the backwards-compatibility policies. Empty
# skips the comparison. CI passes the model on the pull request's base branch;
# locally, compare against a release:
#   make check BASELINE_REGISTRY='https://github.com/Arize-ai/openinference.git@semantic-conventions-v0.1.0[semantic-conventions/model]'
BASELINE_REGISTRY ?=
BASELINE_FLAG := $(if $(BASELINE_REGISTRY),--baseline-registry '$(BASELINE_REGISTRY)',)

# X.Y.Z from the schema_url in model/manifest.yaml, the single source of truth
# for the registry version. release-please bumps it.
VERSION := $(shell awk '/^schema_url:/ { n = split($$2, parts, "/"); print parts[n]; exit }' model/manifest.yaml)
GITHUB_REPOSITORY ?= Arize-ai/openinference
RELEASE_TAG := semantic-conventions-v$(VERSION)
RELEASE_DOWNLOAD_URL := https://github.com/$(GITHUB_REPOSITORY)/releases/download/$(RELEASE_TAG)

.PHONY: help check test generate generate-docs generate-spec generate-code check-generated check-parity check-all package print-version clean

help: ## Show this help
	@awk 'BEGIN { FS = ":.*## " } /^[a-zA-Z_-]+:.*## / { printf "  %-16s %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

check: ## Validate the model and run the shared OpenTelemetry + OpenInference policies
	@mkdir -p $(BUILD)
	@code=0; $(WEAVER) registry check \
		-r $(MODEL) \
		--v2 \
		--quiet \
		--diagnostic-format json \
		--diagnostic-stdout \
		--policy '$(POLICY_REPO_URL)@$(POLICY_REPO_REF)[policies/check]' \
		--policy policies/check/openinference \
		$(BASELINE_FLAG) \
		> $(BUILD)/diagnostics.json || code=$$?; \
	$(PYTHON) scripts/check_policies.py $(BUILD)/diagnostics.json policies/exceptions.json --weaver-exit-code $$code

test: ## Check that sample wire-format keys resolve against the registry
	@mkdir -p $(BUILD)
	@for kind in known unknown; do \
		$(WEAVER) registry live-check \
			-r $(MODEL) \
			--quiet \
			--input-source tests/live-check/$$kind.json \
			--input-format json \
			--format json \
			--no-stats \
			--fail-on=none \
			> $(BUILD)/live-$$kind.json 2> $(BUILD)/live-$$kind.log \
			|| { cat $(BUILD)/live-$$kind.log; exit 1; }; \
		$(PYTHON) scripts/check_live.py $$kind $(BUILD)/live-$$kind.json || exit 1; \
	done

generate: generate-docs generate-spec generate-code ## Regenerate everything derived from the model

generate-docs: ## Regenerate the registry reference under docs/registry/
	@mkdir -p $(BUILD)
	rm -rf $(DOCS_OUTPUT)
	@$(WEAVER) registry generate \
		-r $(MODEL) \
		--v2 \
		--quiet \
		-t ./templates/registry \
		markdown \
		./$(DOCS_OUTPUT) \
		2> $(BUILD)/generate-docs.log || { cat $(BUILD)/generate-docs.log; exit 1; }
	@echo "Generated $(DOCS_OUTPUT)"

generate-spec: ## Regenerate the tables in ../spec between `<!-- weaver ... -->` and `<!-- endweaver -->`
	@mkdir -p $(BUILD)
	@$(WEAVER) registry update-markdown \
		-r $(MODEL) \
		--v2 \
		--quiet \
		-t ./templates \
		--target markdown \
		$(SPEC_DIR) \
		2> $(BUILD)/generate-spec.log || { cat $(BUILD)/generate-spec.log; exit 1; }
	@echo "Generated tables in $(SPEC_DIR)"

generate-code: ## Regenerate the constants in the Python, Go, TypeScript and Java packages
	@mkdir -p $(BUILD)
	@for target in $(CODE_TARGETS); do \
		language=$$(echo $$target | cut -d: -f1); output=$$(echo $$target | cut -d: -f2); \
		$(WEAVER) registry generate \
			-r $(MODEL) \
			--v2 \
			--quiet \
			-t ./templates/registry \
			$$language \
			$$output \
			2> $(BUILD)/generate-code-$$language.log \
			|| { cat $(BUILD)/generate-code-$$language.log; exit 1; }; \
		echo "Generated $$language in $$output"; \
	done

check-generated: generate ## Fail if anything generated from the model is out of date or was edited by hand
	@if ! git diff --exit-code -- $(DOCS_OUTPUT) $(SPEC_DIR) $(CODE_OUTPUTS); then \
		echo "FAIL: generated files are out of date. Run 'make -C semantic-conventions generate' and commit."; exit 1; fi
	@untracked=$$(git ls-files --others --exclude-standard -- $(DOCS_OUTPUT) $(CODE_OUTPUTS)); \
	if [ -n "$$untracked" ]; then echo "FAIL: untracked generated files:"; echo "$$untracked"; exit 1; fi
	@echo "OK: everything generated from the model is up to date."

check-parity: ## Fail if the model has drifted from spec/ or the language packages
	uv run --no-project --with pyyaml==6.0.3 $(PYTHON) scripts/check_parity.py

check-all: check test check-generated check-parity ## Run every check CI runs

package: ## Build the release assets: publication manifest + resolved registry
	rm -rf $(PACKAGE_OUTPUT)
	@mkdir -p $(PACKAGE_OUTPUT)
	@$(WEAVER) registry package \
		-r $(MODEL) \
		--v2 \
		--quiet \
		--resolved-registry-uri '$(RELEASE_DOWNLOAD_URL)/registry.yaml' \
		-o ./$(PACKAGE_OUTPUT)/weaver \
		2> $(BUILD)/package.log || { cat $(BUILD)/package.log; exit 1; }
	mv $(PACKAGE_OUTPUT)/weaver/manifest.yaml $(PACKAGE_OUTPUT)/manifest.yaml
	mv $(PACKAGE_OUTPUT)/weaver/resolved.yaml $(PACKAGE_OUTPUT)/registry.yaml
	rm -rf $(PACKAGE_OUTPUT)/weaver
	@echo "Packaged $(VERSION) -> $(PACKAGE_OUTPUT)"

print-version: ## Print the registry version
	@echo $(VERSION)

clean: ## Remove build output
	rm -rf $(BUILD)

# OpenInference Semantic Conventions - Makefile
#
# Weaver warns about every file of the OpenTelemetry dependency on each run, so
# targets send its stderr to $(BUILD)/*.log and print the log only on failure.
#
# Requires either a local weaver >= $(WEAVER_VERSION) or docker/podman. The
# weaver version is pinned in versions.env and, when no suitable local weaver is
# found, weaver runs from the otel/weaver container image.
#
# The generate-* targets and `test` are hermetic: their only inputs are files in
# this repository and the pinned weaver, and in the container they run with no
# network. See "Hermetic generation" below. `check`, `vendor`, `check-vendor`,
# and `package` resolve the real OpenTelemetry registry and need the network.
#
# Run from this directory:  make -C semantic-conventions <target>

SHELL := /bin/bash
.DEFAULT_GOAL := help

# Shared external version pins. Override on the command line when needed, e.g.
# `make check WEAVER_VERSION=v0.29.0`.
include versions.env

# FORCE_CONTAINER=1 runs weaver from the container image even when a suitable
# local weaver exists. CI uses it for `make test`, so that the hermetic targets
# are proven to work with the network disabled.
FORCE_CONTAINER ?=
LOCAL_RAW_VERSION := $(if $(FORCE_CONTAINER),,$(shell weaver --version 2>/dev/null))

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
CONTAINER_RUN := $(CONTAINER_RUNTIME) run --rm \
	-u $(shell id -u):$(shell id -g) \
	-v "$(abspath $(CURDIR)/..):/workspace" \
	-w /workspace/semantic-conventions \
	-e HOME=/tmp
WEAVER := $(CONTAINER_RUN) docker.io/otel/weaver:$(WEAVER_VERSION)
# No network: a hermetic target that reaches for it fails instead of passing.
WEAVER_OFFLINE := $(CONTAINER_RUN) --network none docker.io/otel/weaver:$(WEAVER_VERSION)
else
WEAVER := weaver
WEAVER_OFFLINE := weaver
endif

PYTHON ?= python3
MODEL := ./model
BUILD := .build

# Hermetic generation. The model depends on the OpenTelemetry registry for the
# attributes it reuses, and resolving that dependency means cloning it. The
# definitions of exactly those attributes are vendored in $(VENDOR_DIR) (see
# scripts/vendor_upstream.py). Generation runs against $(OFFLINE_MODEL): a copy
# of the model whose dependency points at the vendored copy. model/manifest.yaml
# itself keeps the real registry, which is what `check` and other registries
# that depend on this one resolve against; `check-vendor` proves the two agree.
VENDOR_DIR := vendor/opentelemetry
OFFLINE_MODEL := $(BUILD)/model
DOCS_OUTPUT := docs/registry
# Prose specification whose tables are generated from the model.
SPEC_DIR := ../spec
# Language packages whose constants are generated from the model, as
# <template directory under templates/registry>:<output directory>.
CODE_TARGETS := \
	python:../python/openinference-semantic-conventions/src/openinference/semconv \
	go:../go/openinference-semantic-conventions \
	typescript:../js/packages/openinference-semantic-conventions/src \
	java:../java/openinference-semantic-conventions/src/main/java/com/arize/semconv/trace
# The files those templates write, for the staleness check.
CODE_OUTPUTS := \
	../python/openinference-semantic-conventions/src/openinference/semconv/trace/__init__.py \
	../python/openinference-semantic-conventions/src/openinference/semconv/resource/__init__.py \
	../go/openinference-semantic-conventions/attributes.go \
	../go/openinference-semantic-conventions/enums.go \
	../go/openinference-semantic-conventions/indexers.go \
	../js/packages/openinference-semantic-conventions/src/trace/SemanticConventions.ts \
	../js/packages/openinference-semantic-conventions/src/resource/SemanticResourceAttributes.ts \
	../java/openinference-semantic-conventions/src/main/java/com/arize/semconv/trace/SemanticConventions.java \
	../java/openinference-semantic-conventions/src/main/java/com/arize/semconv/trace/SemanticResourceAttributes.java
PACKAGE_OUTPUT := $(BUILD)/package
# Where the docs site reads the registry it serves at the schema URL. Not
# committed; see site/src/lib/schemas.ts.
SITE_SCHEMAS := ../site/src/generated/schemas

# Registry to compare against for the backwards-compatibility policies. Empty
# skips the comparison. CI passes the model on the pull request's base branch;
# locally, compare against a release:
#   make check BASELINE_REGISTRY='https://github.com/Arize-ai/openinference.git@semantic-conventions-v0.0.1[semantic-conventions/model]'
BASELINE_REGISTRY ?=
BASELINE_FLAG := $(if $(BASELINE_REGISTRY),--baseline-registry '$(BASELINE_REGISTRY)',)

# X.Y.Z from the schema_url in model/manifest.yaml, the single source of truth
# for the registry version. release-please bumps it.
VERSION := $(shell awk '/^schema_url:/ { n = split($$2, parts, "/"); print parts[n]; exit }' model/manifest.yaml)
SCHEMA_URL := $(shell awk '/^schema_url:/ { print $$2; exit }' model/manifest.yaml)
GITHUB_REPOSITORY ?= Arize-ai/openinference
RELEASE_TAG := semantic-conventions-v$(VERSION)
RELEASE_DOWNLOAD_URL := https://github.com/$(GITHUB_REPOSITORY)/releases/download/$(RELEASE_TAG)

.PHONY: help check offline-model vendor check-vendor test generate generate-docs generate-spec generate-code check-generated check-parity check-all package release-assets site-schemas print-version clean

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

offline-model: # Copy the model with its dependency pointed at the vendored OpenTelemetry attributes
	@mkdir -p $(OFFLINE_MODEL)
	@cp -R model/. $(OFFLINE_MODEL)/
	@sed 's|registry_path: .*|registry_path: $(VENDOR_DIR)|' model/manifest.yaml > $(OFFLINE_MODEL)/manifest.yaml
	@grep -q 'registry_path: $(VENDOR_DIR)$$' $(OFFLINE_MODEL)/manifest.yaml \
		|| { echo "FAIL: could not point the model's dependency at $(VENDOR_DIR)"; exit 1; }
	@stale=$$(cd $(OFFLINE_MODEL) && find . -type f | while read -r file; do [ -e "$(CURDIR)/model/$$file" ] || echo "$$file"; done); \
	if [ -n "$$stale" ]; then \
		echo "FAIL: $(OFFLINE_MODEL) has files that are no longer in model/:"; echo "$$stale"; \
		echo "Run 'make -C semantic-conventions clean' and try again."; exit 1; fi

vendor: ## Refresh the vendored OpenTelemetry attributes from the pinned upstream version (needs network)
	@uv run --no-project --with pyyaml==6.0.3 $(PYTHON) scripts/vendor_upstream.py --weaver '$(WEAVER)'

check-vendor: ## Fail if the vendored OpenTelemetry attributes differ from upstream (needs network)
	@uv run --no-project --with pyyaml==6.0.3 $(PYTHON) scripts/vendor_upstream.py --weaver '$(WEAVER)' --check

test: offline-model ## Check sample keys against the registry, a dependent registry, and that each OpenInference policy fires
	@mkdir -p $(BUILD)
	@for kind in known unknown unvalidated; do \
		$(WEAVER_OFFLINE) registry live-check \
			-r $(OFFLINE_MODEL) \
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
	@$(WEAVER_OFFLINE) registry check -r tests/consumer --v2 --quiet \
		> $(BUILD)/consumer.log 2>&1 || { cat $(BUILD)/consumer.log; exit 1; }
	@echo "OK: a registry that depends on this one resolves (tests/consumer)."
	@$(WEAVER_OFFLINE) registry check -r tests/policies/violations --v2 --quiet \
		--diagnostic-format json --diagnostic-stdout \
		--policy policies/check/openinference \
		> $(BUILD)/policy-violations.json 2> $(BUILD)/policy-violations.log; \
	$(PYTHON) scripts/check_policy_fixtures.py $(BUILD)/policy-violations.json \
		tests/policies/violations/expected.json policies/check/openinference \
		|| { cat $(BUILD)/policy-violations.log; exit 1; }

generate: generate-docs generate-spec generate-code ## Regenerate everything derived from the model

generate-docs: offline-model ## Regenerate the registry reference under docs/registry/
	@mkdir -p $(BUILD)
	rm -rf $(DOCS_OUTPUT)
	@$(WEAVER_OFFLINE) registry generate \
		-r $(OFFLINE_MODEL) \
		--v2 \
		--quiet \
		-t ./templates/registry \
		markdown \
		./$(DOCS_OUTPUT) \
		2> $(BUILD)/generate-docs.log || { cat $(BUILD)/generate-docs.log; exit 1; }
	@echo "Generated $(DOCS_OUTPUT)"

generate-spec: offline-model ## Regenerate the tables in ../spec between `<!-- weaver ... -->` and `<!-- endweaver -->`
	@mkdir -p $(BUILD)
	@$(WEAVER_OFFLINE) registry update-markdown \
		-r $(OFFLINE_MODEL) \
		--v2 \
		--quiet \
		-t ./templates \
		--target markdown \
		$(SPEC_DIR) \
		2> $(BUILD)/generate-spec.log || { cat $(BUILD)/generate-spec.log; exit 1; }
	@echo "Generated tables in $(SPEC_DIR)"

generate-code: offline-model ## Regenerate the constants in the Python, Go, TypeScript and Java packages
	@mkdir -p $(BUILD)
	@for target in $(CODE_TARGETS); do \
		language=$$(echo $$target | cut -d: -f1); output=$$(echo $$target | cut -d: -f2); \
		$(WEAVER_OFFLINE) registry generate \
			-r $(OFFLINE_MODEL) \
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

check-all: check check-vendor test check-generated check-parity site-schemas ## Run every check CI runs
	@$(MAKE) --no-print-directory release-assets TAG=$(RELEASE_TAG)

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

# Everything the release workflow does before uploading. CI runs the same
# target as a rehearsal on every change, with the tag the next release would
# have, so the release path is exercised before a release exists.
release-assets: ## Validate and build the assets for release TAG (e.g. TAG=semantic-conventions-v0.0.1)
	@if [ "$(TAG)" != "$(RELEASE_TAG)" ]; then \
		echo "FAIL: tag '$(TAG)' does not match model/manifest.yaml schema_url (expected '$(RELEASE_TAG)')."; exit 1; fi
	@$(MAKE) --no-print-directory check
	@$(MAKE) --no-print-directory package
	@for asset in manifest.yaml registry.yaml; do \
		[ -s $(PACKAGE_OUTPUT)/$$asset ] || { echo "FAIL: $(PACKAGE_OUTPUT)/$$asset was not built."; exit 1; }; done
	@grep -q '^schema_url: .*/$(subst .,\.,$(VERSION))$$' $(PACKAGE_OUTPUT)/manifest.yaml \
		|| { echo "FAIL: packaged manifest does not carry version $(VERSION)."; exit 1; }
	@grep -qxF 'resolved_registry_uri: $(RELEASE_DOWNLOAD_URL)/registry.yaml' $(PACKAGE_OUTPUT)/manifest.yaml \
		|| { echo "FAIL: packaged manifest does not point at $(RELEASE_DOWNLOAD_URL)/registry.yaml."; exit 1; }
	@echo "OK: release assets for $(TAG) are in $(PACKAGE_OUTPUT)."

# Hermetic, like the other generate targets. The manifest points at the copy of
# the resolved registry that the site serves next to it. The `unreleased`
# marker tells the site that these files do not come from a release: the deploy
# workflow only uses this target for a version that has none.
site-schemas: offline-model ## Build the registry the docs site serves at the schema URL
	@mkdir -p $(BUILD) $(SITE_SCHEMAS)/$(VERSION)
	@$(WEAVER_OFFLINE) registry package \
		-r $(OFFLINE_MODEL) \
		--v2 \
		--quiet \
		--resolved-registry-uri '$(SCHEMA_URL)/registry.yaml' \
		-o ./$(BUILD)/site-schemas \
		2> $(BUILD)/site-schemas.log || { cat $(BUILD)/site-schemas.log; exit 1; }
	@cp $(BUILD)/site-schemas/manifest.yaml $(SITE_SCHEMAS)/$(VERSION)/manifest.yaml
	@cp $(BUILD)/site-schemas/resolved.yaml $(SITE_SCHEMAS)/$(VERSION)/registry.yaml
	@echo "Built from the model, not downloaded from a release." > $(SITE_SCHEMAS)/$(VERSION)/unreleased
	@echo "Wrote the registry for $(VERSION) to $(SITE_SCHEMAS)/$(VERSION)"

print-version: ## Print the registry version
	@echo $(VERSION)

clean: ## Remove build output
	rm -rf $(BUILD)

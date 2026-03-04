# OpenInference Monorepo Makefile
# Unified commands for Python and JavaScript/TypeScript development

SHELL := /bin/bash
.DEFAULT_GOAL := help

# Tools
RUFF_VERSION := 0.9.2
RUFF := uvx ruff@$(RUFF_VERSION)
PNPM := pnpm
TOX := uvx --with tox-uv tox

# Directories
PYTHON_DIR := python
JS_DIR := js

# Source nvm and switch to the Node version in js/.nvmrc before running JS commands.
# Falls back to system Node if nvm is unavailable. Each Make recipe runs in its own subshell,
# so this must be included in every JS target.
NVM_USE = export NVM_DIR="$$HOME/.nvm"; if [ -s "$$NVM_DIR/nvm.sh" ]; then . "$$NVM_DIR/nvm.sh" && cd $(JS_DIR) && nvm use --silent; else cd $(JS_DIR); fi &&

# Colors for output
CYAN := \033[0;36m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

#=============================================================================
# Meta Targets
#=============================================================================

.PHONY: help check-tools setup install-python install-js \
	format format-python format-js fmt fmt-python fmt-js \
	lint lint-python lint-js

help: ## Show this help message
	@echo -e ""
	@echo -e "$(CYAN)OpenInference Monorepo - Available Make Targets$(NC)"
	@echo -e ""
	@echo -e "$(GREEN)Setup:$(NC)"
	@echo -e "  $(YELLOW)setup$(NC)                 - Install all dependencies (Python + JS/TS)"
	@echo -e "  install-python         - pip install deps + add_symlinks + editable install"
	@echo -e "  install-js             - pnpm install + build (Node.js 20.19+ or 22.12+; uses nvm if available)"
	@echo -e "  check-tools            - Verify required tools are installed (uv, node, pnpm)"
	@echo -e ""
	@echo -e "$(GREEN)Code Quality:$(NC)"
	@echo -e "  $(YELLOW)format$(NC)                - Format all code (Python + JS/TS)"
	@echo -e "  format-python          - Format Python with ruff"
	@echo -e "  format-js              - Format JS/TS with oxfmt"
	@echo -e "  $(YELLOW)lint$(NC)                  - Lint all code (Python + JS/TS)"
	@echo -e "  lint-python            - Lint Python with ruff"
	@echo -e "  lint-js                - Lint JS/TS with oxlint"
	@echo -e ""
	@echo -e "Highlighted targets are the most commonly used."
	@echo -e ""

check-tools: ## Verify required tools are installed
	@echo -e "$(CYAN)Checking required tools...$(NC)"
	@command -v uv >/dev/null 2>&1 || { echo -e "$(RED)ERROR: uv is not installed. Install from https://github.com/astral-sh/uv$(NC)"; exit 1; }
	@echo -e "$(GREEN)✓$(NC) uv found: $$(uv --version)"
	@if [ -s "$$HOME/.nvm/nvm.sh" ]; then \
		export NVM_DIR="$$HOME/.nvm"; \
		. "$$NVM_DIR/nvm.sh" && cd $(JS_DIR) && nvm use --silent && \
		node -e "const v=process.versions.node.split('.').map(Number); const ok=(v[0]===20&&v[1]>=19)||(v[0]>=22&&(v[0]>22||v[1]>=12)); if(!ok){process.stderr.write('\033[0;31mERROR: Node.js 20.19+ or 22.12+ required (oxfmt). Current: '+process.version+'\033[0m\n');process.exit(1);}" && \
		echo -e "$(GREEN)✓$(NC) node found: $$(node --version) (via nvm)"; \
	elif command -v node >/dev/null 2>&1; then \
		node -e "const v=process.versions.node.split('.').map(Number); const ok=(v[0]===20&&v[1]>=19)||(v[0]>=22&&(v[0]>22||v[1]>=12)); if(!ok){process.stderr.write('\033[0;31mERROR: Node.js 20.19+ or 22.12+ required (oxfmt). Current: '+process.version+'\033[0m\n');process.exit(1);}" && \
		echo -e "$(GREEN)✓$(NC) node found: $$(node --version)"; \
	else \
		echo -e "$(RED)ERROR: node is not installed. Install Node.js 20.19+ or 22.12+ from https://nodejs.org$(NC)"; \
		exit 1; \
	fi
	@command -v $(PNPM) >/dev/null 2>&1 || { echo -e "$(RED)ERROR: pnpm is not installed. Run: npm install -g pnpm$(NC)"; exit 1; }
	@echo -e "$(GREEN)✓$(NC) pnpm found: $$($(PNPM) --version)"
	@echo -e "$(GREEN)All required tools are installed!$(NC)"

#=============================================================================
# Setup
#=============================================================================

install-python: ## Install Python dev dependencies (per python/DEVELOPMENT.md)
	@echo -e "$(CYAN)Installing Python dev dependencies...$(NC)"
	@pip install -r $(PYTHON_DIR)/dev-requirements.txt
	@echo -e "$(CYAN)Composing openinference-instrumentation namespace package...$(NC)"
	@cd $(PYTHON_DIR) && $(TOX) run -e add_symlinks
	@echo -e "$(CYAN)Installing openinference-instrumentation in editable mode...$(NC)"
	@pip install -e $(PYTHON_DIR)/openinference-instrumentation
	@echo -e "$(GREEN)✓ Done$(NC)"

install-js: ## Install JS/TS dependencies and build packages (per js/DEVELOPMENT.md)
	@echo -e "$(CYAN)Installing JS/TS dependencies...$(NC)"
	@$(NVM_USE) $(PNPM) install --frozen-lockfile -r
	@echo -e "$(CYAN)Building JS/TS packages (includes prebuild)...$(NC)"
	@$(NVM_USE) $(PNPM) run -r build
	@echo -e "$(GREEN)✓ Done$(NC)"

setup: check-tools install-python install-js ## Install all dependencies (Python + JS/TS)
	@echo -e ""
	@echo -e "$(GREEN)✓ OpenInference development environment setup complete!$(NC)"
	@echo -e ""
	@echo -e "Next steps:"
	@echo -e "  - Format code:  make format"
	@echo -e "  - Lint code:    make lint"
	@echo -e ""

#=============================================================================
# Code Quality
#=============================================================================

format-python: ## Format Python code with ruff
	@echo -e "$(CYAN)Formatting Python code...$(NC)"
	@$(RUFF) format $(PYTHON_DIR)/ --config $(PYTHON_DIR)/ruff.toml
	@echo -e "$(GREEN)✓ Done$(NC)"

format-js: ## Format JS/TS code with oxfmt
	@echo -e "$(CYAN)Formatting JS/TS code...$(NC)"
	@$(NVM_USE) $(PNPM) run fmt
	@echo -e "$(GREEN)✓ Done$(NC)"

format: format-python format-js ## Format all code (Python + JS/TS)
	@echo -e "$(GREEN)✓ Code formatting complete$(NC)"

fmt: format ## Alias for format
fmt-python: format-python ## Alias for format-python
fmt-js: format-js ## Alias for format-js

lint-python: ## Lint Python code with ruff
	@echo -e "$(CYAN)Linting Python code...$(NC)"
	@$(RUFF) check $(PYTHON_DIR)/ --config $(PYTHON_DIR)/ruff.toml
	@echo -e "$(GREEN)✓ Done$(NC)"

lint-js: ## Lint JS/TS code with oxlint
	@echo -e "$(CYAN)Linting JS/TS code...$(NC)"
	@$(NVM_USE) $(PNPM) run lint
	@echo -e "$(GREEN)✓ Done$(NC)"

lint: lint-python lint-js ## Lint all code (Python + JS/TS)
	@echo -e "$(GREEN)✓ Linting complete$(NC)"

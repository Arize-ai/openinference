---
name: fix-ci-failures
description: Investigate and fix Python daily cron job failures in GitHub Actions
invocable: true
---

# Fix CI Failures (Python Daily Cron)

Interactive workflow for diagnosing and fixing Python daily cron job test failures in GitHub Actions.

This skill focuses on the **"Python Canary Cron"** workflow that runs daily against latest dependency versions to catch compatibility issues early.

## About the Python Canary Cron

**Purpose**: Early detection of breaking changes in upstream libraries
**Schedule**: Runs daily at 6PM UTC, weekdays only (configured in `.github/workflows/python-cron.yaml`)
**Scope**: Tests instrumentations with latest available dependency versions
**Target Branch**: `main`
**Job Selection**: Only runs test environments with `-latest` suffix (e.g., `py310-ci-openai-latest`)
  - Filters tox environments using `egrep -e '-latest$'`
  - Most packages have both regular (pinned) and `-latest` test environments in tox.ini
  - Regular CI (on PRs) runs pinned versions; daily cron runs `-latest` versions
  - This separation allows catching upstream breaking changes without blocking PR merges

**Why It Matters**:
- Catches breaking changes before users encounter them
- Provides early warning for needed instrumentation updates
- Helps maintain compatibility with evolving AI/ML ecosystem
- Allows proactive fixes rather than reactive bug reports

**Common Causes of Failures**:
1. **Breaking API changes** in instrumented libraries (most common)
2. **Deprecation removals** after warning period
3. **New features** that expose gaps in instrumentation coverage
4. **Dependency conflicts** in latest version combinations
5. **Upstream bugs** in newly released versions
6. **Python version incompatibilities** (especially with Python 3.13+)

## Workflow

**Key Principles:**
1. **Autonomous**: Reproduce failures immediately, don't wait for user input
2. **Parallel**: Run multiple tests and agents simultaneously for speed
3. **Thorough**: Always test both `-latest` and pinned versions
4. **Safe**: Get approval for complex fixes, but auto-apply simple ones
5. **Efficient**: Use `uvx` for fast iteration and local source inspection

**Step 1: Identify the Cron Job Run**
- Ask user for:
  - Run ID (optional, e.g., `21917652020`)
  - Specific failing package if known (optional)
- If no run ID provided, fetch latest cron job:
  - Use `gh run list --workflow="Python Canary Cron" --branch=main --limit 5`
  - Select the most recent completed run
- If run ID provided:
  - Use `gh run view <run-id>` directly

**Step 2: Fetch Cron Job Status**
- Use `gh run view <run-id>` to get detailed status
- Verify it's the Python Canary Cron workflow:
  - Check `workflowName: "Python Canary Cron"`
  - Check `event: "schedule"`
  - Check `headBranch: "main"`
- Identify failed jobs - all jobs have `-latest` suffix (testing latest dependency versions):
  - `py3X-ci-<package>-latest`: Test failures with latest dependency versions
  - Examples: `py310-ci-langchain-latest`, `py313-ci-openai-latest`
- Parse job names to extract:
  - Python version (py310, py313, py314)
  - Package name (openai, anthropic, langchain, pipecat, openlit, agent_framework, etc.)
  - All jobs use latest versions of instrumented libraries

**Step 3: Reproduce Locally (Autonomous)**
- For each failing package, immediately run local test to reproduce:
  ```bash
  cd python
  uvx --with tox-uv tox r -e py310-ci-<package>-latest -- -ra -x
  ```
- Run tests in parallel using multiple Bash tool calls (one per package)
- Capture output to verify if issue reproduces locally
- Benefits:
  - Confirms the failure is reproducible
  - Provides local environment for debugging
  - Source code available in `.tox/` for inspection
  - Faster than relying solely on CI logs

**Step 4: Parallel Root Cause Analysis (Launch Subagents)**
- For each failing package, launch a separate Task agent with `subagent_type: general-purpose`
- Run agents in parallel (single message with multiple Task tool calls)
- Each agent should:

  **Investigate the Failure:**
  1. Read relevant instrumentor code in `python/instrumentation/openinference-instrumentation-<package>/`
  2. Read test files that failed
  3. Inspect library source in `.tox/py310-ci-<package>-latest/lib/python3.X/site-packages/<library>/`
  4. Check recent commits: `git log --oneline -20 -- python/instrumentation/openinference-instrumentation-<package>/`

  **Identify Root Cause:**
  - **Breaking API changes** in latest version of instrumented library (most common)
  - **Deprecation removals** in latest library version
  - **Module reorganization** (import path changes)
  - **New API features** not yet supported by instrumentor
  - **Dependency conflicts** with latest versions
  - **Upstream bugs** in latest library releases
  - **Python version incompatibility** (especially Python 3.13+)

  **Special Case: When non-latest fails but -latest passes**

  This indicates a compatibility issue with older dependency versions. Investigate systematically:

  1. **Compare installed versions** between environments:
     ```bash
     # Get package lists
     .tox/py313-ci-<package>/bin/python -m pip list > /tmp/pinned.txt
     .tox/py313-ci-<package>-latest/bin/python -m pip list > /tmp/latest.txt

     # Key packages to compare:
     # - The instrumented library itself
     # - opentelemetry-* packages (especially opentelemetry-instrumentation)
     # - Any packages mentioned in error traceback
     ```

  2. **Identify the problematic version**:
     - If opentelemetry-instrumentation differs (e.g., 0.48b0 vs 0.60b1), investigate what changed
     - Look for removed features, deprecated APIs, or Python version incompatibilities
     - Check if old version uses removed stdlib modules (e.g., pkg_resources in Python 3.13+)

  3. **Check transitive dependencies**:
     - The failure might not be in the main package
     - Use `.tox/<env>/bin/python -m pip show <package>` to see what it depends on
     - Old version may pull in incompatible transitive deps
     - Example: openlit 1.26.0 → opentelemetry-instrumentation 0.48b0 → pkg_resources (missing in Python 3.13)

  4. **Determine fix strategy**:
     - Update minimum version if old version is incompatible with declared Python support
     - Add compatibility shims only if truly necessary
     - Document the dependency chain issue in commit/PR

  **Research Upstream Changes:**
  - Use WebSearch to find library changelog and breaking changes
  - Check GitHub releases: `gh api repos/<org>/<repo>/releases --jq '.[0:3]'`
  - Look for migration guides

  **Propose Fix:**
  - Specific code changes needed (with file paths and line numbers)
  - Whether fix is simple (imports, attribute names) or complex (API redesign)
  - Estimated risk of breaking non-latest (pinned) version tests
  - If version update needed, specify minimum version and rationale

  **Return:**
  - Structured analysis with root cause, fix proposal, and risk assessment

**Step 5: Review and Aggregate Findings**
- Wait for all subagents to complete their analysis
- Aggregate findings from all packages
- Categorize by:
  - **Simple fixes**: Import updates, attribute renames, type annotation fixes
  - **Complex fixes**: API redesigns, new callback implementations, significant refactoring
  - **Version updates**: Minimum version requirement changes for compatibility
- Identify dependencies between fixes (e.g., shared utility functions)
- Assess risk of breaking non-latest tests

**Step 6: Present Findings and Seek Approval (If Complex)**
```
## Daily Cron Job Failure Analysis

**Run**: Python Canary Cron #<run-number>
**Run ID**: <run-id>
**Date**: <date>
**Branch**: main
**Status**: ❌ Failed

### Failed Jobs (Latest Dependency Versions)
1. [py310-ci-langchain-latest] - 2 test failures
2. [py313-ci-langchain-latest] - 2 test failures
3. [py310-ci-pipecat-latest] - 1 import error
4. [py313-ci-openlit-latest] - 3 test failures

**Note**: All failures are with `-latest` versions, indicating potential breaking changes in upstream libraries.

### Root Causes

#### Critical Issues

**Import Error: py310-ci-pipecat-latest**
- Test: All tests
- Location: python/instrumentation/openinference-instrumentation-pipecat/
- Error: `ModuleNotFoundError: No module named 'pipecat.processors'`
- Root Cause: Pipecat v0.0.50 restructured module layout, moved processors to new location
- Evidence: Check Pipecat changelog - https://github.com/pipecat-ai/pipecat/releases
- Impact: Complete instrumentation failure

#### High Priority

**Test Failure: py310-ci-langchain-latest + py313-ci-langchain-latest**
- Test: `tests/test_chat_models.py::test_streaming_with_tool_calls`
- Location: python/instrumentation/openinference-instrumentation-langchain/tests/
- Error: `AssertionError: Expected 5 spans, got 3. Missing: [TOOL, CHAIN]`
- Root Cause: LangChain v0.3.0 changed tool calling API - `tool_calls` now in separate callback
- Evidence: Pinned version tests pass, only `-latest` fails
- Impact: Missing observability for tool usage in latest LangChain

**Test Failure: py313-ci-openlit-latest**
- Test: `tests/test_traces.py::test_trace_attributes`
- Error: `AttributeError: 'NoneType' object has no attribute 'model'`
- Root Cause: OpenLIT v1.20.0 changed response object structure
- Evidence: Latest release notes mention response format changes
- Impact: Attribute extraction fails for latest OpenLIT version

### Recommended Fixes

1. **[pipecat] Update import paths for v0.0.50**
   - File: `python/instrumentation/openinference-instrumentation-pipecat/src/openinference/instrumentation/pipecat/_wrappers.py`
   - Action: Change `from pipecat.processors import X` to `from pipecat.core.processors import X`
   - Urgency: Critical - instrumentation completely broken

2. **[langchain] Add tool call span creation for v0.3.0+**
   - File: `python/instrumentation/openinference-instrumentation-langchain/src/openinference/instrumentation/langchain/_tracer.py`
   - Action: Implement `on_tool_start` callback to capture new tool calling format
   - Urgency: High - missing observability for tool usage

3. **[openlit] Update response object handling for v1.20.0**
   - File: `python/instrumentation/openinference-instrumentation-openlit/src/openinference/instrumentation/openlit/_wrappers.py`
   - Action: Add None check and update attribute extraction logic for new response format
   - Urgency: High - breaks attribute collection
```

**Display organized report:**
```
## Daily Cron Job Failure Analysis

**Run**: Python Canary Cron #<run-number>
**Run ID**: <run-id>
**Date**: <date>
**Branch**: main
**Status**: ❌ X failures

### Summary by Package
1. **langchain** (py310, py313) - Type error: Missing attribute
   - **Severity**: HIGH
   - **Fix Complexity**: Simple (update import/attribute)
   - **Risk**: Low - pinned version should be unaffected

2. **pipecat** (py310, py313) - Import error: Module reorganization
   - **Severity**: CRITICAL
   - **Fix Complexity**: Simple (update import paths)
   - **Risk**: Medium - need version check for backward compatibility

3. **openlit** (py310, py313) - Test failure: API change
   - **Severity**: HIGH
   - **Fix Complexity**: Complex (new callback implementation)
   - **Risk**: High - may need version-conditional logic

[Detailed findings for each package...]
```

**Decision Point:**
- **If all fixes are simple**: Proceed automatically to Step 7
- **If any fixes are complex**: Use AskUserQuestion to get approval:
  - "Complex fixes detected. How would you like to proceed?"
    - "Proceed with all fixes" (auto-apply all)
    - "Show me the proposed changes first" (review before applying)
    - "Fix only simple issues now" (defer complex fixes)
    - "Pin problematic versions temporarily" (quick mitigation)

**Step 7: Dispatch Parallel Fix Agents**
- Launch separate Task agents for each package that needs fixes
- Run agents in parallel (single message with multiple Task tool calls)
- Each agent should:

  **Apply the Fix:**
  1. Read the files that need changes
  2. Use Edit tool to apply the fixes identified in Step 4
  3. Handle version compatibility:
     - Check if fix breaks non-latest tests
     - Add version checks if needed: `if packaging.version.parse(library.__version__) >= "X.Y.Z"`
     - Use try/except for imports if supporting multiple versions

  **Agent Instructions:**
  ```
  Fix <package> instrumentation for latest version compatibility:

  Root cause: [from Step 4 analysis]
  Required changes: [specific edits]
  Files to modify: [list]

  IMPORTANT:
  - Ensure changes work with latest version
  - Check if changes break pinned (non-latest) version
  - Add version checks or try/except if needed for backward compatibility

  After applying fixes, report:
  - Files modified
  - Changes made
  - Backward compatibility approach
  ```

**Step 8: Verify Fixes (Both Versions)**
After all fix agents complete:

**A. Test with -latest (primary target):**
- Run tests for all fixed packages in parallel:
  ```bash
  cd python
  # Run multiple in parallel (separate Bash calls)
  uvx --with tox-uv tox r -e py310-ci-<package1>-latest -- -ra -x
  uvx --with tox-uv tox r -e py310-ci-<package2>-latest -- -ra -x
  ...
  ```

**B. Test with pinned versions (regression check):**
- **CRITICAL**: Also test non-latest to ensure no regressions:
  ```bash
  cd python
  # Test pinned versions
  uvx --with tox-uv tox r -e py310-ci-<package1> -- -ra -x
  uvx --with tox-uv tox r -e py310-ci-<package2> -- -ra -x
  ...
  ```

**C. Report Results:**
- Show pass/fail for each package, both versions:
  ```
  ## Test Results

  ✓ langchain-latest: PASSED
  ✓ langchain (pinned): PASSED

  ✓ pipecat-latest: PASSED
  ⚠ pipecat (pinned): FAILED - needs version check

  ✓ openlit-latest: PASSED
  ✓ openlit (pinned): PASSED
  ```

**D. Handle Regressions:**
- If pinned version breaks, launch fix agent to add version compatibility:
  - Add version checks: `if version >= "X.Y.Z"`
  - Add fallback logic for older versions
  - Use try/except for changed imports/APIs
  - Re-test both versions

**Step 9: Final Verification and Commit**
Once all tests pass for both latest and pinned versions:
- Show summary of all changes made
- Ask: "All tests passing. Commit and push changes?"
  - If yes: Create commit(s) with descriptive messages
    ```
    fix(langchain): update for chat_models API changes in v1.2.x
    fix(pipecat): update imports for module reorganization in v0.0.50
    fix(openlit): add compatibility for response object changes in v1.20.0

    Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
    ```
  - Push to main
  - Note: Next cron job will validate across all Python versions

**When Updating Version Requirements:**

If you update minimum version requirements (e.g., `openlit>=1.0.0` → `openlit>=1.36.0`):

1. **In commit message**: Explain why version was updated with technical details
   ```
   fix(openlit): require openlit>=1.36.0 for Python 3.13 compatibility

   The older openlit 1.26.0 depends on opentelemetry-instrumentation 0.48b0,
   which uses pkg_resources (from setuptools). Python 3.13 doesn't include
   setuptools by default, causing import failures.

   Updating to openlit>=1.36.0 ensures compatibility with Python 3.13 by
   using the newer opentelemetry-instrumentation 0.60b1.

   Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
   ```

2. **In PR description**: Add or update "Version Requirement Changes" section:
   ```markdown
   ## Version Requirement Changes

   - **package-name**: `>= old.version` → `>= new.version` (brief reason)
     - **Root Cause**: Detailed explanation of compatibility issue
     - **Dependency Chain**: If transitive dependency issue (e.g., "old version → dep A 0.48b0 → pkg_resources")
     - **Rationale**: Why this specific version solves it
     - **Backward Compatibility**: Impact on older Python versions
   ```

3. **Include technical details**:
   - Specific incompatibility (e.g., "depends on pkg_resources not in Python 3.13")
   - Dependency chain if relevant (e.g., "via opentelemetry-instrumentation 0.48b0")
   - Why this version specifically (e.g., "0.60b1+ doesn't use pkg_resources")
   - Backward compatibility statement (e.g., "works on Python 3.9-3.13")

## Tox Commands Reference

Based on `python/tox.ini`:

### For Daily Cron Job Debugging (use -latest environments)

**Recommended: Using uvx for fast iteration** (no tox-uv installation required)
```bash
cd python

# Full CI check with latest dependencies (what cron runs)
uvx --with tox-uv tox r -e py310-ci-<package>-latest -- -ra -x

# Examples:
uvx --with tox-uv tox r -e py310-ci-openai-latest -- -ra -x
uvx --with tox-uv tox r -e py313-ci-langchain-latest -- -ra -x
uvx --with tox-uv tox r -e py310-ci-pipecat-latest -- -ra -x

# Stop at first failure and show all outcomes
# Source code available in .tox/py310-ci-<package>-latest/ for inspection
```

**Alternative: Traditional tox** (requires tox-uv installed)
```bash
cd python && tox run -e ci-<package>-latest
```

**Benefits of uvx approach:**
- No need to install `tox-uv` globally
- Faster iteration with `-- -ra -x` pytest flags
- Source code available in `.tox/<env>/lib/python3.X/site-packages/` for debugging
- Can easily inspect instrumented library source to understand breaking changes

The `-latest` environments:
1. Install latest available versions of all dependencies
2. Run same checks as regular CI: ruff + mypy + pytest
3. May install versions newer than what's in pyproject.toml
4. Useful for reproducing cron job failures locally

### Individual Checks (latest versions)

**Using uvx (Recommended):**
```bash
cd python

# Run tests only with latest deps
uvx --with tox-uv tox r -e py310-test-<package>-latest -- -ra -x

# Run mypy only (usually doesn't need -latest variant)
uvx --with tox-uv tox r -e mypy-<package>

# Run ruff with exact CI version (avoids version mismatch)
uvx --from ruff==0.9.2 ruff check --fix instrumentation/openinference-instrumentation-<package>/
```

**Traditional tox:**
```bash
cd python

# Run tests only with latest deps
tox run -e test-<package>-latest

# Run mypy only
tox run -e mypy-<package>

# Run ruff only
tox run -e ruff-<package>
```

**Pro tip for ruff**: Always use `uvx --from ruff==X.Y.Z` with the exact version from CI to avoid "works locally but fails in CI" scenarios.

### Common Packages in Daily Cron (from envlist)
**Most Active** (frequently updated libraries):
- openai-latest, anthropic-latest
- langchain-latest, langchain_core-latest
- llama_index-latest, llamaindex_core-latest
- pipecat-latest, openlit-latest
- crewai-latest, autogen-latest

**Stable but Monitored**:
- bedrock-latest, mistralai-latest, vertexai-latest
- groq-latest, litellm-latest, instructor-latest
- haystack-latest, dspy-latest
- pydantic_ai-latest, smolagents-latest, beeai-latest
- mcp-latest, google_genai-latest, google_adk-latest

**Note**: Not all packages have `-latest` variants. Check tox.ini envlist for complete list.

### Prerequisites

**Option 1: Using uvx (Recommended - No Installation Required)**
```bash
# No setup needed! uvx handles tox-uv automatically
cd python
uvx --with tox-uv tox r -e py310-ci-<package>-latest -- -ra -x
```

**Option 2: Traditional Installation**
```bash
# Install tox-uv first
pip install tox-uv==1.11.2

# Install dev requirements
pip install -r python/dev-requirements.txt

# Add symlinks for namespace packages
cd python && tox run -e add_symlinks
```

### Inspecting .tox Environment for Debugging

After running tests with uvx or tox, the environment persists for inspection:

**View installed packages and versions:**
```bash
.tox/py310-ci-<package>-latest/bin/python -m pip list
```

**Inspect instrumented library source code:**
```bash
# Navigate to library source
cd .tox/py310-ci-<package>-latest/lib/python3.10/site-packages/<library-name>

# Example: Check langchain-core source
cd .tox/py310-ci-langchain-latest/lib/python3.10/site-packages/langchain_core

# Read files to understand API changes
cat chat_models.py
```

**Run Python REPL with exact CI environment:**
```bash
.tox/py310-ci-<package>-latest/bin/python
>>> import langchain_core
>>> dir(langchain_core.chat_models)  # Inspect available APIs
```

**Why this is useful:**
- See exact code causing failures without searching GitHub
- Understand how new APIs work by reading source
- Test API behavior interactively in same environment as CI
- Compare old vs new behavior side-by-side

## Common Failure Patterns

### Import Errors
**Symptom**: `ModuleNotFoundError: No module named 'openinference.instrumentation.<package>'`
**Cause**: Namespace package not properly composed
**Fix**: Run `cd python && tox run -e add_symlinks`

### Import-Time vs Runtime Failures

Understanding the difference is critical for choosing the right fix approach.

**Import-Time Failure** (collection error):
- **Symptom**: Error during `import` statement or module-level code
  ```
  ERROR collecting tests/test_foo.py
  ImportError while importing test module
  ```
- **Characteristics**:
  - Occurs before any test code runs
  - Happens during pytest's collection phase
  - Often caused by: missing dependencies, incompatible library versions, syntax errors
- **Important**: `pytest.mark.skip` decorator **WON'T help** because the test file can't even be loaded
- **Fix Strategies**:
  - Update dependencies to compatible versions
  - Fix import statements
  - Add missing packages
  - Update minimum version requirements if old version is incompatible

**Runtime Failure** (execution error):
- **Symptom**: Error during test execution
  ```
  FAILED tests/test_foo.py::test_bar
  AssertionError: ...
  ```
- **Characteristics**:
  - Occurs after imports succeed
  - Happens when test code actually runs
  - Test file successfully loads
- **Important**: `pytest.mark.skip` decorator **WILL help** and is appropriate
- **Fix Strategies**:
  - Skip problematic tests temporarily
  - Fix test assertions
  - Update test expectations

**How to distinguish**:
```
ERROR collecting tests/test_foo.py       ← Import-time (can't skip with decorator)
FAILED tests/test_foo.py::test_bar       ← Runtime (can skip with decorator)
```

**Real Example** (from this session):
- openlit with Python 3.13: `ModuleNotFoundError: No module named 'pkg_resources'`
  - This was import-time (ERROR collecting)
  - Skip decorator didn't help because imports failed
  - Solution: Update minimum openlit version

### Test Failures - Missing Spans
**Symptom**: `AssertionError: Expected X spans, got Y`
**Cause**:
- Missing `_instrument()` call in test setup
- New code path not creating spans
- Span processor not registered
**Fix**:
- Check conftest.py has proper instrumentation setup
- Verify instrumentor creates spans for all code paths
- Ensure span processors are registered in correct order

### Test Failures - Wrong Attributes
**Symptom**: `KeyError: 'attribute.name'` or `AssertionError: attribute value mismatch`
**Cause**:
- Outdated semantic conventions version
- Missing attribute mapping
- Attribute name changed in spec
**Fix**:
- Update to latest openinference-semantic-conventions
- Check spec/ directory for correct attribute names
- Update instrumentor to use correct attribute keys

### Type Errors - Incompatible Types
**Symptom**: `Argument 1 has incompatible type "X"; expected "Y"`
**Cause**:
- Missing type annotations
- Incorrect type hints
- Missing None checks for optional values
**Fix**:
- Add proper type hints
- Use `Optional[T]` or `T | None` for optional values
- Add runtime None checks before passing to functions

### Ruff Errors - Linting Issues
**Symptom**: `F401 'module.Class' imported but unused` or other ruff violations
**Cause**:
- Import no longer needed after refactoring
- Code doesn't meet linting standards
- Version mismatch between local ruff and CI ruff
**Fix**:
1. **Use pinned ruff version** (matches CI exactly):
   ```bash
   uvx --from ruff==0.9.2 ruff check --fix python/instrumentation/openinference-instrumentation-<package>/
   ```
   - Check `.github/workflows/` or `tox.ini` for exact ruff version used in CI
   - Using `uvx` ensures same version as CI without installing globally
2. **Automatic fix**:
   ```bash
   cd python/instrumentation/openinference-instrumentation-<package>
   ruff check --fix .
   ```
3. **Manual fix**: Remove unused imports or fix violations manually

**Finding CI's ruff version:**
- Check tox.ini for ruff version constraint
- Check .github/workflows/ for ruff-action version
- Look at CI logs for "ruff X.Y.Z" version output

### Version Compatibility Issues (Most Common in Cron Jobs)
**Symptom**: Daily cron job fails with `ci-<package>-latest` errors
**Cause**: Breaking change in instrumented library's latest version
**Common Breaking Changes**:
- **API Restructuring**: Methods moved, renamed, or removed
- **Module Reorganization**: Import paths changed (e.g., `pipecat.processors` → `pipecat.core.processors`)
- **Signature Changes**: Function parameters added, removed, or reordered
- **Response Format Changes**: Return values have different structure
- **Deprecation Removals**: Previously deprecated APIs finally removed
**Fix Strategy**:
1. **Immediate**: Pin to last working version (temporary)
   - Example: `langchain-core<0.3.0` in pyproject.toml
2. **Permanent**: Update instrumentor to support new API
   - Check library's migration guide
   - Update imports, method calls, and response handling
   - Add version-conditional logic if supporting multiple versions
3. **Report Upstream**: If it's a bug in the library, file an issue
**Investigation Steps**:
- Compare error between pinned and latest versions
- Check library's GitHub releases and changelog
- Search for migration guides or breaking change announcements
- Look for similar issues in library's GitHub issues

### Python Version-Specific Compatibility Issues

**Symptom**: Non-latest environment fails on py313+ but passes on py310/py311, or -latest passes but pinned fails on same Python version

**Common Causes**:
1. **Missing stdlib modules**: Python 3.13+ doesn't include setuptools/distutils by default
2. **pkg_resources dependency**: Older packages using pkg_resources fail on 3.13+
3. **Transitive dependencies**: Package A (old version) depends on Package B (old version) that uses removed APIs
4. **Deprecated stdlib removals**: APIs deprecated in 3.11-3.12 removed in 3.13

**Investigation Strategy**:

1. **Compare installed package versions** between working and failing environments:
   ```bash
   # Check what's installed in failing environment
   .tox/py313-ci-<package>/bin/python -m pip list

   # Compare with working -latest environment
   .tox/py313-ci-<package>-latest/bin/python -m pip list

   # Look for differences in key packages:
   # - opentelemetry-instrumentation (0.48b0 vs 0.60b1)
   # - The instrumented library itself
   # - Any setuptools/pkg_resources users
   ```

2. **Look for old versions of problematic packages**:
   - `opentelemetry-instrumentation==0.48b0` uses pkg_resources (incompatible with Python 3.13)
   - `opentelemetry-instrumentation==0.60b1+` doesn't use pkg_resources (compatible)
   - Check what the failing package's old version depends on

3. **Trace the dependency chain**:
   ```bash
   # See what a package depends on
   .tox/py313-ci-<package>/bin/python -m pip show <package>

   # Example investigation:
   # openlit 1.26.0
   #   → depends on opentelemetry-instrumentation 0.48b0
   #   → which imports from pkg_resources
   #   → which requires setuptools
   #   → which isn't in Python 3.13 by default
   ```

4. **Check if main package's old version is incompatible**:
   - If the package predates Python 3.13 support
   - If it has known Python 3.13 issues

**Fix Strategies**:

1. **Update minimum version** (preferred if old version is incompatible):
   - Example: `openlit>=1.0.0` → `openlit>=1.36.0`
   - Use when: Old version predates Python version support
   - Benefits: Clean fix, no workarounds needed
   - Document: Explain dependency chain in commit/PR

2. **Add missing dependencies** (only if package truly needs them):
   - Example: Add `setuptools` to test dependencies
   - Use when: Package legitimately uses setuptools features (plugins, entry points)
   - Caution: Don't use as workaround for deeper incompatibility

3. **Conditional imports** (when supporting multiple versions):
   ```python
   try:
       from pkg_resources import ...
   except ImportError:
       from importlib.metadata import ...
   ```
   - Use when: Need to support both old and new Python versions
   - More complex but backward compatible

**Real Example** (from this session):
```
Problem: py313-ci-openlit failing with "No module named 'pkg_resources'"
Investigation:
  - py313-ci-openlit: openlit 1.26.0 → opentelemetry-instrumentation 0.48b0 → pkg_resources ❌
  - py313-ci-openlit-latest: openlit 1.36.8 → opentelemetry-instrumentation 0.60b1 → no pkg_resources ✓
Solution: Update `openlit>=1.0.0` → `openlit>=1.36.0`
Rationale: Old version incompatible with Python 3.13 via transitive dependency
```

## Tips for Daily Cron Debugging

**Autonomous Workflow Best Practices:**
- **Always run in parallel**: Use multiple Bash tool calls in a single message for concurrent execution
- **Test first, analyze later**: Reproduce locally before deep investigation to save time
- **Use subagents for parallelism**: Launch one Task agent per package for concurrent analysis
- **Aggregate intelligently**: Wait for all agents, then synthesize findings before presenting
- **Version compatibility is critical**: Always test both `-latest` and pinned versions before committing
- **Simple vs Complex decision**: Auto-apply simple fixes (imports, attributes), ask for approval on complex (API redesigns)

**Understanding Cron Failures:**
- Daily cron runs against `main` branch, so failures indicate regressions or upstream changes
- All jobs use `-latest` suffix, installing newest available versions
- Failures often indicate breaking changes in instrumented libraries
- Multiple packages failing simultaneously may indicate shared dependency issue

**Debugging Strategy:**
- Check when failure started: review previous cron runs with `gh run list --workflow="Python Canary Cron" --limit 10`
- Identify pattern: single package or multiple packages?
- Check library release dates: did a new version just release?
- Look for version updates in pyproject.toml or tox.ini in recent commits
- Use `uv pip list -v` in tox output to see exact installed versions
- Compare with pinned version tests (if available) to isolate version-specific issues

**Detecting and Fixing Stale Environments:**

Tests can fail if `.tox` environments cache old test code or dependencies. This is a common source of confusion.

**Symptoms**:
- Test keeps failing even after fix is applied
- Changes to pyproject.toml dependencies not reflected
- Old import errors persist after import fixes
- Installed package versions don't match expectations

**Why This Happens**:
- Tox reuses existing environments for speed
- Dependency changes in pyproject.toml may not trigger rebuild
- Test code is cached in site-packages
- Environment created before your fixes

**Solution**:
```bash
# Always rebuild environment after dependency changes
rm -rf .tox/py310-ci-<package>-latest
uvx --with tox-uv tox run -e py310-ci-<package>-latest -- -ra -x

# Or rebuild all environments for a package
rm -rf .tox/py*-ci-<package>*
```

**When to Rebuild**:
- **Always**: After changing pyproject.toml dependencies or version requirements
- **Usually**: After updating test code that's being cached
- **Sometimes**: After any unexplained "this should work now" failures
- **Debug step**: If a fix seems correct but tests still fail

**Pro Tip**: After rebuilding, verify the change took effect:
```bash
.tox/py310-ci-<package>-latest/bin/python -m pip show <changed-package>
# Check that version matches your new requirement
```

**Local Reproduction:**
- **Best practice**: Use `uvx` for fast iteration without installing tox-uv:
  ```bash
  cd python
  uvx --with tox-uv tox r -e py310-ci-<package>-latest -- -ra -x
  ```
  - `-ra`: Shows summary of all outcomes (passed, failed, skipped, etc.)
  - `-x`: Stops on first failure for faster feedback loop
  - Source code lands in `.tox/py310-ci-<package>-latest/` for inspection
- **Alternative**: Traditional tox (requires `pip install tox-uv==1.11.2`)
  ```bash
  cd python && tox run -e ci-<package>-latest
  ```
- **Debugging tip**: After test run, inspect library source in `.tox/<env>/lib/python3.X/site-packages/<library>` to understand what changed
- CI uses `pytest -n auto` (parallel), local can use `-- -n 0` (serial) for easier debugging

**Version Investigation:**
- Check library's GitHub releases: `gh api repos/<org>/<repo>/releases`
- Use WebSearch to find changelogs and migration guides
- Look for "breaking changes", "deprecations", or "migration" in release notes

**When to Pin vs Fix:**
- **Pin temporarily** if: Breaking change is severe, needs research, or blocking other work
- **Fix immediately** if: Simple import change, obvious API update, or quick adaptation
- Always create a tracking issue for temporary pins

## Example Usage - Autonomous Workflow

```
User: /fix-ci-failures

Claude: Let me check the latest Python Canary Cron job...
Claude: [Runs gh run list --workflow="Python Canary Cron"]
Claude: Found run 21917652020 with 8 failures across 4 packages:
        - langchain (py310, py313)
        - pipecat (py310, py313)
        - openlit (py310, py313)
        - agent_framework (py310, py314)

Claude: Reproducing failures locally...
Claude: [Runs 4 parallel uvx commands to reproduce each package]
        uvx --with tox-uv tox r -e py310-ci-langchain-latest -- -ra -x
        uvx --with tox-uv tox r -e py310-ci-pipecat-latest -- -ra -x
        uvx --with tox-uv tox r -e py310-ci-openlit-latest -- -ra -x
        uvx --with tox-uv tox r -e py310-ci-agent_framework-latest -- -ra -x

Claude: ✓ All 4 failures reproduced locally

Claude: Launching root cause analysis agents...
Claude: [Launches 4 parallel Task agents, one per package]

[Agents work in parallel for ~30-60 seconds]

Claude: Analysis complete! Here's the summary:

## Daily Cron Job Failure Analysis

**Run**: Python Canary Cron #21917652020
**Date**: 2026-02-11
**Branch**: main
**Status**: ❌ 8 failures (4 packages)

### Summary by Package

1. **langchain** - Type error
   - **Root Cause**: `_SUPPORTED_PROVIDERS` removed from `langchain.chat_models.base` in v1.2.10
   - **Fix**: Remove usage in tests, use public API instead
   - **Severity**: HIGH | **Complexity**: Simple | **Risk**: Low

2. **pipecat** - Import error
   - **Root Cause**: Module reorganization in v0.0.50: `pipecat.processors` → `pipecat.core.processors`
   - **Fix**: Update all import statements
   - **Severity**: CRITICAL | **Complexity**: Simple | **Risk**: Medium (needs version check)

3. **openlit** - Test failure
   - **Root Cause**: Response object structure changed in v1.20.0
   - **Fix**: Add None checks, update attribute extraction
   - **Severity**: HIGH | **Complexity**: Medium | **Risk**: Low

4. **agent_framework** - API change
   - **Root Cause**: New callback API in v2.0.0
   - **Fix**: Implement new callback methods
   - **Severity**: HIGH | **Complexity**: Complex | **Risk**: High

Claude: Complex fix detected for agent_framework. How would you like to proceed?
User: [Selects "Proceed with all fixes"]

Claude: Dispatching fix agents...
Claude: [Launches 4 parallel Task agents to apply fixes]

[30-60 seconds later]

Claude: Fixes applied! Verifying with tests...

Claude: Testing -latest versions...
Claude: [Runs 4 parallel test commands]
        ✓ langchain-latest: PASSED
        ✓ pipecat-latest: PASSED
        ✓ openlit-latest: PASSED
        ✓ agent_framework-latest: PASSED

Claude: Testing pinned versions (regression check)...
Claude: [Runs 4 parallel test commands for non-latest]
        ✓ langchain (pinned): PASSED
        ⚠ pipecat (pinned): FAILED - old version doesn't have pipecat.core

Claude: Detected regression in pipecat. Adding version check...
Claude: [Launches fix agent to add version compatibility]
Claude: [Re-tests pipecat pinned version]
        ✓ pipecat (pinned): PASSED

## Final Test Results

✓ All 8 tests passing (4 packages × 2 versions)

Changes made:
- langchain: Updated test to use public API (1 file)
- pipecat: Updated imports with version check (3 files)
- openlit: Added None checks for response handling (2 files)
- agent_framework: Implemented new callback API (4 files)

Claude: All tests passing. Commit and push changes?
User: [Confirms]

Claude: [Creates commit with changes]
Claude: [Pushes to main]
Done! Next daily cron will validate across all Python versions.
```

## Quick Reference: Essential Commands

**Check latest cron job:**
```bash
gh run list --workflow="Python Canary Cron" --limit 5
gh run view <run-id>
gh run view <run-id> --log-failed
```

**Run tests locally (fast iteration):**
```bash
cd python
uvx --with tox-uv tox r -e py310-ci-<package>-latest -- -ra -x
```

**Fix linting with exact CI version:**
```bash
uvx --from ruff==0.9.2 ruff check --fix python/instrumentation/openinference-instrumentation-<package>/
```

**Inspect library source after test run:**
```bash
# View installed packages
.tox/py310-ci-<package>-latest/bin/python -m pip list

# Navigate to library source
cd .tox/py310-ci-<package>-latest/lib/python3.10/site-packages/<library-name>/

# Interactive Python with exact CI environment
.tox/py310-ci-<package>-latest/bin/python
```

**Compare package versions between environments:**
```bash
# Check specific package version in environment
.tox/py313-ci-<package>/bin/python -c "import <lib>; print(<lib>.__version__)"

# Compare dependency trees
.tox/py313-ci-<package>/bin/python -m pip show <package>
.tox/py313-ci-<package>-latest/bin/python -m pip show <package>

# Full package list comparison
.tox/py313-ci-<package>/bin/python -m pip list > /tmp/pinned.txt
.tox/py313-ci-<package>-latest/bin/python -m pip list > /tmp/latest.txt
diff /tmp/pinned.txt /tmp/latest.txt
```

## Monitoring and Triage

**Finding Recent Cron Runs:**
```bash
# List recent cron runs
gh run list --workflow="Python Canary Cron" --limit 10

# Check specific run
gh run view <run-id>

# Get failed job logs
gh run view <run-id> --log-failed
```

**Prioritization Guidelines:**
1. **Critical** (fix immediately): Import errors, complete instrumentation failures
2. **High** (fix within 1-2 days): Multiple test failures, missing core functionality
3. **Medium** (fix within week): Single test failures, attribute issues
4. **Low** (monitor): Intermittent failures, edge cases

**When to Pin vs Fix:**
- **Pin** if breaking change requires significant refactoring (>1 day work)
- **Fix** if change is straightforward (import updates, simple API changes)
- Always document pinning decision in commit message or issue

**Communication:**
- For severe breaking changes, consider notifying team via Slack/issue
- Document workarounds in issue tracker for users encountering problems
- Update documentation if instrumentation capabilities change

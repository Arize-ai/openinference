#!/usr/bin/env bash
# Prepares a factory sandbox for the languages an issue touches: `python`, `js`, or both
# (the default). Python needs no setup here, since tox builds each package's environment.
# JS runs `make install-js` on the sandbox's Node, which must meet the Makefile's floor
# (20.19+ or 22.12+, for oxfmt); CI runs Node 20. It ends with a report of what works.
# Anything it reports missing is a verification gap for the PR body; CI runs the full
# matrix.
set -uo pipefail
cd "$(dirname "$0")/.."

languages=("$@")
[ ${#languages[@]} -eq 0 ] && languages=(python js)

wants() {
  local language
  for language in "${languages[@]}"; do [ "$language" = "$1" ] && return 0; done
  return 1
}

if wants js; then
  make install-js >/tmp/factory-install-js.log 2>&1
  install_js_status=$?
fi

echo "--- factory bootstrap report"
echo "languages: ${languages[*]}"
echo "gh: $(gh --version 2>/dev/null | head -n 1 | grep . || echo missing)"
if wants python; then
  echo "python: $(python3 --version 2>/dev/null || echo missing)"
  echo "uv: $(uv --version 2>/dev/null || echo missing)"
  echo "tox: $(tox --version 2>/dev/null | head -n 1 | grep . || echo missing)"
  echo "tox-uv: $(python3 -c 'import tox_uv' 2>/dev/null && echo ok || echo missing)"
  echo "tox envs: \`tox run -e ci-<env>\` from python/ runs on this interpreter; CI runs the two versions on the package's envlist line in python/tox.ini (usually py310 and py314)"
fi
if wants js; then
  echo "node: $(node --version 2>/dev/null || echo missing) (CI runs Node 20; the Makefile needs 20.19+ or 22.12+)"
  echo "pnpm: $(pnpm --version 2>/dev/null || echo missing)"
  if [ "$install_js_status" -eq 0 ]; then
    echo "make install-js: ok"
  else
    echo "make install-js: failed (exit $install_js_status); last lines of /tmp/factory-install-js.log:"
    tail -n 20 /tmp/factory-install-js.log
  fi
fi

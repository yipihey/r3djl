#!/usr/bin/env bash
# Push r3djl to GitHub. Run this from the directory containing the r3djl/ folder.
#
# Prerequisites:
#   - gh CLI installed and authenticated: https://cli.github.com
#     OR you create the repo manually on github.com and skip the `gh repo create` step.
#   - git installed.
#
# Usage:
#   ./push_to_github.sh                  # creates repo as yipihey/r3djl
#   ./push_to_github.sh myname/r3d.jl    # creates repo at the given path
#
# This script is idempotent enough to re-run if something fails partway.

set -euo pipefail

REPO="${1:-yipihey/r3djl}"

if [[ ! -d "r3djl" ]]; then
    echo "Error: r3djl/ directory not found in $(pwd)" >&2
    echo "Extract r3djl.tar.gz here first." >&2
    exit 1
fi

cd r3djl

if [[ ! -d .git ]]; then
    echo "==> Initializing git repo"
    git init -b main
    git add -A
    git commit -m "Initial commit: pure-Julia r3d port + C wrapper + benchmark suite

A Julia port of Devon Powell's r3d library for fast, robust polyhedral
clipping and analytic moment integration, plus a thin ccall wrapper
around the upstream C library for differential testing and performance
comparison.

Status:
- 599 tests passing (33 pure-Julia, 158 differential vs C, 408 Flat
  including 200 random three-way cross-validation trials)
- Two implementations: AoS reference (parametric over D, T, S) and
  Flat SoA variant (3.7× faster end-to-end, ~50× fewer allocations)
- Performance: full pipeline at 4.4× C, reduce kernel at 2× C
- C wrapper covers init_box, init_tet, clip, reduce, is_good

See README.md and docs/performance.md for details."
else
    echo "==> Git repo already initialized; skipping init"
fi

if [[ -z "$(git remote 2>/dev/null)" ]]; then
    if command -v gh >/dev/null 2>&1; then
        echo "==> Creating GitHub repo $REPO via gh CLI"
        gh repo create "$REPO" --public --source=. --remote=origin --push \
            --description "Pure-Julia port of devonmpowell/r3d with differential testing against the C library"
    else
        echo "==> gh CLI not found. Create the repo manually:"
        echo "    1. Visit https://github.com/new and create $REPO (public)"
        echo "    2. Then run:"
        echo "       git remote add origin git@github.com:$REPO.git"
        echo "       git push -u origin main"
        exit 0
    fi
else
    echo "==> Remote already configured; pushing"
    git push -u origin main
fi

echo "==> Done. Repo URL: https://github.com/$REPO"

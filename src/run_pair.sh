#!/usr/bin/env bash
# run_pair.sh — eps-sweep helper. Runs only cogeo + pgd_bare for one (eps, backbone)
# cell, then runs eval for those two. Reuses the rest of run_4way.sh logic.
set -euo pipefail
exec env METHODS="cogeo pgd_bare" "$(dirname "$0")/run_4way.sh"

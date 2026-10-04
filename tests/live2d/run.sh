#!/usr/bin/env bash
# Run the Live2D detailed-split test-suite (no GPU / torch needed).
# On the agent box the root FS is full: keep temp files on /mnt/stx.
set -e
cd "$(dirname "$0")/../.."
for d in /workspace /mnt/stx; do if [ -w "$d" ] && [ -d "$d/venv" ]; then export TMPDIR=$d/tmp; mkdir -p "$TMPDIR"; PY=${PY:-$d/venv/bin/python}; break; fi; done
PY=${PY:-/mnt/stx/venv/bin/python}
[ -x "$PY" ] || PY=python
exec "$PY" -m pytest tests/live2d -q -p no:cacheprovider "$@"

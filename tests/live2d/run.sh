#!/usr/bin/env bash
# Run the Live2D detailed-split test-suite (no GPU / torch needed).
# On the agent box the root FS is full: keep temp files on /mnt/stx.
set -e
cd "$(dirname "$0")/../.."
if [ -d /mnt/stx ]; then export TMPDIR=/mnt/stx/tmp; mkdir -p "$TMPDIR"; fi
PY=${PY:-/mnt/stx/venv/bin/python}
[ -x "$PY" ] || PY=python
exec "$PY" -m pytest tests/live2d -q -p no:cacheprovider "$@"

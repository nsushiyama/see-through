#!/usr/bin/env bash
# usage: AGENT/tools/checkpoint.sh "commit message"   (commits all, pushes to origin fork)
set -e
cd "$(dirname "$0")/../.."
git add -A
git commit -qm "$1"
git push -q origin live2d-detailed-split
git log --oneline -1

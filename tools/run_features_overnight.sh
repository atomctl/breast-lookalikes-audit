#!/bin/zsh
# Extract features for all six models, most important first. Resumable: run it again to continue.
# Usage: caffeinate -is zsh tools/run_features_overnight.sh [model ...]
set -u
P=~/learning/pathology-lookalikes
MODELS=(${@:-resnet50 vit_l_imagenet uni virchow2 uni2_h hoptimus0})
for m in $MODELS; do
  "$P/.venv/bin/python" -I -W ignore "$P/tools/extract_features.py" \
    "$P/data/bracs/BRACS_RoI/latest_version" "$P/data/features" "$m" --check-fp32
  echo "$(date '+%Y-%m-%d %H:%M:%S') [$m] exit code $?"
done

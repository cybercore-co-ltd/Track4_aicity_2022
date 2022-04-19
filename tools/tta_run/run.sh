#!/usr/bin/env bash
if [ -z "$1" ]; then
    echo "No videos path specified"
    exit 0
fi

if [ -z "$2" ]; then
    echo "No output path specified"
    exit 0
fi

for f in ./tools/tta_run/models/*.sh; do
  bash "$f" $1 $2
done
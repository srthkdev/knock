#!/usr/bin/env bash
# Run Sol, Luna, Terra and Sonnet on every factory level, 4 episodes at a time per model.
#   scripts/run_factory_suite.sh levels/factory runs/factory_v1
set -u
LEVELS=${1:-levels/factory}; OUT=${2:-runs/factory_v1}
mkdir -p "$OUT"
for m in gpt-6.1-sol gpt-6-luna gpt-5.6-terra sonnet; do
  s=${m##*-}
  ls "$LEVELS"/*.json | xargs -P 4 -I{} sh -c "uv run python scripts/run_factory.py $m $OUT/$s {} >> $OUT/$s.log 2>&1 < /dev/null" &
done
wait

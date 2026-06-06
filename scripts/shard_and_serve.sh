#!/usr/bin/env bash
# Shard TinyLlama and launch the local shard swarm.
#
# Usage:
#   cd hivemind-models
#   bash scripts/shard_and_serve.sh          # shard + launch
#   bash scripts/shard_and_serve.sh --skip-shard  # launch only (if already sharded)
#
# Prerequisites:
#   pip install -r requirements.txt

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="configs/tinyllama-1b.yaml"
OUTPUT_DIR="output/tinyllama-1b-q4"
SKIP_SHARD=false
DEVICE="cpu"

for arg in "$@"; do
  case $arg in
    --skip-shard) SKIP_SHARD=true ;;
    --cuda) DEVICE="cuda" ;;
  esac
done

# ── Step 1: Shard the model ──────────────────────────────────────────
if [ "$SKIP_SHARD" = false ] || [ ! -f "$OUTPUT_DIR/manifest.json" ]; then
  echo "=== Sharding TinyLlama 1.1B ==="
  python -c "
import sys, logging
sys.path.insert(0, '.')
logging.basicConfig(level=logging.INFO)
from src.convert.sharder import create_sharder_from_config
sharder = create_sharder_from_config('$CONFIG')
result = sharder.shard()
print(f'Done: {len(result.shards)} shards, {result.total_size_bytes / 1e6:.1f} MB')
"
  echo ""
fi

# ── Step 2: Launch shard swarm ───────────────────────────────────────
echo "=== Launching shard servers ==="
python scripts/run_local_swarm.py \
  --shard-dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --test

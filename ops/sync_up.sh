#!/usr/bin/env bash
# Upload code + packed data to the instance (no raw data, no secrets).
#   ops/sync_up.sh <ssh_host> <ssh_port>     e.g. ops/sync_up.sh ssh5.vast.ai 12345
set -euo pipefail

HOST=${1:?host}; PORT=${2:?port}
SSH="ssh -p $PORT -o StrictHostKeyChecking=no"
DEST="root@$HOST:/workspace/claims-fm"

$SSH "root@$HOST" 'mkdir -p /workspace/claims-fm/data/processed'
rsync -avz -e "$SSH" --exclude __pycache__ \
  src scripts configs Makefile "$DEST/"
rsync -avz -e "$SSH" \
  data/processed/pretrain_pack data/processed/vocab.json data/processed/token_counts.parquet \
  "$DEST/data/processed/"
$SSH "root@$HOST" 'cd /workspace/claims-fm && pip install -q polars pyyaml && python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"'
echo "ready. start training with:"
echo "  $SSH root@$HOST"
echo "  tmux new -s train"
echo "  cd /workspace/claims-fm && PYTHONPATH=src python scripts/train_mlm.py --config configs/pretrain.yaml"

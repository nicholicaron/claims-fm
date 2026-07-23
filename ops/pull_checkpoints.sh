#!/usr/bin/env bash
# Off-instance checkpoint sync (SPEC §10: assume the instance dies).
# Pulls best.pt / last.pt / metrics.jsonl every 2 minutes until interrupted.
# Rolling step_*.pt files are NOT pulled: the remote prunes them to 3 but a
# local mirror never would — at 46M-model sizes that's ~100GB/run of dead
# weight, and resume only ever needs last.pt (or best.pt).
#   ops/pull_checkpoints.sh <ssh_host> <ssh_port> [ckpt_dir]
# ckpt_dir defaults to data/checkpoints/pretrain; Phase 2 cells pass their own
# out_dir (e.g. data/checkpoints/pretrain_46m_18s). The same relative path is
# used locally and on the instance.
set -euo pipefail
cd "$(dirname "$0")/.."

HOST=${1:?host}; PORT=${2:?port}; CKPT=${3:-data/checkpoints/pretrain}
mkdir -p "$CKPT"
while true; do
  rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    --include='best.pt' --include='last.pt' --include='metrics.jsonl' --exclude='*' \
    "root@$HOST:/workspace/claims-fm/$CKPT/" \
    "$CKPT/" 2>/dev/null || echo "$(date +%T) pull failed (instance busy/down?)"
  echo "$(date +%T) synced $(ls "$CKPT"/*.pt 2>/dev/null | wc -l | tr -d ' ') checkpoints"
  sleep 120
done

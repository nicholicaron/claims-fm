#!/usr/bin/env bash
# Off-instance checkpoint sync (SPEC §10: assume the instance dies).
# Pulls checkpoints + metrics every 2 minutes until interrupted.
#   ops/pull_checkpoints.sh <ssh_host> <ssh_port>
set -euo pipefail

HOST=${1:?host}; PORT=${2:?port}
mkdir -p data/checkpoints/pretrain
while true; do
  rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=no" \
    "root@$HOST:/workspace/claims-fm/data/checkpoints/pretrain/" \
    data/checkpoints/pretrain/ 2>/dev/null || echo "$(date +%T) pull failed (instance busy/down?)"
  echo "$(date +%T) synced $(ls data/checkpoints/pretrain/*.pt 2>/dev/null | wc -l | tr -d ' ') checkpoints"
  sleep 120
done

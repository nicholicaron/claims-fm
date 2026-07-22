#!/usr/bin/env bash
# Find and rent the cheapest reliable RTX 4090 on Vast.
#   ops/provision.sh              -> list candidate offers (cheapest first)
#   ops/provision.sh <offer_id>   -> create instance, wait until running, print ssh cmd
set -euo pipefail

QUERY='gpu_name=RTX_4090 num_gpus=1 reliability>0.98 inet_down>200 disk_space>30 rentable=true'
IMAGE='pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime'

if [[ $# -eq 0 ]]; then
  vastai search offers "$QUERY" -o 'dph' | head -12
  echo; echo "pick an offer id, then: ops/provision.sh <offer_id>"
  exit 0
fi

OFFER_ID=$1
vastai create instance "$OFFER_ID" --image "$IMAGE" --disk 30 --ssh --direct

echo "waiting for instance to start..."
for _ in $(seq 1 60); do
  STATUS=$(vastai show instances --raw | python3 -c 'import json,sys; xs=json.load(sys.stdin); print(xs[-1]["actual_status"] if xs else "none")')
  [[ "$STATUS" == "running" ]] && break
  sleep 10
done

INSTANCE_ID=$(vastai show instances --raw | python3 -c 'import json,sys; print(json.load(sys.stdin)[-1]["id"])')
echo "instance $INSTANCE_ID running"
vastai ssh-url "$INSTANCE_ID"

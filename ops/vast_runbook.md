# Vast.ai pretraining runbook (M3)

Budget: ≤$8 (M3 allocation). Expected: ~2 h @ ~$0.40/h ≈ $1.

## Preflight (local, free)

1. `uv run pytest tests/test_pretrain.py -q` — green, incl. resume determinism.
2. Smoke run done: `scripts/pretokenize.py --config configs/pretrain_smoke.yaml --limit 5000`
   then `scripts/train_mlm.py --config configs/pretrain_smoke.yaml` on MPS/CPU.
3. Full pack built: `scripts/pretokenize.py --config configs/pretrain.yaml`
   (`data/processed/pretrain_pack/`, ~500 MB).
4. `vastai show user` — confirm credit.

## Run

1. `ops/provision.sh` → pick cheapest sane offer → `ops/provision.sh <offer_id>`.
2. `ops/sync_up.sh <host> <port>` (from the printed ssh url).
3. In `tmux` on the instance:
   `cd /workspace/claims-fm && PYTHONPATH=src python scripts/train_mlm.py --config configs/pretrain.yaml`
4. On the Mac, in parallel: `ops/pull_checkpoints.sh <host> <port>` (keeps
   checkpoints + metrics.jsonl synced off-instance every 2 min).
5. **Resume drill** (once, mid-run): kill the training process on the
   instance, then restart with
   `... scripts/train_mlm.py --config configs/pretrain.yaml --resume data/checkpoints/pretrain/last.pt`
   — metrics.jsonl should continue from the same step with no loss spike.
6. Early stop fires on val plateau; final pull; verify `best.pt` is local.

## Teardown & ledger

1. `vastai show instances` → `vastai destroy instance <id>`.
2. `vastai show invoices` → record actual $ in README budget ledger.
3. Never commit anything from `data/checkpoints/` except rendered figures.

## If the instance dies

Checkpoints are already on the Mac. Provision a new instance, `sync_up`, and
resume from the last pulled `last.pt` (upload it back first:
`rsync -e "ssh -p PORT" data/checkpoints/pretrain/last.pt root@HOST:/workspace/claims-fm/data/checkpoints/pretrain/`).
Runs are sized ≤2 h; a full restart costs ≈$1, not a schedule.

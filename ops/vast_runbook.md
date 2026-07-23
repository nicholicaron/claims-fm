# Vast.ai pretraining runbook (M3; Phase 2 addendum at bottom)

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

---

# Phase 2 addendum — scaling cells C1–C3 (prereg: reports/scaling_prereg.md)

Budget: pretraining ≈$3.3 point estimate; whole phase ceiling $10.

| cell | config | pack (sync_up arg) | ckpt dir (pull arg) | est wall | est $ |
|---|---|---|---|---|---|
| C1 17M@18s | `configs/pretrain_17m_18s.yaml` | `data/processed/pretrain_pack_18s` | `data/checkpoints/pretrain_17m_18s` | ~2.5 h | ~$0.62 |
| C2 46M@18s | `configs/pretrain_46m_18s.yaml` | `data/processed/pretrain_pack_18s` | `data/checkpoints/pretrain_46m_18s` | ~8.6 h | ~$2.12 |
| C3 46M@5s | `configs/pretrain_46m_5s.yaml` | `data/processed/pretrain_pack` | `data/checkpoints/pretrain_46m_5s` | ~2.4 h | ~$0.59 |

1. Preflight: both packs built locally (`pretrain_pack` ~483 MB, `pretrain_pack_18s`
   ~1.8 GB), `make test` green, `vastai show user --raw` credit check.
2. Provision one 24 GB 4090 as before (disk 30 GB fits both packs + ~3 GB of
   46M checkpoints; a 46M `best.pt` is ~550 MB vs 200 MB at 17M).
3. `ops/sync_up.sh <host> <port> data/processed/pretrain_pack data/processed/pretrain_pack_18s`
   (rsync --partial; the 1.8 GB pack survives flaky-link retries).
4. Run cells **sequentially** in tmux; restart the pull watcher per cell:
   `ops/pull_checkpoints.sh <host> <port> data/checkpoints/pretrain_17m_18s` etc.
5. C2/C3 OOM fallback: `train.tokens_per_batch` 16384 → 12288 → 8192 (see
   config comments); nothing else changes.
6. Teardown as above; record per-cell $ in the ledger.

**⚠️ `pretrain_pack` is a frozen artifact.** With samples 8–20 now marked
`pretrain` in `configs/data.yaml`, re-running `make sequences` + pretokenize
against `configs/pretrain.yaml` (or `pretrain_46m_5s.yaml`) would silently
rebuild the 5-sample pack from the 18-sample corpus. Never re-pretokenize into
`data/processed/pretrain_pack`; tripwire: its `meta.json` must keep
`n_members: 517390`. Pack meta gains a `source_samples` field with the Phase 2
pretokenize changes.

Hier grids: the default `scripts/finetune_task_b_hier.py` run covers full +
probe + the full-mode LE grid; the REGISTERED hier-scratch LE arm needs the
separate `--scratch-le` invocation — don't forget it, or the report renders
"—" cells for scratch.

Downstream fine-tunes (Task A/B, hier grids) also run on the instance; sync the
task packs separately when that phase starts:
`rsync -avz --partial -e "ssh -p PORT ..." data/processed/task_a_pack data/processed/task_b_pack data/processed/task_b_pack_chunked data/processed/task_?_meta.json root@HOST:/workspace/claims-fm/data/processed/`
(46M fine-tune configs land with that phase.)

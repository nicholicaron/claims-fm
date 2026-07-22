CONFIG := configs/data.yaml

.PHONY: env download tables sequences vocab overlap eda test m1

env:
	uv sync --all-groups

download:
	uv run python scripts/download.py --config $(CONFIG)

tables:
	uv run python scripts/build_tables.py --config $(CONFIG)

sequences:
	uv run python scripts/build_sequences.py --config $(CONFIG)
	uv run python scripts/build_sequences.py --config $(CONFIG) --window 2008:2009 --roles eval_only

vocab:
	uv run python scripts/build_vocab.py --config $(CONFIG)

overlap:
	uv run python scripts/vocab_overlap.py --config $(CONFIG)

eda:
	uv run jupyter nbconvert --to notebook --execute --inplace notebooks/eda_m0.ipynb

test:
	uv run pytest -q

m1: tables sequences vocab overlap test

task-a:
	uv run python scripts/build_task_a.py --config configs/baselines.yaml

task-b:
	uv run python scripts/build_task_b.py --config configs/baselines.yaml

baselines:
	uv run python scripts/run_baselines.py --config configs/baselines.yaml

baselines-final:
	uv run python scripts/run_baselines.py --config configs/baselines.yaml --final-eval

m2: task-a task-b baselines baselines-final test

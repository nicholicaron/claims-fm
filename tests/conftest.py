import pytest

from claimsfm.config import data_path, load_config, load_lock


@pytest.fixture(scope="session")
def cfg():
    return load_config("configs/data.yaml")


@pytest.fixture(scope="session")
def lock():
    lock = load_lock()
    if not lock.get("synpuf"):
        pytest.skip("no data downloaded yet (configs/data.lock.yaml empty)")
    return lock


@pytest.fixture(scope="session")
def processed(cfg):
    p = data_path(cfg, "processed")
    if not (p / "sequences_pretrain.parquet").exists():
        pytest.skip("sequences not built yet (make sequences)")
    return p

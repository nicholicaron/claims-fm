import datetime as dt
import json

import pytest

from claimsfm.tokenizer import ClaimsTokenizer
from claimsfm.vocab import SPECIALS


@pytest.fixture()
def tok():
    vocab = {t: i for i, t in enumerate(SPECIALS)}
    for code in ["DX_25000", "DX_4019", "PX_9904", "RX_00003089321"]:
        vocab[code] = len(vocab)
    return ClaimsTokenizer({"tokens": vocab, "meta": {"min_count": 1}})


def _member():
    return dict(
        tokens=["DX_25000", "PX_9904", "DX_4019", "RX_00003089321", "DX_UNSEEN"],
        dates=[
            dt.date(2008, 3, 1),
            dt.date(2008, 3, 1),
            dt.date(2009, 6, 2),
            dt.date(2009, 6, 2),
            dt.date(2010, 1, 15),
        ],
        claim_types=["IP", "IP", "OP", "RX", "OP"],
        visit_ids=[1, 1, 2, 3, 4],
    )


def test_encode_structure(tok):
    enc = tok.encode(**_member())
    decoded = tok.decode(enc["input_ids"])
    assert decoded[0] == "[CLS]"
    assert decoded.count("[VISIT]") == 3  # 4 visits -> 3 separators
    assert "[UNK]" in decoded  # DX_UNSEEN mapped to UNK
    assert len(enc["input_ids"]) == len(enc["visit_ids"]) == len(enc["claim_type_ids"])


def test_encode_window_excludes_out_of_window_events(tok):
    enc = tok.encode(**_member(), window=(2008, 2009))
    decoded = tok.decode(enc["input_ids"])
    assert "[UNK]" not in decoded  # the only OOV token is the 2010 event
    assert decoded.count("[VISIT]") == 2  # visit 4 dropped


def test_encode_max_len_keeps_most_recent(tok):
    enc = tok.encode(**_member(), max_len=4)
    assert len(enc["input_ids"]) == 4
    assert enc["input_ids"][0] == tok.cls_id
    decoded = tok.decode(enc["input_ids"])
    assert "[UNK]" in decoded  # most recent (2010) event retained


def test_roundtrip_known_tokens(tok):
    ids = tok.encode(**_member())["input_ids"]
    assert tok.decode(ids)[1] == "DX_25000"


def test_real_vocab_roundtrip_if_built(cfg, processed):
    path = processed / "vocab.json"
    tok = ClaimsTokenizer.load(path)
    with open(path) as f:
        meta = json.load(f)["meta"]
    assert len(tok) == meta["vocab_size"]
    assert tok.pad_id == 0
    sample = [t for t in list(tok.tokens)[5:10]]
    ids = [tok.tokens[t] for t in sample]
    assert tok.decode(ids) == sample

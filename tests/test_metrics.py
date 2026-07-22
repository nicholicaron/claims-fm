"""Metric implementations against hand-computed toy vectors."""

import numpy as np
import pytest

from claimsfm.eval.metrics import capture_at, ece, precision_at_k, recall_at_k


Y = np.array([1, 0, 1, 0, 0, 0, 1, 0, 0, 0])
P = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])


def test_precision_at_k():
    assert precision_at_k(Y, P, 1) == 1.0          # top-1 is a positive
    assert precision_at_k(Y, P, 2) == 0.5          # {1, 0}
    assert precision_at_k(Y, P, 4) == 0.5          # {1, 0, 1, 0}
    assert precision_at_k(Y, P, 100) == pytest.approx(0.3)  # k > n clamps


def test_recall_at_k():
    assert recall_at_k(Y, P, 1) == pytest.approx(1 / 3)
    assert recall_at_k(Y, P, 4) == pytest.approx(2 / 3)
    assert recall_at_k(Y, P, 10) == 1.0


def test_capture_at():
    # top 20% = top 2 = one of three positives
    assert capture_at(Y, P, 0.2) == pytest.approx(1 / 3)
    assert capture_at(Y, P, 1.0) == 1.0


def test_capture_handles_no_positives():
    assert np.isnan(capture_at(np.zeros(5), np.linspace(0, 1, 5), 0.2))


def test_ece_perfectly_calibrated_bins():
    # two bins: p=0.25 with 25% positives, p=0.75 with 75% positives
    y = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    p = np.array([0.25, 0.25, 0.25, 0.25, 0.75, 0.75, 0.75, 0.75])
    assert ece(y, p, n_bins=2) == pytest.approx(0.0)


def test_ece_worst_case():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.95, 0.95, 0.05, 0.05])
    assert ece(y, p, n_bins=2) == pytest.approx(0.95 * 0.5 + 0.95 * 0.5, abs=0.01)

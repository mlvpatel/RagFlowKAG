"""Unit tests for the retrieval evaluation metrics."""

from eval.metrics import (
    f1_score,
    hit_at_k,
    mean,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_hit_at_k():
    assert hit_at_k([3, 1, 2], {1}, 3) == 1.0
    assert hit_at_k([3, 4, 5], {1}, 3) == 0.0
    assert hit_at_k([1, 2], {1}, 1) == 1.0
    assert hit_at_k([2, 1], {1}, 1) == 0.0


def test_precision_at_k():
    assert precision_at_k([1, 2, 3], {1, 2}, 3) == 2 / 3
    assert precision_at_k([], {1}, 3) == 0.0


def test_recall_at_k():
    assert recall_at_k([1, 3], {1, 2}, 2) == 0.5
    assert recall_at_k([1, 2], {1, 2}, 5) == 1.0
    assert recall_at_k([3], set(), 3) == 0.0


def test_f1_score():
    assert f1_score(0.5, 0.5) == 0.5
    assert f1_score(0.0, 0.0) == 0.0
    assert abs(f1_score(1.0, 0.5) - 2 / 3) < 1e-9


def test_reciprocal_rank():
    assert reciprocal_rank([3, 1, 2], {1}) == 0.5
    assert reciprocal_rank([1, 2], {1}) == 1.0
    assert reciprocal_rank([2, 3], {1}) == 0.0


def test_mean():
    assert mean([1.0, 2.0, 3.0]) == 2.0
    assert mean([]) == 0.0

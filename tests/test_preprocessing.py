"""Tests for the preprocessing pipeline (leakage-safe behaviour)."""

import numpy as np
import pytest

from src.data_loader import RAW_DATA_DIR, load_student_data
from src.preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_preprocessor,
    make_train_test_split,
    prepare_datasets,
)

pytestmark = pytest.mark.skipif(
    not (RAW_DATA_DIR / "student-mat.csv").exists(),
    reason="raw dataset not downloaded (see README, Phase 2)",
)


@pytest.fixture(scope="module")
def math_split():
    return prepare_datasets("math")


class TestSplitting:
    def test_shapes(self, math_split):
        X_train, X_test, y_train, y_test, cleaned, _ = math_split
        assert X_train.shape[0] == int(0.8 * len(cleaned))
        assert X_test.shape[0] == len(cleaned) - X_train.shape[0]
        assert len(y_train) == X_train.shape[0]
        assert len(y_test) == X_test.shape[0]

    def test_features_match_target_lengths(self, math_split):
        X_train, X_test, y_train, y_test, _, _ = math_split
        assert set(X_train.columns) == set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)

    def test_reproducible_with_same_random_state(self):
        first = make_train_test_split(load_student_data("math"))
        second = make_train_test_split(load_student_data("math"))
        # Same seed -> identical splits, row for row.
        assert first[0].index.equals(second[0].index)


def _to_dense(matrix):
    """Convert a sparse matrix to a dense array; pass arrays through."""
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


class TestPreprocessor:
    def test_fit_transform_shapes_consistent(self, math_split):
        X_train, X_test, *_ = math_split
        pre = build_preprocessor()
        X_train_t = pre.fit_transform(X_train)
        X_test_t = pre.transform(X_test)
        assert X_train_t.shape[1] == X_test_t.shape[1]

    def test_no_nan_after_transform(self, math_split):
        X_train, X_test, *_ = math_split
        pre = build_preprocessor()
        X_train_t = pre.fit_transform(X_train)
        X_test_t = pre.transform(X_test)
        assert not np.isnan(_to_dense(X_train_t)).any()
        assert not np.isnan(_to_dense(X_test_t)).any()

    def test_numeric_features_are_scaled(self, math_split):
        """After StandardScaler the numeric block has ~0 mean, ~1 std."""
        X_train, *_ = math_split
        pre = build_preprocessor()
        X_train_t = pre.fit_transform(X_train)
        n_num = len(NUMERIC_FEATURES)
        num_block = _to_dense(X_train_t)[:, :n_num]
        assert np.allclose(num_block.mean(axis=0), 0, atol=1e-7)
        assert np.allclose(num_block.std(axis=0), 1, atol=1e-7)

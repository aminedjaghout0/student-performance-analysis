"""Basic tests for the data loader.

These tests require the raw dataset in data/raw/ (it is not committed to
Git). If the file is missing, the tests are skipped rather than failed,
so the test suite still passes on a fresh clone.
"""

from pathlib import Path

import pandas as pd
import pytest

from src import data_loader

RAW_CSV = data_loader.RAW_DATA_DIR / "student-mat.csv"

pytestmark = pytest.mark.skipif(
    not RAW_CSV.exists(),
    reason="raw dataset not downloaded (see README, Phase 2)",
)


def test_load_math_dataset_shape():
    """The Math file must contain 395 students x 33 columns."""
    df = data_loader.load_student_data("math")
    assert df.shape == (395, 33)


def test_columns_match_uci_schema():
    df = data_loader.load_student_data("math")
    assert list(df.columns) == data_loader.EXPECTED_COLUMNS


def test_target_column_values_in_valid_range():
    """Grades G1, G2, G3 are integers from 0 to 20 in the UCI dataset."""
    df = data_loader.load_student_data("math")
    for col in ("G1", "G2", "G3"):
        assert pd.api.types.is_numeric_dtype(df[col])
        assert df[col].between(0, 20).all()


def test_unknown_course_raises():
    with pytest.raises(ValueError):
        data_loader.load_student_data("chemistry")


def test_string_values_have_no_leftover_quotes_or_whitespace():
    df = data_loader.load_student_data("math")
    for col in df.select_dtypes(include="object").columns:
        assert not df[col].str.startswith('"').any()
        # No value should differ from its own stripped version.
        assert (df[col] == df[col].str.strip()).all()

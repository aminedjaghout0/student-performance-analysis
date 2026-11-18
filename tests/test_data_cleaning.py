"""Tests for the data cleaning module."""

import pytest

from src.data_cleaning import (
    check_value_ranges,
    clean_student_data,
    count_non_participants,
    strip_string_whitespace,
)
from src.data_loader import RAW_DATA_DIR, load_student_data

pytestmark = pytest.mark.skipif(
    not (RAW_DATA_DIR / "student-mat.csv").exists(),
    reason="raw dataset not downloaded (see README, Phase 2)",
)


class TestCleaning:
    def test_clean_math_dataset_report(self):
        raw = load_student_data("math")
        cleaned, report = clean_student_data(raw, drop_non_participants=True)
        # 395 raw rows; 38 documented non-participants with G3 == 0.
        assert report["non_participants_found"] == 38
        assert report["rows_after_cleaning"] == 395 - 38 == len(cleaned)
        assert report["range_violations"] == []

    def test_non_participants_can_be_kept(self):
        raw = load_student_data("math")
        cleaned, report = clean_student_data(raw, drop_non_participants=False)
        assert len(cleaned) == 395
        assert report["non_participants_dropped"] == 0

    def test_no_missing_values_after_cleaning(self):
        raw = load_student_data("math")
        cleaned, _ = clean_student_data(raw)
        assert cleaned.isna().sum().sum() == 0

    def test_strip_whitespace(self):
        raw = load_student_data("math")
        stripped = strip_string_whitespace(raw)
        for col in stripped.select_dtypes(include="object").columns:
            assert (stripped[col] == stripped[col].str.strip()).all()

    def test_range_check_detects_bad_values(self):
        import pandas as pd

        bad = pd.DataFrame({"age": [14, 20, 23]})
        violations = check_value_ranges(bad)
        assert any("age" in v for v in violations)

    def test_count_non_participants(self):
        raw = load_student_data("portuguese")
        assert count_non_participants(raw) == 15

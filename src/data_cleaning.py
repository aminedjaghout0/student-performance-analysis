"""Data cleaning for the UCI Student Performance dataset.

Cleaning philosophy
-------------------
Only decisions that are *justified by the dataset's documented structure*
are applied here, before any train/test split. Statistical fitting
(imputation medians, scaling, encoding) is deliberately NOT done in this
module: it belongs to the scikit-learn preprocessing pipeline and must be
fitted on training data only to avoid data leakage.
"""

import pandas as pd

from src.data_loader import TARGET_COLUMN

# Documented valid ranges from the UCI student.txt file.
VALID_RANGES = {
    "age": (15, 22),
    "Medu": (0, 4),        # mother's education (0 = none ... 4 = higher ed)
    "Fedu": (0, 4),        # father's education
    "traveltime": (1, 4),  # 1 = <15 min ... 4 = >1 h
    "studytime": (1, 4),   # 1 = <2 h ... 4 = >10 h weekly
    "failures": (0, 4),    # number of past class failures
    "famrel": (1, 5),      # family relationship quality
    "freetime": (1, 5),
    "goout": (1, 5),
    "Dalc": (1, 5),        # workday alcohol consumption
    "Walc": (1, 5),        # weekend alcohol consumption
    "health": (1, 5),
    "absences": (0, 93),   # observed max is 75; small upper headroom
    "G1": (0, 20),
    "G2": (0, 20),
    "G3": (0, 20),
}


def strip_string_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    """Trim surrounding whitespace from every string cell.

    Necessary because silent trailing spaces in category labels
    (e.g. "yes " vs "yes") would later split one category into two
    during one-hot encoding.
    """
    cleaned = df.copy()
    for col in cleaned.select_dtypes(include="object").columns:
        cleaned[col] = cleaned[col].str.strip()
    return cleaned


def check_value_ranges(df: pd.DataFrame) -> list[str]:
    """Return a list of human-readable range violations (empty = clean)."""
    violations = []
    for col, (low, high) in VALID_RANGES.items():
        if col not in df.columns:  # defensive: subset frames may not have all cols
            continue
        out_of_range = ~df[col].between(low, high)
        if out_of_range.any():
            bad = df.loc[out_of_range, col].unique()
            violations.append(f"{col}: values {sorted(bad)} outside [{low}, {high}]")
    return violations


def count_non_participants(df: pd.DataFrame) -> int:
    """Count students with G3 == 0.

    In this dataset G3 = 0 does NOT mean a student scored zero on a
    20-point exam: per the dataset documentation and the source paper,
    those students did not sit the final exam (absence/withdrawal).
    Treating them as genuine grades of 0 would bias the model towards
    under-prediction, so we treat them as missing target values.
    """
    return int((df[TARGET_COLUMN] == 0).sum())


def clean_student_data(
    df: pd.DataFrame,
    drop_non_participants: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Apply all documented cleaning decisions and report what was done.

    Args:
        df: Raw DataFrame from src.data_loader.load_student_data.
        drop_non_participants: If True, drop rows with G3 == 0 (students
            who did not sit the final exam). Recommended for regression,
            because a missing exam is not a measured grade of 0.

    Returns:
        (cleaned_df, report) where report is a dictionary describing
        every action taken, suitable for logging or display.
    """
    report: dict = {}

    # 1. Whitespace in string columns (defensive; inspection found none,
    #    but the rule costs nothing and protects the encoding step).
    df = strip_string_whitespace(df)
    report["whitespace_trimmed"] = True

    # 2. Missing values: inspection showed none, but we verify again and
    #    would drop rows only if they existed (none to impute here).
    missing_before = int(df.isna().sum().sum())
    if missing_before:
        df = df.dropna()
    report["missing_values_found"] = missing_before
    report["missing_rows_dropped"] = missing_before  # rows affected

    # 3. Duplicates: inspection showed none; the check stays in the
    #    pipeline so it runs on any future re-download.
    duplicates = int(df.duplicated().sum())
    if duplicates:
        df = df.drop_duplicates()
    report["duplicates_found"] = duplicates
    report["duplicates_dropped"] = duplicates

    # 4. Value ranges against the documented schema.
    report["range_violations"] = check_value_ranges(df)

    # 5. Non-participants (G3 == 0): documented as "did not sit the exam".
    report["non_participants_found"] = count_non_participants(df)
    if drop_non_participants and report["non_participants_found"]:
        df = df[df[TARGET_COLUMN] != 0].copy()
    report["non_participants_dropped"] = (
        report["non_participants_found"] if drop_non_participants else 0
    )
    report["drop_non_participants_enabled"] = drop_non_participants

    report["rows_after_cleaning"] = len(df)
    return df, report

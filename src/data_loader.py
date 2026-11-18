"""Load the UCI Student Performance dataset.

Source: Cortez, P., & Silva, A. (2008). "Using Data Mining to Predict
Secondary School Student Performance". UCI Machine Learning Repository.
https://archive.ics.uci.edu/dataset/320/student+performance

The dataset comes as two CSV files (semicolon-separated, quoted strings):
  - student-mat.csv : 395 students, Mathematics course
  - student-por.csv : 649 students, Portuguese course
"""

from pathlib import Path

import pandas as pd

# Project root (repository folder), independent of the current working
# directory, so the code runs from anywhere on any machine.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Files shipped with the UCI dataset, mapped to a short course name.
COURSE_FILES = {
    "math": "student-mat.csv",
    "portuguese": "student-por.csv",
}

# The 33 expected columns, in order. Used to validate that the raw file
# is intact and correctly parsed.
EXPECTED_COLUMNS = [
    "school", "sex", "age", "address", "famsize", "Pstatus",
    "Medu", "Fedu", "Mjob", "Fjob", "reason", "guardian",
    "traveltime", "studytime", "failures",
    "schoolsup", "famsup", "paid", "activities", "nursery", "higher",
    "internet", "romantic",
    "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences",
    "G1", "G2", "G3",
]

TARGET_COLUMN = "G3"


def load_student_data(course: str = "math") -> pd.DataFrame:
    """Load one UCI Student Performance CSV file into a DataFrame.

    Args:
        course: "math" for student-mat.csv or "portuguese" for
            student-por.csv.

    Returns:
        A pandas DataFrame with 33 validated columns.

    Raises:
        FileNotFoundError: If the raw CSV file is missing (the dataset
            is not committed to Git and must be downloaded first).
        ValueError: If the file was parsed with unexpected columns.
    """
    if course not in COURSE_FILES:
        raise ValueError(
            f"Unknown course '{course}'. Choose from: {sorted(COURSE_FILES)}"
        )

    csv_path = RAW_DATA_DIR / COURSE_FILES[course]
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {csv_path}\n"
            "Download the UCI Student Performance dataset from "
            "https://archive.ics.uci.edu/dataset/320/student+performance "
            "and extract it into data/raw/ (see README)."
        )

    # The UCI files use ';' as the separator and double quotes around
    # string values. Standard CSV quoting handles them correctly.
    df = pd.read_csv(csv_path, sep=";", encoding="latin-1")

    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            f"Parsed columns differ from the documented UCI schema. "
            f"Got {list(df.columns)}, expected {EXPECTED_COLUMNS}."
        )

    return df


def save_processed_data(df: pd.DataFrame, filename: str = "student_math_cleaned.csv") -> Path:
    """Save a cleaned DataFrame to data/processed/ as CSV."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DATA_DIR / filename
    df.to_csv(out_path, index=False)
    return out_path

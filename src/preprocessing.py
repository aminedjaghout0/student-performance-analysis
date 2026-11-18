"""Preprocessing pipeline and train/test splitting.

Key principle (avoiding data leakage)
-------------------------------------
Everything that *learns from data* — imputation medians, scaling
statistics, one-hot categories — is fitted on the TRAINING set only.
The fitted pipeline is then applied, unchanged, to the test set.
Fitting on the full dataset would let test-set information influence
preprocessing and produce optimistically biased evaluation scores.

Why G1 and G2 are kept as features
----------------------------------
G1 (first period) and G2 (second period) grades are known *before* the
final exam takes place, so using them to predict G3 is a legitimate
prediction task ("predict the final grade from the two previous
periods"), not data leakage. We note, however, that they are by far the
strongest predictors — an important point to discuss when interpreting
feature importance in Phase 7.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_cleaning import clean_student_data
from src.data_loader import TARGET_COLUMN, load_student_data

RANDOM_STATE = 42  # fixed seed so every experiment is reproducible
TEST_SIZE = 0.2    # 80% train / 20% test — a common default

# 16 numerical features (all except the target G3).
NUMERIC_FEATURES = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures",
    "famrel", "freetime", "goout", "Dalc", "Walc", "health",
    "absences", "G1", "G2",
]

# 17 categorical features (no ordering implied -> one-hot encoding).
CATEGORICAL_FEATURES = [
    "school", "sex", "address", "famsize", "Pstatus",
    "Mjob", "Fjob", "reason", "guardian",
    "schoolsup", "famsup", "paid", "activities", "nursery", "higher",
    "internet", "romantic",
]


def build_preprocessor() -> ColumnTransformer:
    """Create the (not yet fitted) preprocessing pipeline.

    Numeric branch:
        SimpleImputer(median) -> StandardScaler
        The median is robust to outliers (e.g. absences up to 75).
        Scaling puts all numeric features on comparable ranges, which
        matters for linear models.

    Categorical branch:
        SimpleImputer(most_frequent) -> OneHotEncoder(handle_unknown="ignore")
        One-hot encoding turns each category into its own 0/1 column;
        handle_unknown="ignore" keeps the pipeline working if an unseen
        category appears at prediction time.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def make_train_test_split(df: pd.DataFrame):
    """Split features/target into train and test sets.

    Returns:
        (X_train, X_test, y_train, y_test)
    """
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COLUMN]
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )


def prepare_datasets(course: str = "math", drop_non_participants: bool = True):
    """End-to-end: load -> clean -> split. Returns split sets + report.

    The preprocessor is returned UNFITTED on purpose; the caller fits
    it on X_train only (see train_model.py in Phase 6).
    """
    raw = load_student_data(course)
    cleaned, report = clean_student_data(raw, drop_non_participants)
    X_train, X_test, y_train, y_test = make_train_test_split(cleaned)
    return X_train, X_test, y_train, y_test, cleaned, report


if __name__ == "__main__":
    # Quick manual check:  python -m src.preprocessing
    from src.data_loader import save_processed_data

    X_train, X_test, y_train, y_test, cleaned, report = prepare_datasets("math")
    out = save_processed_data(cleaned, "student_math_cleaned.csv")
    print(f"Cleaned dataset saved to: {out}")
    print("Cleaning report:")
    for key, value in report.items():
        print(f"  {key}: {value}")
    print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

    preprocessor = build_preprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train)  # fit on TRAIN only
    X_test_transformed = preprocessor.transform(X_test)        # transform TEST only
    print(f"After preprocessing: train {X_train_transformed.shape}, "
          f"test {X_test_transformed.shape}")
    n_onehot = len(preprocessor.named_transformers_["cat"]
                    .named_steps["onehot"].get_feature_names_out())
    print(f"One-hot columns created from categoricals: {n_onehot}")

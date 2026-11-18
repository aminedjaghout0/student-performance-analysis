# Student Performance Analysis & Prediction

> ⚠️ **Work in progress** — this repository is being built phase by phase.
> The full documentation (dataset description, methodology, results, and
> visualisations) will be completed as each phase of the project is finished.

## Dataset

The project uses the [UCI Student Performance dataset](https://archive.ics.uci.edu/dataset/320/student+performance)
(Cortez & Silva, 2008), which records academic and demographic information for
secondary-school students in Portugal:

- `student-mat.csv` — 395 students, Mathematics course
- `student-por.csv` — 649 students, Portuguese course

Each record has **33 columns** (16 numerical, 17 categorical) and three grade
variables: `G1` (first period), `G2` (second period) and `G3` (final grade,
0–20, the prediction target). The dataset contains **no missing values and no
duplicate rows**, and all values fall within the documented ranges
(e.g. age 15–22, absences 0–75). The raw files are kept unmodified in
`data/raw/`.

## Data-cleaning methodology

Cleaning decisions are documented and reproducible (`src/data_cleaning.py`):

1. **Whitespace** in string categories is trimmed (prevents a category
   splitting in two during one-hot encoding).
2. **Missing values** — verified: the UCI dataset contains none.
3. **Duplicates** — verified: no duplicated rows.
4. **Range validation** — every value is checked against the ranges
   documented in the UCI `student.txt` file (age 15–22, grades 0–20,
   ordinal scales 1–5, …).
5. **Non-participants** — 38 Math students have `G3 = 0`. The dataset
   documentation shows these students **did not sit the final exam**
   (absence/withdrawal), so a 0 is a *missing* grade, not a measured one.
   They are dropped for regression modelling; the decision and count are
   reported by the pipeline instead of being silently applied.

Result: **357 clean Math records** (395 − 38).

## Preprocessing (leakage-safe)

`src/preprocessing.py` builds a scikit-learn `ColumnTransformer` fitted
**on the training set only**:

- **Numeric features** → median imputation + `StandardScaler`
  (median is robust to outliers such as `absences = 75`).
- **Categorical features** → most-frequent imputation + one-hot encoding
  with `handle_unknown="ignore"`.
- **Split**: 80 % train / 20 % test, `random_state=42` for full
  reproducibility → 285 training / 72 test records.
- After encoding: **58 model features** (15 numeric + 43 one-hot).

`G1` and `G2` are kept as features by design: both are known before the
final exam, so predicting `G3` from them is a legitimate forecasting
scenario, not target leakage.

## Planned objectives

- Analyse students' academic performance using the
  [UCI Student Performance dataset](https://archive.ics.uci.edu/dataset/320/student+performance).
- Explore patterns and relationships between academic factors.
- Train and compare machine-learning models (Linear Regression,
  Decision Tree, Random Forest) to predict final grades.
- Evaluate models with MAE, MSE, RMSE and R², and analyse feature importance.
- Present results through professional visualisations and, later,
  an interactive Streamlit dashboard.

## Current status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Environment setup & repository structure | ✅ In progress |
| 2 | Dataset acquisition & inspection | ✅ Complete |
| 3 | Data cleaning & preprocessing | ✅ Complete |
| 4 | Exploratory Data Analysis | ⬜ Not started |
| 5 | Data visualisation | ⬜ Not started |
| 6 | Machine-learning model development | ⬜ Not started |
| 7 | Model evaluation & feature importance | ⬜ Not started |
| 8 | Interactive Streamlit dashboard | ⬜ Not started |
| 9 | Testing, debugging & documentation | ⬜ Not started |
| 10 | Final GitHub presentation | ⬜ Not started |

## Repository structure

```text
student-performance-analysis/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/                  # Original, unchanged dataset (not committed)
│   └── processed/            # Cleaned dataset (not committed)
│
├── notebooks/
│   └── student_performance_analysis.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── data_cleaning.py
│   ├── exploratory_analysis.py
│   ├── visualization.py
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── predict.py
│
├── models/                   # Trained model artifacts (not committed)
│
├── results/
│   ├── visualizations/       # Exported charts (not committed)
│   └── model_metrics/        # Metric tables (not committed)
│
├── tests/                    # Basic unit tests
│
└── app/
    └── dashboard.py          # Streamlit dashboard (Phase 8)
```

## Installation (quick start)

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/student-performance-analysis.git
cd student-performance-analysis

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Author

**Your Name** — Computer Science student building a portfolio in
Data Science and Machine Learning.
<!-- TODO: add GitHub profile link and contact information. -->

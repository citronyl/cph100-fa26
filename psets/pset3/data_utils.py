"""Load the diabetes encounter dataset and create reproducible partitions."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit

LABEL = "readmitted"
EXCLUDE_FROM_FEATURES = {LABEL, "encounter_id", "patient_nbr"}


def load_data(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, na_values=["?", "Unknown/Invalid"])
    required = {LABEL, "age", "race", "gender", "patient_nbr"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")
    frame = frame.copy()
    frame["target"] = (frame[LABEL] == "<30").astype(int)
    return frame


def age_midpoint(value) -> float:
    """Convert dataset bins such as '[40-50)' to approximate age in years."""
    if pd.isna(value):
        return float("nan")
    import re
    numbers = re.findall(r"\d+", str(value))
    return (int(numbers[0]) + int(numbers[1])) / 2 if len(numbers) == 2 else float("nan")


def add_age_group(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    midpoints = frame["age"].map(age_midpoint)
    frame["age_group"] = np.where(midpoints < 50, "under_50", "50_plus")
    frame.loc[midpoints.isna(), "age_group"] = "unknown"
    return frame


def split_data(frame, seed=42, patient_disjoint=False):
    """Return train/val/test frames (60/20/20).

    Encounter-level is stratified; patient-disjoint prevents patient overlap but
    does not guarantee outcome stratification. Never tune on the test set.
    """
    if not patient_disjoint:
        train, remaining = train_test_split(frame, test_size=.4, random_state=seed, stratify=frame["target"])
        val, test = train_test_split(remaining, test_size=.5, random_state=seed, stratify=remaining["target"])
    else:
        first = GroupShuffleSplit(n_splits=1, test_size=.4, random_state=seed)
        train_idx, remaining_idx = next(first.split(frame, groups=frame["patient_nbr"]))
        train, remaining = frame.iloc[train_idx], frame.iloc[remaining_idx]
        second = GroupShuffleSplit(n_splits=1, test_size=.5, random_state=seed + 1)
        val_idx, test_idx = next(second.split(remaining, groups=remaining["patient_nbr"]))
        val, test = remaining.iloc[val_idx], remaining.iloc[test_idx]
    return train.copy(), val.copy(), test.copy()


def model_features(frame):
    """Return features and outcome, removing identifiers and derived labels."""
    X = frame.drop(columns=list(EXCLUDE_FROM_FEATURES | {"target", "age_group"}), errors="ignore")
    return X, frame["target"].astype(int)


def make_llm_sample(test, n=200, seed=42):
    """Reproducible sample stratified by age group and outcome when feasible."""
    if n > len(test):
        raise ValueError("Sample size exceeds test set")
    if n < 1:
        raise ValueError("Sample size must be positive")
    frame = add_age_group(test)
    strata = frame["age_group"].astype(str) + "_" + frame["target"].astype(str)
    if strata.value_counts().min() < 2 or n < strata.nunique():
        return frame.sample(n=n, random_state=seed).copy()
    from sklearn.model_selection import train_test_split
    _, sample = train_test_split(frame, test_size=n, stratify=strata, random_state=seed)
    return sample.sort_index().copy()

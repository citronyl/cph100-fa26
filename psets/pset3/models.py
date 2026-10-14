"""Part 1.2: preprocessing and three prediction models.

You may use scikit-learn. Do not fit transformations on validation/test data.
"""


def build_preprocessor(X_train):
    """Return an unfitted preprocessing transformer for diabetes features.

    Numeric columns: median imputation. Categorical columns: most-frequent
    imputation followed by one-hot encoding with unknown categories ignored.
    The returned transformer must be fitted ONLY on the training partition.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    # The caller supplies feature columns, not labels or patient identifiers.
    excluded = {"readmitted", "target", "encounter_id", "patient_nbr", "age_group"}
    feature_names = [c for c in X_train.columns if c not in excluded]
    numeric = X_train[feature_names].select_dtypes(include="number").columns.tolist()
    categorical = [c for c in feature_names if c not in numeric]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(
        [("numeric", numeric_pipe, numeric),
         ("categorical", categorical_pipe, categorical)],
        remainder="drop",
    )


def build_model(kind, X_train, params=None, seed=42):
    """Return an UNFITTED model pipeline.

    kind is one of 'linear', 'tree', 'neural'.

    TODO:
      - Choose and instantiate one model from each family.
      - Apply the same train-only preprocessing appropriately.
      - Expose tunable hyperparameters via params.
      - Set random seeds where supported.
    """
    if kind not in {'linear', 'tree', 'neural'}:
        raise ValueError(f'Unknown model family: {kind}')
    raise NotImplementedError(f'Student TODO: implement {kind} model')


def tune_model(kind, train, val, seed=42):
    """Select hyperparameters using train and validation partitions only.

    TODO: Train candidate models, compute validation AUC, select a candidate,
    and record the search and results. Never select using the test set.
    """
    raise NotImplementedError("Student TODO: hyperparameter search")

"""Part 1.2: preprocessing and three prediction models.

You may use scikit-learn. Do not fit transformations on validation/test data.
"""
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import ParameterGrid
from data_utils import model_features
from evaluation import compute_auc

CODE_COLUMNS = ["admission_type_id", "discharge_disposition_id", "admission_source_id"]

def build_preprocessor(X_train):
    """Return an unfitted preprocessing transformer for diabetes features.

    Numeric columns: median imputation. Categorical columns: constant
    imputation followed by one-hot encoding with unknown categories ignored.
    The returned transformer must be fitted ONLY on the training partition.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    # The caller supplies feature columns, not labels or patient identifiers.
    excluded = {"readmitted", "target", "encounter_id", "patient_nbr", "age_group", "weight"}
    feature_names = [c for c in X_train.columns if c not in excluded]
    #numeric = X_train[feature_names].select_dtypes(include="number").columns.tolist()
    numeric = [c for c in X_train[feature_names].select_dtypes(include="number").columns if c not in CODE_COLUMNS]
    categorical = [c for c in feature_names if c not in numeric]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        #("imputer", SimpleImputer(strategy="most_frequent")),
        #("onehot", OneHotEncoder(handle_unknown="ignore")),
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50)),
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
    params = params or {}
    if kind == "linear":
        classifier = LogisticRegression(random_state=seed, max_iter = 2000, **params)
    if kind == "tree":
        classifier = RandomForestClassifier(random_state=seed, n_jobs=-1, n_estimators = 300, **params)
    if kind == "neural":
        classifier = MLPClassifier(random_state=seed, early_stopping=True, **params)
    if kind not in {'linear', 'tree', 'neural'}:
        raise ValueError(f'Unknown model family: {kind}')
    return Pipeline([("prep", build_preprocessor(X_train)), ("model", classifier)])


def tune_model(kind, train, val, seed=42):
    """Select hyperparameters using train and validation partitions only.

    TODO: Train candidate models, compute validation AUC, select a candidate,
    and record the search and results. Never select using the test set.
    """
    param_grid = {
        "linear": {"C": [0.0001, 0.001, 0.003, 0.01, 1.0, 10.0]},
        "tree": {"max_depth": [10, 5, 1, None], "min_samples_leaf": [1, 5, 10, 50]},
        "neural": {"hidden_layer_sizes": [(64,), (128, 64)], "alpha": [1e-5, 1e-2, 1]},
    }

    X_train, y_train = model_features(train)
    X_val, y_val = model_features(val)

    results = []
    best_auc, best_model, best_params = -1, None, None
    for params in ParameterGrid(param_grid[kind]):
        model = build_model(kind, X_train, params=params, seed=seed)
        model.fit(X_train, y_train)
        auc = compute_auc(y_val, model.predict_proba(X_val)[:,1])
        results.append({"kind": kind, "params": params, "val_auc": auc})
        print(f"{kind} {params}: val AUC {auc:.3f}")
        if auc > best_auc:
            best_auc, best_model, best_params = auc, model, params
    return best_model, best_params, results

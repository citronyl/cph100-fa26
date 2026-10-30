"""Student implementations for Parts 1.1, 1.2, and 1.4."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_readmission_rates(frame, output_dir):
    """TODO: Plot readmission rates and group sizes by age, gender, race."""
    overall = frame["target"].mean()
    fig, axes = plt.subplots(1, 3, figsize = (16,5), sharey = True)

    for ax, group in zip(axes, ["age", "gender", "race"]):
        summary = frame.groupby(group, dropna = False).target.agg(n="size", rate="mean")
        summary.index = summary.index.fillna("Missing")
        summary["ci"] = 1.96 * np.sqrt(summary["rate"] * (1-summary["rate"]) / summary["n"])
        print(summary)

        labels = [f"{name}\n(n={n:,})" for name, n in zip(summary.index, summary["n"])]
        ax.bar(labels, summary["rate"], yerr = summary["ci"], capsize = 3)
        ax.axhline(overall, linestyle="--", color = "green", label=f"Overall ({100 * overall:.2f}%)")
        ax.set_title(f"30-day readmission by {group}")
        plt.setp(ax.get_xticklabels(), rotation=90, ha="center")

    axes[0].set_ylabel("Readmission rate (<30 days)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "readmission_by_demographics.png", dpi=150)
    plt.close(fig)

def compute_auc(y_true, probabilities):
    """TODO: Compute ROC AUC from predicted probabilities (not class labels)."""
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y_true, probabilities)


def auc_confidence_interval(y_true, probabilities, seed=42, confidence=0.95,
                            n_bootstrap=1000):
    """Return (lower, upper) percentile bootstrap CI for ROC AUC.

    Resample observation pairs, skipping single-class samples. This assumes
    independently sampled evaluation rows; repeated patient encounters may
    require patient-level clustered resampling instead.
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y_true)
    p = np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p):
        raise ValueError("y_true and probabilities must be equal-length vectors")
    if len(y) == 0 or not np.isfinite(p).all():
        raise ValueError("Nonempty finite probabilities are required")
    if len(np.unique(y)) != 2:
        raise ValueError("ROC AUC requires both outcome classes")
    if not 0 < confidence < 1 or n_bootstrap < 1:
        raise ValueError("Invalid confidence or n_bootstrap")
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(y), size=len(y))
        if len(np.unique(y[indices])) < 2:
            continue
        aucs.append(roc_auc_score(y[indices], p[indices]))
    if not aucs:
        raise ValueError("No bootstrap samples contained both classes")
    alpha = (1 - confidence) / 2
    return tuple(float(v) for v in np.quantile(aucs, [alpha, 1 - alpha]))


def compare_models(results, output_dir):
    """TODO: Plot model AUCs with confidence intervals and describe variation."""
    kinds = ["linear", "tree", "neural"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, design in zip(axes, ["patient", "encounter"]):
        sub = results[results["design"] == design]
        seeds = sorted(sub["seed"].unique())
        for j, seed in enumerate(seeds):
            rows = sub[sub["seed"] == seed].set_index("kind").loc[kinds]
            x = np.arange(len(kinds)) + (j - (len(seeds) -1) / 2) * 0.15
            yerr = [rows["test_auc"] - rows["ci_low"], rows["ci_high"] - rows["test_auc"]]
            ax.errorbar(x, rows["test_auc"], yerr=yerr, fmt = "o", capsize = 3, label = f"seed {seed}")
        ax.set_xticks(range(len(kinds)), kinds)
        ax.set_title("Patient-disjoint split" if design == "patient" else "Encounter-level split")
    axes[0].set_ylabel("Test AUC (95% bootstrap CI)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "model_comparison.png", dpi=150)
    plt.close(fig)
    print(results.groupby(["design", "kind"])["test_auc"].agg(["mean", "std"]).round(3))
    #raise NotImplementedError("Student TODO: model comparison plot")


def analyze_age_shift(frame, seed=42):
    """TODO: Train on under-50 patients, evaluate on held-out under-50 and
    on 50-plus patients. Investigate outcome/feature differences and avoid
    using the older group's outcomes in training or model selection.
    """
    from data_utils import split_data, model_features
    from models import tune_model

    train, val, test = split_data(frame, seed=seed, patient_disjoint=True)
    train_young = train[train["age_group"] == "under_50"]
    val_young = val[val["age_group"] == "under_50"]
    model, params, _ = tune_model("linear", train_young, val_young, seed = seed)
    print(params)
    for group in ["under_50", "50_plus"]:
        X, y = model_features(test[test["age_group"] == group])
        probs = model.predict_proba(X)[:, 1]
        print(group, len(y), y.mean(), compute_auc(y, probs), auc_confidence_interval(y, probs, seed=seed))
    return model

    raise NotImplementedError("Student TODO: distribution shift")

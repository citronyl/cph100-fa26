"""Starter orchestration; no model results are assumed or supplied."""
import argparse
from pathlib import Path
import pandas as pd
from data_utils import load_data, add_age_group, split_data, make_llm_sample, model_features
from evaluation import plot_readmission_rates, compute_auc, auc_confidence_interval, compare_models
from models import tune_model

NO_READMIT_CODES = [11, 13, 14, 19, 20, 21]
def run_experiments(frame, seeds, patient_disjoint):
    """Tune each model family on val, eval winner on test, per split."""
    design = "patient" if patient_disjoint else "encounter"
    rows, search_log = [], []

    for seed in seeds:
        train, val, test = split_data(frame, seed=seed, patient_disjoint=patient_disjoint)
        X_test, y_test = model_features(test)
        for kind in ["linear", "tree", "neural"]:
            print(f"\n=== seed {seed}: {kind} ===")
            model, params, search = tune_model(kind, train, val, seed=seed)
            probs = model.predict_proba(X_test)[:,1]
            low, high = auc_confidence_interval(y_test, probs, seed=seed)
            row = {"design": design, "seed": seed, "kind": kind, "params": str(params),
                   "test_auc": compute_auc(y_test, probs), "ci_low": low, "ci_high": high}
            if not patient_disjoint:
                seen = test["patient_nbr"].isin(train["patient_nbr"]).to_numpy()
                row["overlap"] = seen.mean()
                row["auc_seen"] = compute_auc(y_test[seen], probs[seen])
                row["auc_unseen"] = compute_auc(y_test[~seen], probs[~seen])
            rows.append(row)
            search_log += [{**r, "seed": seed} for r in search]
    Path("outputs").mkdir(exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv(f"outputs/model_results_{design}.csv", index=False)
    pd.DataFrame(search_log).to_csv(f"outputs/tuning_log_{design}.csv", index=False)
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/diabetic_data.csv")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--make-llm-sample", action="store_true")
    ap.add_argument("--sample-size", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patient-disjoint", action="store_true")
    ap.add_argument("--explore", action="store_true")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = ap.parse_args()
    frame = add_age_group(load_data(args.data))
    #train, val, test = split_data(frame, seed=args.seed, patient_disjoint=args.patient_disjoint)
    cohort = frame[~frame["discharge_disposition_id"].isin(NO_READMIT_CODES)]
    train, val, test = split_data(cohort, seed=args.seed, patient_disjoint=args.patient_disjoint)
    if args.inspect:
        print(f"Full dataset: {len(frame)} rows; positives: {frame.target.mean():.3f}")
        for label, part in (("train", train), ("val", val), ("test", test)):
            print(f"{label}: {len(part)} rows; positives: {part.target.mean():.3f}")
        print("Readmission by age:")
        print(frame.groupby("age", dropna=False).target.agg(["size", "mean"]))
    if args.make_llm_sample:
        sample = make_llm_sample(test, n=args.sample_size, seed=args.seed)
        output = Path("outputs/llm_sample.csv")
        output.parent.mkdir(exist_ok=True)
        sample.to_csv(output, index=False)
        print(f"Saved {len(sample)} held-out encounters to {output}")
    if args.explore:
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        plot_readmission_rates(frame, output_dir)
    if args.compare:
        files = sorted(Path("outputs").glob("model_results_*.csv"))
        results = pd.concat([pd.read_csv(f) for f in files])
        compare_models(results, Path("outputs"))
    if not (args.inspect or args.make_llm_sample or args.explore or args.compare):
        results = run_experiments(cohorts, args.seeds, args.patient_disjoint)
        print(results)


if __name__ == "__main__":
    main()

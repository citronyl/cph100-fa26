"""Starter orchestration; no model results are assumed or supplied."""
import argparse
from pathlib import Path
from data_utils import load_data, add_age_group, split_data, make_llm_sample
from evaluation import plot_readmission_rates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/diabetic_data.csv")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--make-llm-sample", action="store_true")
    ap.add_argument("--sample-size", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patient-disjoint", action="store_true")
    ap.add_argument("--explore", action="store_true")
    args = ap.parse_args()
    frame = add_age_group(load_data(args.data))
    train, val, test = split_data(frame, seed=args.seed, patient_disjoint=args.patient_disjoint)
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
    if not args.inspect and not args.make_llm_sample and not args.explore:
        print("TODO: Fit and compare three model classes on train/val/test splits.")
        print("See models.py, evaluation.py and README.md.")


if __name__ == "__main__":
    main()

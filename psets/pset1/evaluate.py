import csv
import json
import os, subprocess
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve
)
import sys

def generate_model():
    """
    Run main.py with custom parameters and save model as prediction and weight files.
    If model exists, skip. 
    """

    predictions_file = "test_predictions.csv"
    weights_file = "model_weights.json"

    if not os.path.exists(predictions_file) or not os.path.exists(weights_file):
        print("Model not found, training now...")

        command = [sys.executable, "main.py"]
        command.extend(["--learning_rate", '0.01'])
        command.extend(["--batch_size", '64'])
        command.extend(["--num_epochs", '1000'])
        command.extend(["--regularization_lambda", '0.01'])

        subprocess.run(command, check=True)
        # TODO: Parse the results from the experiment and return them as a dict
        #train_auc, val_auc = None, None

        #with open(temp_results_path, "r", encoding="utf-8") as f:
            #data = json.load(f)
            #train_auc = data.get("train_auc")
            #val_auc = data.get("val_auc")
        # os.remove(temp_results_path) # I think I'm supposed to keep the results from the experiment?

        #results = {
            #**experiment_config,
            #"train_auc": train_auc,
            #"val_auc": val_auc
        #}
        #return results

def main():
    generate_model()

    print("2.1 LOSS CURVES")
    if os.path.exists("loss_history.json"):
        with open("loss_history.json", "r") as f:
            loss_data = json.load(f)

        epochs = range(1, len(loss_data["train_loss"]) + 1)
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, loss_data["train_loss"], label="Train Loss", color="blue")
        if loss_data["val_loss"]:
            plt.plot(
                epochs,
                loss_data["val_loss"],
                label="Val Loss",
                color="orange",
                linestyle="--",
            )
        plt.title("Training and Validation Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Binary Cross-Entropy Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("loss_curves.png", dpi=300)
        print("Saved loss plot to 'loss_curves.png'\n")

    pred = pd.read_csv("test_predictions.csv")
    pred = pred.apply(pd.to_numeric, errors = "coerce").fillna(0).astype(float)

    y_real = pred["lung_cancer"].values
    y_choice = pred["model_pred"].values
    y_prob = pred["model_prob"].values

    print("2.2 ANALYZING OVERALL MODEL PERFORMANCE")

    cm = confusion_matrix(y_real, y_choice)
    tn, fp, fn, tp = cm.ravel()

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0

    print(f"Sensitivity: {sens:.4f} | Specificity: {spec:.4f} | Precision: {prec:.4f}")


    fpr, tpr, roc_thresholds = roc_curve(y_real, y_prob)
    precision, recall, pr_thresholds = precision_recall_curve(y_real, y_prob)
    pr_auc = auc(recall, precision)

    nlst_cm = confusion_matrix(y_real, pred["nlst_flag"])
    tn_n, fp_n, fn_n, tp_n = nlst_cm.ravel()

    nlst_sens = tp_n / (tp_n + fn_n) if (tp_n + fn_n) > 0 else 0
    nlst_spec = tn_n / (tn_n + fp_n) if (tn_n + fp_n) > 0 else 0
    nlst_prec = tp_n / (tp_n + fp_n) if (tp_n + fp_n) > 0 else 0
    nlst_fpr = 1 - nlst_spec

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(
        fpr, 
        tpr,
        label = f"Model ROC (AUC = {roc_auc_score(y_real, y_prob)})"
    )
    axes[0].plot([0,1], [0,1], color="gray", linestyle="--")
    axes[0].plot(
        nlst_fpr,
        nlst_sens,
        marker="o",
        markersize=8,
        color="red",
        label="NLST Criterion"
    )

    axes[0].set_title("ROC Curve")
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)")
    axes[0].set_ylabel("True Positive Rate (Sensitivity)")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.4)

    axes[1].plot(
        recall,
        precision,
        label=f"Model PR (AUC = {pr_auc:.4f})"
    )
    axes[1].plot(
        nlst_sens,
        nlst_prec,
        marker="o",
        color="cyan",
        label="NLST Criterion"
    )
    axes[1].set_title("Precision-Recall Curve")
    axes[1].set_xlabel("Recall (Sensitivity)")
    axes[1].set_ylabel("Precision (PPV)")
    axes[1].legend(loc="lower left")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("roc_pr_curves.png", dpi=500)
    print("Saved ROC and PR curves to roc_pr_curves.png")

    print("\n SUBGROUP ANALYSIS \n")
    subgroup_cols = ["sex", "race7", "educat", "cig_stat", "nlst_flag"]
    subgroup_results = []

    for col in subgroup_cols:
        for group_val, group_df in pred.groupby(col):
            if len(group_df["lung_cancer"].unique()) > 1:
                group_auc = roc_auc_score(
                    group_df["lung_cancer"], group_df["model_prob"]
                )
                subgroup_results.append({
                    "Feature": col,
                    "Subgroup": group_val,
                    "Sample Count": len(group_df),
                    "Positives": group_df["lung_cancer"].sum(),
                    "AUC": round(group_auc, 4)
                })

    subgroup_df = pd.DataFrame(subgroup_results)
    print(subgroup_df.to_string(index=False))
    print("\n")

    print("2.3 MODEL FEATUREs")
    with open("model_weights.json", "r") as f:
        w_data = json.load(f)

    weights = np.array(w_data["weights"])
    bias = w_data.get("bias", 0.0)

    feature_names = w_data.get(
        "feature_names", [f"Feature_{i}" for i in range(len(weights))]
    )

    top_indices = np.argsort(np.abs(weights))[::-1][:5]

    print(f"Model Intercept (Bias): {bias:.4f}\n")
    print(
        f"{'Rank':<5} | {'Feature':<18} | {'Weight (Theta)':<15} | {'Odds Ratio (e^w)':<15}"
    )
    print("-" * 66)

    for rank, idx in enumerate(top_indices, start=1):
      w_val = weights[idx]
      odds_ratio = np.exp(w_val)
      print(
          f"{rank:<5} | {feature_names[idx]:<22} | {w_val:<15.4f} |"
          f" {odds_ratio:<15.4f}"
      )

    print("\n")

    print("2.4 SIMULATING CLINICAL UTILITY")
    specificity = 1 - fpr
    idx_matched = np.argmin(np.abs(specificity - nlst_spec))
    matched_threshold = roc_thresholds[idx_matched]
    matched_sens = tpr[idx_matched]
    matched_spec = specificity[idx_matched]

    matched_preds = (y_prob >= matched_threshold).astype(int)
    cm_m = confusion_matrix(y_real, matched_preds)
    matched_prec = (
        cm_m[1,1]/(cm_m[1,1] + cm_m[0,1])
        if (cm_m[1,1] + cm_m[0,1] > 0) else 0
    )

    print(
      f"NLST Criteria Baseline -> Sensitivity: {nlst_sens:.4f} | Specificity:"
      f" {nlst_spec:.4f} | PPV: {nlst_prec:.4f}"
    )
    print(
      f"Matched Model          -> Sensitivity: {matched_sens:.4f} | Specificity:"
      f" {matched_spec:.4f} | PPV: {matched_prec:.4f} (Threshold:"
      f" {matched_threshold:.4f})"
    )

    print(f"SUBGROUP PERFORMANCE AT MATCHED THRESHOLD ({matched_threshold:.4f})")
    subgroup_metrics = []
    for col in subgroup_cols:
        for group_val, group_df in pred.groupby(col):
            y_sub = group_df["lung_cancer"].values
            p_sub = group_df["model_prob"].values
            preds_sub = (p_sub >= matched_threshold).astype(int)

            cm_sub = confusion_matrix(y_sub, preds_sub, labels=[0,1])
            tn_s, fp_s, fn_s, tp_s = cm_sub.ravel()

            sens_s = tp_s / (tp_s + fn_s) if (tp_s + fn_s) > 0 else 0
            spec_s = tn_s / (tn_s + fp_s) if (tn_s + fp_s) > 0 else 0
            ppv_s = tp_s / (tp_s + fp_s) if (tp_s + fp_s) > 0 else 0

            subgroup_metrics.append({
                "Feature": col,
                "Subgroup": group_val,
                "Count": len(group_df),
                "Sens": round(sens_s, 4),
                "Spec": round(spec_s, 4),
                "PPV": round(ppv_s, 4),
            })
    sub_metrics_df = pd.DataFrame(subgroup_metrics)
    print(sub_metrics_df.to_string(index=False))

if __name__ == "__main__":
    main()
"""
Model Evaluation and Visual Reporting Engine.
Calculates classification metrics, confusion matrices, anomaly distributions,
and exports high-resolution publication charts and benchmark logs.
"""

import os
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score


def evaluate_and_generate_reports(
    y_true: np.ndarray,
    y_pred: List[str],
    class_names: List[str],
    comparison_df: pd.DataFrame,
    anomaly_scores: np.ndarray,
    load_profiles: List[str],
    reports_dir: str = "reports"
) -> Dict[str, Any]:
    """
    Evaluates ML predictions, generates publication figures and text audit logs.
    """
    os.makedirs(reports_dir, exist_ok=True)

    # Convert y_true indices to string labels if needed
    if isinstance(y_true[0], (int, np.integer)):
        y_true_labels = [class_names[idx] for idx in y_true]
    else:
        y_true_labels = list(y_true)

    # Metrics
    acc = accuracy_score(y_true_labels, y_pred)
    f1 = f1_score(y_true_labels, y_pred, average="weighted", zero_division=0)
    rep_text = classification_report(y_true_labels, y_pred, target_names=class_names, zero_division=0)

    # 1. Export model_evaluation.txt
    txt_path = os.path.join(reports_dir, "model_evaluation.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("========================================================================\n")
        f.write("SMART LOAD MONITOR: MACHINE LEARNING BENCHMARK & EVALUATION AUDIT\n")
        f.write("========================================================================\n\n")
        f.write("1. ALGORITHM BENCHMARK COMPARISON:\n")
        f.write(comparison_df.to_string(index=False))
        f.write("\n\n" + "-" * 72 + "\n")
        f.write("2. TEST SET CLASSIFICATION REPORT (BEST MODEL):\n")
        f.write(f"Test Accuracy: {acc:.4f} | Weighted F1-Score: {f1:.4f}\n\n")
        f.write(rep_text)
        f.write("\n" + "=" * 72 + "\n")

    # 2. Confusion Matrix Plot
    cm = confusion_matrix(y_true_labels, y_pred, labels=class_names)
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True
    )
    plt.title("Multiclass Fault Classification Confusion Matrix", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Operating Condition", fontsize=11, labelpad=8)
    plt.ylabel("Ground Truth Condition", fontsize=11, labelpad=8)
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    cm_path = os.path.join(reports_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()

    # 3. Model Comparison Plot
    plt.figure(figsize=(10, 5))
    comp_melted = comparison_df.melt(
        id_vars="Algorithm",
        value_vars=["Accuracy", "Precision", "Recall", "F1_Score"],
        var_name="Metric",
        value_name="Score"
    )
    palette = ["#0284c7", "#10b981", "#f59e0b", "#6366f1"]
    ax = sns.barplot(data=comp_melted, x="Algorithm", y="Score", hue="Metric", palette=palette)
    plt.title("Comparative Performance Across Evaluated ML Algorithms", fontsize=13, fontweight="bold", pad=12)
    plt.ylabel("Score (0.0 - 1.0)", fontsize=11)
    plt.ylim(0.0, 1.08)
    for p in ax.patches:
        height = p.get_height()
        if height > 0.05:
            ax.annotate(f"{height:.2f}",
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=8, rotation=0, xytext=(0, 2),
                        textcoords='offset points')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    comp_path = os.path.join(reports_dir, "model_comparison.png")
    plt.savefig(comp_path, dpi=200)
    plt.close()

    # 4. Anomaly Score Distribution Plot
    plt.figure(figsize=(8, 5))
    sns.histplot(anomaly_scores, bins=35, kde=True, color="#0ea5e9", edgecolor="white")
    plt.axvline(0.55, color="#ef4444", linestyle="--", linewidth=2, label="Anomaly Threshold (0.55)")
    plt.title("Isolation Forest Anomaly Score Distribution", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Normalized Anomaly Score [0 = Nominal, 1 = Outlier]", fontsize=11)
    plt.ylabel("Observation Count", fontsize=11)
    plt.legend()
    plt.tight_layout()
    anom_path = os.path.join(reports_dir, "anomaly_distribution.png")
    plt.savefig(anom_path, dpi=200)
    plt.close()

    # 5. Load Cluster Regime Distribution Plot
    plt.figure(figsize=(8, 5))
    profile_series = pd.Series(load_profiles).value_counts()
    colors = ["#10b981", "#0284c7", "#f59e0b", "#ef4444"]
    sns.barplot(x=profile_series.index, y=profile_series.values, hue=profile_series.index, palette=colors[:len(profile_series)], legend=False)
    plt.title("K-Means Energy Regime Distribution (Test Partition)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Load Regime Category", fontsize=11)
    plt.ylabel("Telemetry Count", fontsize=11)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    clust_path = os.path.join(reports_dir, "cluster_distribution.png")
    plt.savefig(clust_path, dpi=200)
    plt.close()

    return {
        "accuracy": acc,
        "f1_score": f1,
        "classification_report": rep_text
    }

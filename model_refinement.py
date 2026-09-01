"""Grouped evaluation helpers for the second HateCR baseline iteration."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src import modeling


TARGET_COLUMNS = {
    "hostility": "y_hostility",
    "hate": "y_hate_speech",
}


def metrics_at_threshold(
    y_true: pd.Series,
    scores: pd.Series,
    threshold: float,
) -> Dict[str, float]:
    true_values = pd.to_numeric(y_true, errors="raise").astype(int).to_numpy()
    score_values = pd.to_numeric(scores, errors="raise").to_numpy()
    predictions = (score_values >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "predicted_positive": int(predictions.sum()),
        "accuracy": float(accuracy_score(true_values, predictions)),
        "precision_positive": float(
            precision_score(true_values, predictions, zero_division=0)
        ),
        "recall_positive": float(
            recall_score(true_values, predictions, zero_division=0)
        ),
        "f1_positive": float(
            f1_score(true_values, predictions, zero_division=0)
        ),
        "macro_f1": float(
            f1_score(true_values, predictions, average="macro", zero_division=0)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(true_values, predictions)
        ),
    }


def summarize_default_model_consensus(
    consensus_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize default out-of-fold decisions for every target and model."""
    required = {
        "target",
        "model",
        "y_true",
        "cv_score_mean",
        "score_threshold",
        "cv_consensus_pred_default",
    }
    missing = required.difference(consensus_df.columns)
    if missing:
        raise KeyError(f"Missing consensus columns: {sorted(missing)}")

    metric_rows = []
    confusion_rows = []
    for (target, model), subset in consensus_df.groupby(
        ["target", "model"], sort=False
    ):
        y_true = pd.to_numeric(subset["y_true"], errors="raise").astype(int)
        predictions = pd.to_numeric(
            subset["cv_consensus_pred_default"], errors="raise"
        ).astype(int)
        scores = pd.to_numeric(
            subset["cv_score_mean"], errors="raise"
        ).astype(float)
        threshold = float(subset["score_threshold"].iloc[0])
        metric_rows.append(
            {
                "target": str(target),
                "model": str(model),
                "n_rows": int(len(subset)),
                "score_threshold": threshold,
                "predicted_positive": int(predictions.sum()),
                "accuracy": float(accuracy_score(y_true, predictions)),
                "precision_positive": float(
                    precision_score(y_true, predictions, zero_division=0)
                ),
                "recall_positive": float(
                    recall_score(y_true, predictions, zero_division=0)
                ),
                "f1_positive": float(
                    f1_score(y_true, predictions, zero_division=0)
                ),
                "macro_f1": float(
                    f1_score(
                        y_true, predictions, average="macro", zero_division=0
                    )
                ),
                "balanced_accuracy": float(
                    balanced_accuracy_score(y_true, predictions)
                ),
                "average_precision": float(
                    average_precision_score(y_true, scores)
                ),
            }
        )
        matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
        for actual in [0, 1]:
            for predicted in [0, 1]:
                confusion_rows.append(
                    {
                        "target": str(target),
                        "model": str(model),
                        "actual": actual,
                        "predicted": predicted,
                        "n_rows": int(matrix[actual, predicted]),
                    }
                )
    return pd.DataFrame(metric_rows), pd.DataFrame(confusion_rows)


def evaluate_combined_training_set(
    training_df: pd.DataFrame,
    text_column: str = "text_model",
    group_column: str = "source_post_id",
    n_splits: int = 5,
    n_repeats: int = 5,
    random_state: int = 42,
    min_precision: float = 0.80,
    lexicon_prediction_column: str = "categorized_lexicon_hit_count",
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Evaluate word and word+character models for both binary targets."""
    id_columns = [
        "review_id",
        "tweet_id",
        "source_post_id",
        "annotation_source",
        "event_id",
        "anchor_media_handle",
    ]
    all_folds = []
    all_summaries = []
    all_consensus = []
    threshold_tables = []
    selected_rows = []
    operating_rows = []

    for target_name, target_column in TARGET_COLUMNS.items():
        folds, summary, consensus = modeling.run_repeated_stratified_group_text_cv(
            training_df,
            text_column=text_column,
            target_column=target_column,
            group_column=group_column,
            id_columns=id_columns,
            lexicon_prediction_column=lexicon_prediction_column,
            include_models=(
                "lexicon_hits",
                "logreg_tfidf",
                "linearsvc_tfidf",
                "word_char_logreg",
            ),
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=random_state,
        )
        folds.insert(0, "target", target_name)
        summary.insert(0, "target", target_name)
        consensus.insert(0, "target", target_name)
        all_folds.append(folds)
        all_summaries.append(summary)
        all_consensus.append(consensus)

        word_char = consensus[consensus["model"].eq("word_char_logreg")].copy()
        balanced_table, balanced = modeling.select_threshold_max_f1(
            word_char["y_true"], word_char["cv_score_mean"]
        )
        high_precision_table, high_precision = (
            modeling.select_threshold_for_min_precision(
                word_char["y_true"],
                word_char["cv_score_mean"],
                min_precision=min_precision,
            )
        )
        threshold_table = balanced_table.copy()
        threshold_table.insert(0, "target", target_name)
        threshold_tables.append(threshold_table)

        for profile, selected in [
            ("balanced", balanced),
            ("high_precision", high_precision),
        ]:
            selected_row = {"target": target_name, "profile": profile, **selected}
            selected_rows.append(selected_row)
            metrics = metrics_at_threshold(
                word_char["y_true"],
                word_char["cv_score_mean"],
                float(selected["threshold"]),
            )
            operating_rows.append(
                {"target": target_name, "profile": profile, **metrics}
            )

    folds_df = pd.concat(all_folds, ignore_index=True)
    summary_df = pd.concat(all_summaries, ignore_index=True)
    consensus_df = pd.concat(all_consensus, ignore_index=True)
    threshold_df = pd.concat(threshold_tables, ignore_index=True)
    selected_df = pd.DataFrame(selected_rows)
    operating_df = pd.DataFrame(operating_rows)
    return (
        folds_df,
        summary_df,
        consensus_df,
        threshold_df,
        selected_df,
        operating_df,
    )


def selected_threshold_map(selected_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    output: Dict[str, Dict[str, float]] = {}
    for row in selected_df.itertuples(index=False):
        output.setdefault(str(row.target), {})[str(row.profile)] = float(
            row.threshold
        )
    return output


def add_profile_predictions(
    consensus_df: pd.DataFrame,
    selected_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add balanced and high-precision decisions to word+character OOF rows."""
    thresholds = selected_threshold_map(selected_df)
    output = consensus_df[consensus_df["model"].eq("word_char_logreg")].copy()
    for profile in ["balanced", "high_precision"]:
        output[f"pred_{profile}"] = output.apply(
            lambda row: int(
                row["cv_score_mean"] >= thresholds[str(row["target"])][profile]
            ),
            axis=1,
        )
    return output


def save_confusion_figure(
    consensus_profiles_df: pd.DataFrame,
    target: str,
    profile: str,
    output_path: Path,
) -> np.ndarray:
    subset = consensus_profiles_df[
        consensus_profiles_df["target"].eq(target)
    ].copy()
    prediction_column = f"pred_{profile}"
    matrix = confusion_matrix(
        subset["y_true"], subset[prediction_column], labels=[0, 1]
    )
    normalized = np.divide(
        matrix,
        matrix.sum(axis=1, keepdims=True),
        out=np.zeros_like(matrix, dtype=float),
        where=matrix.sum(axis=1, keepdims=True) != 0,
    )
    target_label = "hostilidad" if target == "hostility" else "odio"
    profile_label = "balanceado" if profile == "balanced" else "alta precisión"
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    image = ax.imshow(normalized, cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks([0, 1], labels=["Pred. 0", "Pred. 1"])
    ax.set_yticks([0, 1], labels=["Manual 0", "Manual 1"])
    ax.set_xlabel("Predicción fuera de muestra")
    ax.set_ylabel("Etiqueta manual")
    ax.set_title(
        f"Modelo v2 de {target_label}: perfil {profile_label}\n"
        "Word+char TF-IDF, CV agrupada por post madre"
    )
    for row in range(2):
        for column in range(2):
            rate = normalized[row, column]
            color = "white" if rate >= 0.58 else "#2D2D2D"
            ax.text(
                column,
                row,
                f"{int(matrix[row, column])}\n{rate:.1%}",
                ha="center",
                va="center",
                color=color,
                fontsize=13,
                fontweight="bold",
            )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Proporción dentro de etiqueta manual")
    fig.text(
        0.5,
        0.02,
        "Evaluación exploratoria sobre muestra combinada y enriquecida; no estima prevalencia.",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return matrix


def recheck_known_false_negatives(
    consensus_profiles_df: pd.DataFrame,
    known_false_negatives_df: pd.DataFrame,
) -> pd.DataFrame:
    known_ids = set(
        known_false_negatives_df["tweet_id"].astype("string").str.strip()
    )
    hate_rows = consensus_profiles_df[
        consensus_profiles_df["target"].eq("hate")
        & consensus_profiles_df["tweet_id"].astype("string").isin(known_ids)
    ].copy()
    keep_columns = [
        "review_id",
        "tweet_id",
        "source_post_id",
        "event_id",
        "anchor_media_handle",
        "y_true",
        "cv_score_mean",
        "cv_score_std",
        "pred_balanced",
        "pred_high_precision",
    ]
    return hate_rows[keep_columns].sort_values(
        "cv_score_mean", ascending=False
    ).reset_index(drop=True)

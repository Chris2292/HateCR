"""Historical baseline reconstruction for the first HateCR labeled sample."""

from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from src import modeling


MODEL_LABELS = {
    "lexicon_hits": "Lexicón",
    "logreg_tfidf": "Logistic Regression",
    "linearsvc_tfidf": "LinearSVC",
}
TARGET_LABELS = {
    "hostility": "Hostilidad",
    "hate": "Odio",
}


def _binary_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_positive": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall_positive": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1_positive": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
    }


def build_initial_reference_metrics(
    manual_df: pd.DataFrame,
    heldout_metrics_df: pd.DataFrame,
) -> pd.DataFrame:
    """Combine the original held-out supervised metrics and full lexicon audit."""
    required_manual = {
        "y_hostility",
        "y_hate_speech",
        "categorized_lexicon_hit_count",
    }
    missing_manual = required_manual.difference(manual_df.columns)
    if missing_manual:
        raise KeyError(f"Missing manual columns: {sorted(missing_manual)}")

    heldout_map = {
        "linearsvc_tfidf": "LinearSVC",
        "logreg_tfidf": "Logistic Regression",
    }
    rows = []
    for model_name, model_label in heldout_map.items():
        subset = heldout_metrics_df[heldout_metrics_df["model"].eq(model_name)]
        if subset.empty:
            raise ValueError(f"Missing historical held-out model: {model_name}")
        row = subset.iloc[0]
        rows.append(
            {
                "target": "hostility",
                "target_label": "Hostilidad",
                "model": model_name,
                "model_label": model_label,
                "evaluation_design": "holdout_stratified_25pct",
                "n_eval": int(row["n_test"]),
                "accuracy": float(row["accuracy"]),
                "precision_positive": float(row["precision_hostile"]),
                "recall_positive": float(row["recall_hostile"]),
                "f1_positive": float(row["f1_hostile"]),
                "macro_f1": float(row["macro_f1"]),
            }
        )

    lexicon_predictions = (
        pd.to_numeric(
            manual_df["categorized_lexicon_hit_count"], errors="coerce"
        )
        .fillna(0)
        .gt(0)
        .astype(int)
    )
    for target, target_column in [
        ("hostility", "y_hostility"),
        ("hate", "y_hate_speech"),
    ]:
        y_true = pd.to_numeric(manual_df[target_column], errors="raise").astype(int)
        metrics = _binary_metrics(y_true, lexicon_predictions)
        rows.append(
            {
                "target": target,
                "target_label": TARGET_LABELS[target],
                "model": "lexicon_hits",
                "model_label": "Lexicón",
                "evaluation_design": "full_manual_sample",
                "n_eval": int(len(manual_df)),
                **metrics,
            }
        )
    order = {
        ("hostility", "linearsvc_tfidf"): 0,
        ("hostility", "logreg_tfidf"): 1,
        ("hostility", "lexicon_hits"): 2,
        ("hate", "lexicon_hits"): 3,
    }
    output = pd.DataFrame(rows)
    output["display_order"] = [
        order[(target, model)]
        for target, model in zip(output["target"], output["model"])
    ]
    return output.sort_values("display_order").drop(columns="display_order")


def run_initial_repeated_cv(
    manual_df: pd.DataFrame,
    n_splits: int = 5,
    n_repeats: int = 5,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate initial-model variability using its original ungrouped CV design."""
    working = manual_df.copy()
    if "text_model" not in working.columns:
        working["text_model"] = working["text"].map(
            modeling.normalize_text_for_model
        )
    fold_frames = []
    summary_frames = []
    for target, target_column in [
        ("hostility", "y_hostility"),
        ("hate", "y_hate_speech"),
    ]:
        folds, summary, _ = modeling.run_repeated_stratified_text_cv(
            working,
            text_column="text_model",
            target_column=target_column,
            id_columns=["review_id", "tweet_id", "source_post_id"],
            lexicon_prediction_column="categorized_lexicon_hit_count",
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=random_state,
        )
        models = set(MODEL_LABELS)
        folds = folds[folds["model"].isin(models)].copy()
        summary = summary[summary["model"].isin(models)].copy()
        folds.insert(0, "target", target)
        folds.insert(1, "target_label", TARGET_LABELS[target])
        summary.insert(0, "target", target)
        summary.insert(1, "target_label", TARGET_LABELS[target])
        summary["n_source_rows"] = len(working)
        summary["n_positive"] = int(working[target_column].sum())
        summary["evaluation_design"] = "RepeatedStratifiedKFold_5x5_ungrouped"
        fold_frames.append(folds)
        summary_frames.append(summary)
    return (
        pd.concat(fold_frames, ignore_index=True),
        pd.concat(summary_frames, ignore_index=True),
    )


def build_variability_display_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Create a compact human-readable mean, standard deviation, and range table."""
    rows = []
    for row in summary_df.itertuples(index=False):
        rows.append(
            {
                "target": row.target,
                "target_label": row.target_label,
                "model": row.model,
                "model_label": MODEL_LABELS.get(row.model, row.model),
                "n_evaluations": int(row.n_evaluations),
                "n_positive": int(row.n_positive),
                "accuracy_mean_sd": f"{row.accuracy_mean:.3f} ± {row.accuracy_std:.3f}",
                "precision_mean_sd": (
                    f"{row.precision_positive_mean:.3f} ± "
                    f"{row.precision_positive_std:.3f}"
                ),
                "recall_mean_sd": (
                    f"{row.recall_positive_mean:.3f} ± "
                    f"{row.recall_positive_std:.3f}"
                ),
                "f1_mean_sd": (
                    f"{row.f1_positive_mean:.3f} ± {row.f1_positive_std:.3f}"
                ),
                "f1_min_max": (
                    f"{row.f1_positive_min:.3f}–{row.f1_positive_max:.3f}"
                ),
                "macro_f1_mean_sd": (
                    f"{row.macro_f1_mean:.3f} ± {row.macro_f1_std:.3f}"
                ),
            }
        )
    return pd.DataFrame(rows)

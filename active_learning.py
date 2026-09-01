"""Utilities for error review and active-learning annotation batches."""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


LABEL_COLUMNS = [
    "hostility_relevance",
    "manual_hostility",
    "manual_hate_speech",
    "notes",
]


def normalize_text_key(value: object) -> str:
    """Create a stable comparison key without changing annotation text."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).lower().strip()
    text = unicodedata.normalize("NFD", text)
    text = "".join(
        character
        for character in text
        if unicodedata.category(character) != "Mn"
    )
    return re.sub(r"\s+", " ", text).strip()


def _as_string_id(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)


def _source_context(source_posts_df: pd.DataFrame) -> pd.DataFrame:
    required = ["source_post_id", "source_post_text", "source_post_url"]
    context = source_posts_df.copy()
    for column in required:
        if column not in context.columns:
            context[column] = pd.NA
    context["source_post_id"] = _as_string_id(context["source_post_id"])
    return context[required].drop_duplicates("source_post_id", keep="first")


def identify_hate_false_negatives(
    cv_predictions_df: pd.DataFrame,
    manual_annotations_df: pd.DataFrame,
    source_posts_df: pd.DataFrame,
    model_name: str = "logreg_tfidf",
) -> pd.DataFrame:
    """Return manually positive hate cases missed by out-of-fold consensus."""
    cv = cv_predictions_df.copy()
    required = ["model", "tweet_id", "y_true", "cv_consensus_pred"]
    missing = [column for column in required if column not in cv.columns]
    if missing:
        raise KeyError("Missing CV columns: " + ", ".join(missing))

    cv["tweet_id"] = _as_string_id(cv["tweet_id"])
    false_negatives = cv[
        cv["model"].eq(model_name)
        & pd.to_numeric(cv["y_true"], errors="coerce").eq(1)
        & pd.to_numeric(cv["cv_consensus_pred"], errors="coerce").eq(0)
    ].copy()

    manual = manual_annotations_df.copy()
    manual["tweet_id"] = _as_string_id(manual["tweet_id"])
    manual_columns = [
        column
        for column in [
            "tweet_id",
            "review_id",
            "source_post_id",
            "event_name",
            "hostility_relevance",
            "manual_hostility",
            "manual_hate_speech",
            "notes",
            "lexicon_terms_found",
            "lexicon_categories_found",
        ]
        if column in manual.columns
    ]
    false_negatives = false_negatives.merge(
        manual[manual_columns].drop_duplicates("tweet_id"),
        on="tweet_id",
        how="left",
        suffixes=("", "_manual"),
    )

    if "source_post_id" not in false_negatives.columns:
        false_negatives["source_post_id"] = pd.NA
    false_negatives["source_post_id"] = _as_string_id(
        false_negatives["source_post_id"]
    )
    false_negatives = false_negatives.merge(
        _source_context(source_posts_df),
        on="source_post_id",
        how="left",
    )
    false_negatives["error_type"] = "false_negative_hate"
    false_negatives["audit_confirm_original_label"] = pd.NA
    false_negatives["error_analysis_category"] = pd.NA
    false_negatives["audit_notes"] = pd.NA

    output_columns = [
        "review_id",
        "tweet_id",
        "event_id",
        "event_name",
        "anchor_media_handle",
        "source_type",
        "source_post_id",
        "source_post_url",
        "source_post_text",
        "text",
        "hostility_relevance",
        "manual_hostility",
        "manual_hate_speech",
        "lexicon_terms_found",
        "lexicon_categories_found",
        "cv_pred_rate",
        "cv_score_mean",
        "cv_score_std",
        "n_evaluations",
        "error_type",
        "audit_confirm_original_label",
        "error_analysis_category",
        "audit_notes",
    ]
    for column in output_columns:
        if column not in false_negatives.columns:
            false_negatives[column] = pd.NA
    return false_negatives[output_columns].sort_values(
        ["cv_score_mean", "tweet_id"], ascending=[False, True]
    ).reset_index(drop=True)


def _take_ranked(
    pool: pd.DataFrame,
    selected_ids: set,
    selected_texts: set,
    mask: pd.Series,
    n_rows: Optional[int],
    group_name: str,
    priority: int,
    sort_columns: Sequence[str],
) -> pd.DataFrame:
    candidates = pool[
        mask
        & ~pool["tweet_id"].isin(selected_ids)
        & ~pool["_text_key"].isin(selected_texts)
    ].copy()
    candidates = candidates.sort_values(
        list(sort_columns), ascending=[False] * len(sort_columns)
    )
    if n_rows is not None:
        candidates = candidates.head(max(0, n_rows))
    candidates["selection_group"] = group_name
    candidates["selection_priority"] = priority
    selected_ids.update(candidates["tweet_id"].tolist())
    selected_texts.update(candidates["_text_key"].tolist())
    return candidates


def _stratified_event_sample(
    pool: pd.DataFrame,
    n_rows: int,
    random_state: int,
) -> pd.DataFrame:
    if n_rows <= 0 or pool.empty:
        return pool.head(0).copy()
    n_rows = min(n_rows, len(pool))
    event_values = pool["event_id"].fillna("unknown_event")
    counts = event_values.value_counts().sort_index()
    exact = counts / counts.sum() * n_rows
    quotas = np.floor(exact).astype(int)
    if n_rows >= len(counts):
        quotas = quotas.clip(lower=1)
    while quotas.sum() > n_rows:
        reducible = quotas[quotas > 1].sort_values(ascending=False)
        if reducible.empty:
            break
        quotas.loc[reducible.index[0]] -= 1
    remainders = (exact - np.floor(exact)).sort_values(ascending=False)
    while quotas.sum() < n_rows:
        for event_id in remainders.index:
            if quotas.sum() >= n_rows:
                break
            if quotas.loc[event_id] < counts.loc[event_id]:
                quotas.loc[event_id] += 1

    sampled_parts: List[pd.DataFrame] = []
    for event_index, event_id in enumerate(counts.index):
        group = pool[event_values.eq(event_id)]
        take = min(int(quotas.loc[event_id]), len(group))
        if take:
            sampled_parts.append(
                group.sample(n=take, random_state=random_state + event_index)
            )
    sampled = pd.concat(sampled_parts, ignore_index=False) if sampled_parts else pool.head(0)
    if len(sampled) < n_rows:
        remaining = pool.drop(index=sampled.index, errors="ignore")
        sampled = pd.concat(
            [
                sampled,
                remaining.sample(
                    n=min(n_rows - len(sampled), len(remaining)),
                    random_state=random_state + 1000,
                ),
            ]
        )
    return sampled.head(n_rows).copy()


def build_hate_active_learning_batch(
    corpus_df: pd.DataFrame,
    manual_annotations_df: pd.DataFrame,
    source_posts_df: pd.DataFrame,
    target_size: int = 300,
    borderline_size: int = 100,
    hostile_negative_size: int = 74,
    random_size: int = 50,
    random_state: int = 42,
    batch_id: str = "active_hate_001",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build a deduplicated, model-enriched batch without exposing scores to coders."""
    corpus = corpus_df.copy()
    required = [
        "tweet_id",
        "text",
        "ml_hostility_pred",
        "ml_hate_speech_pred_experimental",
        "ml_hate_speech_score_experimental",
    ]
    missing = [column for column in required if column not in corpus.columns]
    if missing:
        raise KeyError("Missing corpus columns: " + ", ".join(missing))

    corpus["tweet_id"] = _as_string_id(corpus["tweet_id"])
    corpus["_text_key"] = corpus["text"].map(normalize_text_key)
    manual = manual_annotations_df.copy()
    manual["tweet_id"] = _as_string_id(manual["tweet_id"])
    manual["_text_key"] = manual["text"].map(normalize_text_key)
    excluded_ids = set(manual["tweet_id"].dropna().tolist())
    excluded_texts = set(manual.loc[manual["_text_key"].ne(""), "_text_key"])

    corpus = corpus[
        corpus["tweet_id"].notna()
        & corpus["_text_key"].ne("")
        & ~corpus["tweet_id"].isin(excluded_ids)
        & ~corpus["_text_key"].isin(excluded_texts)
    ].copy()
    corpus = corpus.drop_duplicates("tweet_id", keep="first")
    corpus = corpus.drop_duplicates("_text_key", keep="first")
    corpus["ml_hostility_pred"] = pd.to_numeric(
        corpus["ml_hostility_pred"], errors="coerce"
    ).fillna(0).astype(int)
    corpus["ml_hate_speech_pred_experimental"] = pd.to_numeric(
        corpus["ml_hate_speech_pred_experimental"], errors="coerce"
    ).fillna(0).astype(int)
    for column in [
        "ml_hate_speech_score_experimental",
        "ml_hostility_score",
        "categorized_lexicon_hit_count",
    ]:
        if column not in corpus.columns:
            corpus[column] = 0
        corpus[column] = pd.to_numeric(corpus[column], errors="coerce").fillna(0)

    if "anchor_post_id" not in corpus.columns:
        corpus["anchor_post_id"] = pd.NA
    corpus["anchor_post_id"] = _as_string_id(corpus["anchor_post_id"])
    corpus = corpus.merge(
        _source_context(source_posts_df),
        left_on="anchor_post_id",
        right_on="source_post_id",
        how="left",
    )

    selected_ids: set = set()
    selected_texts: set = set()
    selected_parts: List[pd.DataFrame] = []
    hate_pred = corpus["ml_hate_speech_pred_experimental"].eq(1)
    hostility_pred = corpus["ml_hostility_pred"].eq(1)

    selected_parts.append(
        _take_ranked(
            corpus,
            selected_ids,
            selected_texts,
            hate_pred & ~hostility_pred,
            None,
            "predicted_hate_without_hostility",
            1,
            ["ml_hate_speech_score_experimental"],
        )
    )
    selected_parts.append(
        _take_ranked(
            corpus,
            selected_ids,
            selected_texts,
            hate_pred & hostility_pred,
            None,
            "predicted_hate_with_hostility",
            2,
            ["ml_hate_speech_score_experimental", "ml_hostility_score"],
        )
    )
    selected_parts.append(
        _take_ranked(
            corpus,
            selected_ids,
            selected_texts,
            ~hate_pred,
            borderline_size,
            "borderline_hate_negative",
            3,
            ["ml_hate_speech_score_experimental", "ml_hostility_score"],
        )
    )
    selected_parts.append(
        _take_ranked(
            corpus,
            selected_ids,
            selected_texts,
            ~hate_pred & hostility_pred,
            hostile_negative_size,
            "hostile_high_hate_score_negative",
            4,
            ["ml_hate_speech_score_experimental", "ml_hostility_score"],
        )
    )

    selected_count = sum(len(part) for part in selected_parts)
    desired_random = min(random_size, max(0, target_size - selected_count))
    remaining = corpus[
        ~corpus["tweet_id"].isin(selected_ids)
        & ~corpus["_text_key"].isin(selected_texts)
    ].copy()
    random_part = _stratified_event_sample(
        remaining,
        desired_random,
        random_state=random_state,
    )
    random_part["selection_group"] = "random_stratified"
    random_part["selection_priority"] = 5
    selected_parts.append(random_part)
    selected_ids.update(random_part["tweet_id"].tolist())
    selected_texts.update(random_part["_text_key"].tolist())

    selected = pd.concat(selected_parts, ignore_index=True)
    if len(selected) < target_size:
        remaining = corpus[
            ~corpus["tweet_id"].isin(selected_ids)
            & ~corpus["_text_key"].isin(selected_texts)
        ].copy()
        fill = _stratified_event_sample(
            remaining,
            target_size - len(selected),
            random_state=random_state + 2000,
        )
        fill["selection_group"] = "random_stratified_fill"
        fill["selection_priority"] = 6
        selected = pd.concat([selected, fill], ignore_index=True)
    selected = selected.head(target_size).copy()

    # Randomized annotation order prevents priority strata from being obvious.
    selected = selected.sample(frac=1, random_state=random_state).reset_index(drop=True)
    selected["review_id"] = [
        f"{batch_id}_{index:04d}" for index in range(1, len(selected) + 1)
    ]
    for column in LABEL_COLUMNS:
        selected[column] = pd.NA

    labeling_columns = [
        "review_id",
        "tweet_id",
        "source_type",
        "event_id",
        "event_name",
        "anchor_media_id",
        "anchor_media_handle",
        "source_post_id",
        "source_post_url",
        "source_post_text",
        "created_at",
        "text",
        "categorized_lexicon_hit_count",
        "lexicon_terms_found",
        "lexicon_categories_found",
        "hostility_relevance",
        "manual_hostility",
        "manual_hate_speech",
        "notes",
    ]
    for column in labeling_columns:
        if column not in selected.columns:
            selected[column] = pd.NA
    labeling = selected[labeling_columns].copy()

    audit_columns = [
        "review_id",
        "tweet_id",
        "selection_group",
        "selection_priority",
        "event_id",
        "anchor_media_handle",
        "ml_hostility_pred",
        "ml_hostility_score",
        "ml_hate_speech_pred_experimental",
        "ml_hate_speech_score_experimental",
        "ml_hate_hostility_disagreement",
        "categorized_lexicon_hit_count",
        "lexicon_terms_found",
        "lexicon_categories_found",
        "text_norm_hash",
    ]
    for column in audit_columns:
        if column not in selected.columns:
            selected[column] = pd.NA
    audit = selected[audit_columns].copy()

    summary_rows: List[Dict[str, object]] = []
    for group_name, count in audit["selection_group"].value_counts().items():
        summary_rows.append(
            {"summary_type": "selection_group", "group": group_name, "n_rows": int(count)}
        )
    for event_id, count in labeling["event_id"].value_counts().items():
        summary_rows.append(
            {"summary_type": "event_id", "group": event_id, "n_rows": int(count)}
        )
    for media_handle, count in labeling["anchor_media_handle"].value_counts().items():
        summary_rows.append(
            {"summary_type": "anchor_media_handle", "group": media_handle, "n_rows": int(count)}
        )
    summary = pd.DataFrame(summary_rows)
    return labeling, audit, summary


def validate_active_learning_batch(
    labeling_df: pd.DataFrame,
    manual_annotations_df: pd.DataFrame,
    expected_size: Optional[int] = None,
) -> Dict[str, int]:
    """Return compact integrity checks and raise on contamination or duplicates."""
    manual_ids = set(_as_string_id(manual_annotations_df["tweet_id"]).dropna())
    batch_ids = _as_string_id(labeling_df["tweet_id"])
    text_keys = labeling_df["text"].map(normalize_text_key)
    checks = {
        "rows": len(labeling_df),
        "unique_tweet_ids": int(batch_ids.nunique()),
        "unique_text_keys": int(text_keys.nunique()),
        "overlap_manual_tweet_ids": int(batch_ids.isin(manual_ids).sum()),
        "blank_manual_labels": int(labeling_df[LABEL_COLUMNS[:3]].isna().all(axis=1).sum()),
    }
    if expected_size is not None and len(labeling_df) != expected_size:
        raise ValueError(f"Expected {expected_size} rows, found {len(labeling_df)}")
    if checks["unique_tweet_ids"] != len(labeling_df):
        raise ValueError("Duplicate tweet_id values in active-learning batch")
    if checks["unique_text_keys"] != len(labeling_df):
        raise ValueError("Duplicate normalized texts in active-learning batch")
    if checks["overlap_manual_tweet_ids"]:
        raise ValueError("Active-learning batch overlaps canonical manual annotations")
    if checks["blank_manual_labels"] != len(labeling_df):
        raise ValueError("New labeling columns must be blank")
    return checks

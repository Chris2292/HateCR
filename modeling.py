from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


DEFAULT_ERROR_STOPWORDS = {
    "ante",
    "como",
    "con",
    "del",
    "desde",
    "donde",
    "el",
    "ella",
    "ellos",
    "en",
    "era",
    "es",
    "esa",
    "ese",
    "esta",
    "este",
    "fue",
    "hay",
    "la",
    "las",
    "le",
    "les",
    "lo",
    "los",
    "mas",
    "muy",
    "no",
    "para",
    "pero",
    "por",
    "porque",
    "que",
    "se",
    "sin",
    "son",
    "su",
    "sus",
    "una",
    "uno",
}


def normalize_text_for_model(text: object) -> str:
    """Normalize social-media text while preserving lexical content."""
    if text is None:
        return ""
    try:
        if pd.isna(text):
            return ""
    except (TypeError, ValueError):
        pass
    value = str(text).lower().strip()
    value = re.sub(r"https?://\S+|www\.\S+", " ", value)
    value = re.sub(r"@\w+", " ", value)
    value = re.sub(r"#(\w+)", r"\1", value)
    value = unicodedata.normalize("NFD", value)
    value = "".join(
        character
        for character in value
        if unicodedata.category(character) != "Mn"
    )
    return re.sub(r"\s+", " ", value).strip()


def _binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_positive": float(
            precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "recall_positive": float(
            recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "f1_positive": float(
            f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "average_precision": float(average_precision_score(y_true, scores)),
    }


def build_logreg_tfidf_pipeline(random_state: int = 42) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_linearsvc_tfidf_pipeline(random_state: int = 42) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LinearSVC(
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_word_char_logreg_pipeline(
    random_state: int = 42,
    c_value: float = 1.0,
) -> Pipeline:
    """Build a transparent word+character TF-IDF Logistic Regression model."""
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98,
                    sublinear_tf=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
        ]
    )
    return Pipeline(
        [
            ("features", features),
            (
                "clf",
                LogisticRegression(
                    C=c_value,
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )


def run_repeated_stratified_group_text_cv(
    data: pd.DataFrame,
    text_column: str,
    target_column: str,
    group_column: str,
    id_columns: Optional[Sequence[str]] = None,
    lexicon_prediction_column: Optional[str] = None,
    include_models: Optional[Sequence[str]] = None,
    n_splits: int = 5,
    n_repeats: int = 5,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare word and word+character models while keeping groups isolated."""
    working = data.reset_index(drop=True).copy()
    required = {text_column, target_column, group_column}
    missing = required.difference(working.columns)
    if missing:
        raise KeyError(f"Missing grouped CV columns: {sorted(missing)}")

    working[text_column] = working[text_column].fillna("").astype(str)
    working[target_column] = pd.to_numeric(
        working[target_column], errors="coerce"
    )
    if not working[target_column].isin([0, 1]).all():
        raise ValueError("Grouped CV target must contain complete binary labels")
    working[group_column] = working[group_column].fillna(
        working.index.to_series().map(lambda index: f"missing_group_{index}")
    ).astype(str)

    texts = working[text_column].to_numpy()
    y = working[target_column].astype(int).to_numpy()
    groups = working[group_column].to_numpy()
    model_names = tuple(
        include_models or ("word_logreg", "word_char_logreg")
    )
    supported_models = {
        "lexicon_hits",
        "word_logreg",
        "logreg_tfidf",
        "linearsvc_tfidf",
        "word_char_logreg",
    }
    unsupported_models = set(model_names).difference(supported_models)
    if unsupported_models:
        raise ValueError(
            f"Unsupported grouped CV models: {sorted(unsupported_models)}"
        )
    if "lexicon_hits" in model_names:
        if not lexicon_prediction_column:
            raise ValueError(
                "lexicon_prediction_column is required for lexicon_hits"
            )
        if lexicon_prediction_column not in working.columns:
            raise KeyError(
                f"Missing lexicon prediction column: {lexicon_prediction_column}"
            )
        lexicon_predictions = (
            pd.to_numeric(
                working[lexicon_prediction_column], errors="coerce"
            )
            .fillna(0)
            .gt(0)
            .astype(int)
            .to_numpy()
        )
    else:
        lexicon_predictions = np.zeros(len(working), dtype=int)
    class_counts = pd.Series(y).value_counts()
    if len(class_counts) != 2 or int(class_counts.min()) < n_splits:
        raise ValueError("Both classes need at least n_splits rows")

    fold_rows: List[Dict[str, object]] = []
    prediction_rows: List[Dict[str, object]] = []
    for repeat in range(1, n_repeats + 1):
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state + repeat - 1,
        )
        for fold, (train_index, test_index) in enumerate(
            splitter.split(texts, y, groups), start=1
        ):
            X_train = texts[train_index]
            X_test = texts[test_index]
            y_train = y[train_index]
            y_test = y[test_index]
            train_groups = set(groups[train_index])
            test_groups = set(groups[test_index])
            if train_groups.intersection(test_groups):
                raise ValueError("Group leakage detected in grouped CV")

            for model_name in model_names:
                seed = random_state + repeat * 100 + fold
                if model_name == "lexicon_hits":
                    predictions = lexicon_predictions[test_index]
                    scores = predictions.astype(float)
                    score_threshold = 0.5
                elif model_name in {"word_logreg", "logreg_tfidf"}:
                    model = build_logreg_tfidf_pipeline(random_state=seed)
                    model.fit(X_train, y_train)
                    scores = model.predict_proba(X_test)[:, 1]
                    predictions = (scores >= 0.5).astype(int)
                    score_threshold = 0.5
                elif model_name == "linearsvc_tfidf":
                    model = build_linearsvc_tfidf_pipeline(random_state=seed)
                    model.fit(X_train, y_train)
                    scores = model.decision_function(X_test)
                    predictions = (scores >= 0.0).astype(int)
                    score_threshold = 0.0
                else:
                    model = build_word_char_logreg_pipeline(random_state=seed)
                    model.fit(X_train, y_train)
                    scores = model.predict_proba(X_test)[:, 1]
                    predictions = (scores >= 0.5).astype(int)
                    score_threshold = 0.5
                metrics = _binary_metrics(y_test, predictions, scores)
                fold_rows.append(
                    {
                        "model": model_name,
                        "repeat": repeat,
                        "fold": fold,
                        "n_train": len(train_index),
                        "n_test": len(test_index),
                        "n_train_groups": len(train_groups),
                        "n_test_groups": len(test_groups),
                        "n_positive_train": int(y_train.sum()),
                        "n_positive_test": int(y_test.sum()),
                        **metrics,
                    }
                )
                for position, row_position in enumerate(test_index):
                    prediction_rows.append(
                        {
                            "model": model_name,
                            "repeat": repeat,
                            "fold": fold,
                            "row_position": int(row_position),
                            "y_true": int(y_test[position]),
                            "y_pred": int(predictions[position]),
                            "score": float(scores[position]),
                            "score_threshold": score_threshold,
                        }
                    )

    fold_metrics = pd.DataFrame(fold_rows)
    metric_columns = [
        "accuracy",
        "precision_positive",
        "recall_positive",
        "f1_positive",
        "macro_f1",
        "balanced_accuracy",
        "average_precision",
    ]
    summary = (
        fold_metrics.groupby("model", as_index=False)[metric_columns]
        .agg(["mean", "std", "min", "max"])
    )
    summary.columns = [
        column if isinstance(column, str) else "_".join(column).strip("_")
        for column in summary.columns.to_flat_index()
    ]
    summary.insert(1, "n_splits", n_splits)
    summary.insert(2, "n_repeats", n_repeats)
    summary.insert(3, "n_evaluations", n_splits * n_repeats)
    summary = summary.sort_values(
        ["f1_positive_mean", "macro_f1_mean"], ascending=False
    ).reset_index(drop=True)

    prediction_long = pd.DataFrame(prediction_rows)
    consensus = (
        prediction_long.groupby(["model", "row_position"], as_index=False)
        .agg(
            y_true=("y_true", "first"),
            cv_pred_rate=("y_pred", "mean"),
            cv_score_mean=("score", "mean"),
            cv_score_std=("score", "std"),
            score_threshold=("score_threshold", "first"),
            n_evaluations=("score", "size"),
        )
    )
    consensus["cv_consensus_pred_default"] = (
        consensus["cv_score_mean"] >= consensus["score_threshold"]
    ).astype(int)
    for column in id_columns or []:
        if column in working.columns:
            consensus[column] = consensus["row_position"].map(working[column])
    return fold_metrics, summary, consensus


def select_threshold_for_min_precision(
    y_true: Sequence[int],
    scores: Sequence[float],
    min_precision: float = 0.80,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Choose the highest-recall threshold meeting a precision constraint."""
    y_array = np.asarray(y_true, dtype=int)
    score_array = np.asarray(scores, dtype=float)
    precision_values, recall_values, thresholds = precision_recall_curve(
        y_array, score_array
    )
    rows: List[Dict[str, object]] = []
    candidate_thresholds = sorted(set(thresholds.tolist() + [0.5]))
    for threshold in candidate_thresholds:
        predictions = (score_array >= threshold).astype(int)
        metrics = _binary_metrics(y_array, predictions, score_array)
        rows.append(
            {
                "threshold": float(threshold),
                "predicted_positive": int(predictions.sum()),
                "meets_min_precision": bool(
                    metrics["precision_positive"] >= min_precision
                ),
                **metrics,
            }
        )
    table = pd.DataFrame(rows)
    eligible = table[table["meets_min_precision"]].copy()
    selection_rule = "max_recall_at_min_precision"
    if eligible.empty:
        eligible = table.copy()
        selection_rule = "fallback_max_f1"
    selected = eligible.sort_values(
        ["recall_positive", "f1_positive", "precision_positive", "threshold"],
        ascending=[False, False, False, False],
    ).iloc[0]
    result: Dict[str, object] = selected.to_dict()
    result["min_precision_requested"] = min_precision
    result["selection_rule"] = selection_rule
    return table.sort_values("threshold").reset_index(drop=True), result


def select_threshold_max_f1(
    y_true: Sequence[int],
    scores: Sequence[float],
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Choose an exploratory operating threshold that maximizes positive F1."""
    table, _ = select_threshold_for_min_precision(
        y_true,
        scores,
        min_precision=0.0,
    )
    selected = table.sort_values(
        ["f1_positive", "macro_f1", "balanced_accuracy", "threshold"],
        ascending=[False, False, False, False],
    ).iloc[0]
    result: Dict[str, object] = selected.to_dict()
    result["min_precision_requested"] = None
    result["selection_rule"] = "max_f1_positive"
    return table, result


def run_repeated_stratified_text_cv(
    data: pd.DataFrame,
    text_column: str,
    target_column: str,
    id_columns: Optional[Sequence[str]] = None,
    lexicon_prediction_column: Optional[str] = None,
    n_splits: int = 5,
    n_repeats: int = 5,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate transparent binary text baselines with repeated stratified CV."""
    working = data.reset_index(drop=True).copy()
    if text_column not in working.columns or target_column not in working.columns:
        raise KeyError("Missing text or target column for cross-validation")

    working[text_column] = working[text_column].fillna("").astype(str)
    working[target_column] = pd.to_numeric(
        working[target_column], errors="coerce"
    )
    if not working[target_column].isin([0, 1]).all():
        raise ValueError("Target must contain only complete binary labels")

    y = working[target_column].astype(int).to_numpy()
    class_counts = pd.Series(y).value_counts()
    if len(class_counts) != 2:
        raise ValueError("Both target classes are required")
    if int(class_counts.min()) < n_splits:
        raise ValueError(
            "The minority class has fewer rows than the requested CV folds"
        )

    texts = working[text_column].to_numpy()
    if lexicon_prediction_column and lexicon_prediction_column in working.columns:
        lexicon_predictions = (
            pd.to_numeric(
                working[lexicon_prediction_column], errors="coerce"
            )
            .fillna(0)
            .gt(0)
            .astype(int)
            .to_numpy()
        )
    else:
        lexicon_predictions = np.zeros(len(working), dtype=int)

    splitter = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    fold_rows: List[Dict[str, object]] = []
    prediction_rows: List[Dict[str, object]] = []

    for split_index, (train_index, test_index) in enumerate(
        splitter.split(texts, y)
    ):
        repeat = split_index // n_splits + 1
        fold = split_index % n_splits + 1
        X_train = texts[train_index]
        X_test = texts[test_index]
        y_train = y[train_index]
        y_test = y[test_index]

        model_outputs: Dict[str, Tuple[np.ndarray, np.ndarray, float]] = {}

        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(np.zeros((len(train_index), 1)), y_train)
        dummy_pred = dummy.predict(np.zeros((len(test_index), 1))).astype(int)
        model_outputs["dummy_most_frequent"] = (
            dummy_pred,
            dummy_pred.astype(float),
            0.5,
        )

        lexicon_pred = lexicon_predictions[test_index]
        model_outputs["lexicon_hits"] = (
            lexicon_pred,
            lexicon_pred.astype(float),
            0.5,
        )

        logreg = build_logreg_tfidf_pipeline(
            random_state=random_state + split_index
        )
        logreg.fit(X_train, y_train)
        model_outputs["logreg_tfidf"] = (
            logreg.predict(X_test).astype(int),
            logreg.predict_proba(X_test)[:, 1],
            0.5,
        )

        svc = build_linearsvc_tfidf_pipeline(
            random_state=random_state + split_index
        )
        svc.fit(X_train, y_train)
        model_outputs["linearsvc_tfidf"] = (
            svc.predict(X_test).astype(int),
            svc.decision_function(X_test),
            0.0,
        )

        for model_name, (predictions, scores, threshold) in model_outputs.items():
            metrics = _binary_metrics(y_test, predictions, scores)
            fold_rows.append(
                {
                    "model": model_name,
                    "repeat": repeat,
                    "fold": fold,
                    "n_train": len(train_index),
                    "n_test": len(test_index),
                    "n_positive_train": int(y_train.sum()),
                    "n_positive_test": int(y_test.sum()),
                    **metrics,
                }
            )

            for position, row_position in enumerate(test_index):
                prediction_rows.append(
                    {
                        "model": model_name,
                        "repeat": repeat,
                        "fold": fold,
                        "row_position": int(row_position),
                        "y_true": int(y_test[position]),
                        "y_pred": int(predictions[position]),
                        "score": float(scores[position]),
                        "score_threshold": threshold,
                    }
                )

    fold_metrics = pd.DataFrame(fold_rows)
    metric_columns = [
        "accuracy",
        "precision_positive",
        "recall_positive",
        "f1_positive",
        "macro_f1",
        "balanced_accuracy",
        "average_precision",
    ]
    summary = (
        fold_metrics.groupby("model", as_index=False)[metric_columns]
        .agg(["mean", "std", "min", "max"])
    )
    summary.columns = [
        column if isinstance(column, str) else "_".join(column).strip("_")
        for column in summary.columns.to_flat_index()
    ]
    summary = summary.sort_values(
        ["f1_positive_mean", "macro_f1_mean"], ascending=False
    ).reset_index(drop=True)
    summary.insert(1, "n_splits", n_splits)
    summary.insert(2, "n_repeats", n_repeats)
    summary.insert(3, "n_evaluations", n_splits * n_repeats)

    prediction_long = pd.DataFrame(prediction_rows)
    consensus = (
        prediction_long.groupby(["model", "row_position"], as_index=False)
        .agg(
            y_true=("y_true", "first"),
            cv_pred_rate=("y_pred", "mean"),
            cv_score_mean=("score", "mean"),
            cv_score_std=("score", "std"),
            score_threshold=("score_threshold", "first"),
            n_evaluations=("y_pred", "size"),
        )
    )
    consensus["cv_consensus_pred"] = (
        consensus["cv_score_mean"] >= consensus["score_threshold"]
    ).astype(int)

    for column in id_columns or []:
        if column in working.columns:
            consensus[column] = consensus["row_position"].map(
                working[column]
            )

    return fold_metrics, summary, consensus


def _normalize_for_error_tokens(text: object) -> str:
    value = str(text or "").lower()
    value = re.sub(r"https?://\S+|www\.\S+|@\w+", " ", value)
    value = unicodedata.normalize("NFD", value)
    value = "".join(
        character
        for character in value
        if unicodedata.category(character) != "Mn"
    )
    return re.sub(r"\s+", " ", value).strip()


def _top_error_terms(
    errors: pd.DataFrame,
    text_column: str,
    top_n: int = 30,
    stopwords: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    stopword_set = set(stopwords or DEFAULT_ERROR_STOPWORDS)
    rows = []
    for error_type, part in errors.groupby("error_type"):
        counter: Counter = Counter()
        for text in part[text_column].fillna(""):
            tokens = re.findall(
                r"\b[a-záéíóúüñ]{3,}\b",
                _normalize_for_error_tokens(text),
            )
            counter.update(
                token for token in tokens if token not in stopword_set
            )
        rows.extend(
            {
                "error_type": error_type,
                "term": term,
                "count": count,
            }
            for term, count in counter.most_common(top_n)
        )
    return pd.DataFrame(rows, columns=["error_type", "term", "count"])


def build_hostility_error_diagnostics(
    predictions: pd.DataFrame,
    text_column: str = "text",
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Build descriptive tables for false positives and false negatives."""
    required = {"y_true", "y_pred", text_column}
    missing = required.difference(predictions.columns)
    if missing:
        raise KeyError(f"Missing prediction columns: {sorted(missing)}")

    working = predictions.copy()
    working["y_true"] = pd.to_numeric(working["y_true"], errors="raise").astype(int)
    working["y_pred"] = pd.to_numeric(working["y_pred"], errors="raise").astype(int)
    errors = working[working["y_true"] != working["y_pred"]].copy()
    errors["error_type"] = np.where(
        (errors["y_true"] == 0) & (errors["y_pred"] == 1),
        "false_positive",
        "false_negative",
    )
    errors["manual_error_notes"] = ""

    if "model_score" in errors.columns:
        errors["model_score"] = pd.to_numeric(
            errors["model_score"], errors="coerce"
        )
        errors["distance_from_boundary"] = errors["model_score"].abs()
        errors["near_decision_boundary"] = (
            errors["distance_from_boundary"] <= 0.20
        )
    else:
        errors["model_score"] = np.nan
        errors["distance_from_boundary"] = np.nan
        errors["near_decision_boundary"] = False

    if "categorized_lexicon_hit_count" in errors.columns:
        errors["categorized_lexicon_hit_count"] = pd.to_numeric(
            errors["categorized_lexicon_hit_count"], errors="coerce"
        ).fillna(0)
    else:
        errors["categorized_lexicon_hit_count"] = 0
    errors["has_categorized_lexicon_match"] = (
        errors["categorized_lexicon_hit_count"] > 0
    )

    summary = pd.DataFrame(
        [
            {"metric": "n_test", "value": len(working)},
            {
                "metric": "n_correct",
                "value": int((working["y_true"] == working["y_pred"]).sum()),
            },
            {"metric": "n_errors", "value": len(errors)},
            {
                "metric": "false_positives",
                "value": int((errors["error_type"] == "false_positive").sum()),
            },
            {
                "metric": "false_negatives",
                "value": int((errors["error_type"] == "false_negative").sum()),
            },
            {
                "metric": "error_rate_pct",
                "value": round(100 * len(errors) / len(working), 2)
                if len(working)
                else 0.0,
            },
        ]
    )

    by_type = (
        errors.groupby("error_type", as_index=False)
        .agg(
            n_rows=("y_true", "size"),
            mean_model_score=("model_score", "mean"),
            median_model_score=("model_score", "median"),
            near_boundary_rows=("near_decision_boundary", "sum"),
            rows_with_lexicon_match=("has_categorized_lexicon_match", "sum"),
            mean_lexicon_hits=("categorized_lexicon_hit_count", "mean"),
        )
        .sort_values("error_type")
    )

    tables: Dict[str, pd.DataFrame] = {
        "summary": summary,
        "by_type": by_type,
        "top_terms": _top_error_terms(errors, text_column=text_column),
    }
    for output_name, column in [
        ("by_event", "event_id"),
        ("by_media", "anchor_media_handle"),
        ("by_level", "hostility_relevance_normalized"),
        ("by_hate_label", "y_hate_speech"),
    ]:
        if column in errors.columns:
            tables[output_name] = (
                errors.groupby(["error_type", column], dropna=False)
                .size()
                .reset_index(name="n_rows")
                .sort_values(["error_type", "n_rows"], ascending=[True, False])
            )

    sort_columns = ["error_type", "distance_from_boundary"]
    errors = errors.sort_values(sort_columns, ascending=[True, True])
    return errors, tables

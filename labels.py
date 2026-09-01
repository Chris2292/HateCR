from __future__ import annotations

from typing import Any, Dict, Optional, Set, Tuple

import pandas as pd


CANONICAL_LABEL_COLUMNS = [
    "hostility_relevance",
    "manual_hostility",
    "manual_hate_speech",
]
HUMAN_LABEL_COLUMNS = CANONICAL_LABEL_COLUMNS + ["notes"]

ALLOWED_HOSTILITY_RELEVANCE = {0, 1, 2, 3}
ALLOWED_BINARY_LABELS = {0, 1}

LEVEL_NAMES = {
    0: "not_offensive",
    1: "incivility_disqualification",
    2: "violent_hostility_non_identity",
    3: "identity_cultural_ideological_attack",
}

def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return not str(value).strip()


def _normalize_integer(value: Any, allowed: Set[int]) -> Optional[int]:
    if _is_missing(value):
        return None
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if not number.is_integer():
        return None
    integer = int(number)
    return integer if integer in allowed else None


def _normalize_binary(value: Any) -> Optional[int]:
    return _normalize_integer(value, ALLOWED_BINARY_LABELS)


def normalize_hostility_relevance(value: Any) -> Optional[int]:
    return _normalize_integer(value, ALLOWED_HOSTILITY_RELEVANCE)


def normalize_manual_hostility(value: Any) -> Optional[int]:
    """Normalize 0=no hostility found and 1=hostility found."""
    return _normalize_binary(value)


def normalize_manual_hate_speech(value: Any) -> Optional[int]:
    """Normalize 0=no hate found and 1=hate found."""
    return _normalize_binary(value)


def _provided_but_invalid(raw_value: Any, normalized_value: Any) -> bool:
    return not _is_missing(raw_value) and normalized_value is None


def _nullable_int(values: list) -> pd.arrays.IntegerArray:
    return pd.array(
        [pd.NA if value is None else value for value in values],
        dtype="Int64",
    )


def prepare_manual_annotations(
    annotations_df: pd.DataFrame,
    strict: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Normalize three human labels and flag inconsistent combinations.

    Human values are preserved. The function never replaces them with values
    inferred from the four-level taxonomy.
    """
    output = annotations_df.copy()
    for column in HUMAN_LABEL_COLUMNS:
        if column not in output.columns:
            output[column] = pd.NA

    levels = [
        normalize_hostility_relevance(value)
        for value in output["hostility_relevance"]
    ]
    hostility_values = [
        normalize_manual_hostility(value)
        for value in output["manual_hostility"]
    ]
    hate_values = [
        normalize_manual_hate_speech(value)
        for value in output["manual_hate_speech"]
    ]

    output["hostility_relevance_normalized"] = _nullable_int(levels)
    output["manual_hostility_normalized"] = _nullable_int(hostility_values)
    output["manual_hate_speech_normalized"] = _nullable_int(hate_values)
    output["hostility_level_label"] = pd.array(
        [pd.NA if level is None else LEVEL_NAMES[level] for level in levels],
        dtype="string",
    )

    expected_hostility = [
        None if level is None else int(level >= 1)
        for level in levels
    ]
    expected_hate = [
        None if level is None else int(level == 3)
        for level in levels
    ]
    severe_or_identity = [
        None if level is None else int(level >= 2)
        for level in levels
    ]

    output["manual_hostility_from_relevance"] = _nullable_int(
        expected_hostility
    )
    output["manual_hate_speech_from_relevance"] = _nullable_int(
        expected_hate
    )

    errors = []
    warnings = []
    for row_position, (_, row) in enumerate(output.iterrows()):
        row_errors = []
        row_warnings = []
        level = levels[row_position]
        hostility = hostility_values[row_position]
        hate = hate_values[row_position]

        if _provided_but_invalid(row["hostility_relevance"], level):
            row_errors.append("invalid_hostility_relevance")
        if _provided_but_invalid(row["manual_hostility"], hostility):
            row_errors.append("invalid_manual_hostility_binary")
        if _provided_but_invalid(row["manual_hate_speech"], hate):
            row_errors.append("invalid_manual_hate_speech_binary")

        if level is not None and hostility is not None:
            if hostility != int(level >= 1):
                row_warnings.append("hostility_level_pattern_review")
        if level is not None and hate is not None:
            if hate != int(level == 3):
                row_warnings.append("hate_level_pattern_review")
        if hate == 1 and hostility == 0:
            row_errors.append("hate_requires_hostility")

        errors.append("|".join(row_errors))
        warnings.append("|".join(row_warnings))

    output["annotation_errors"] = errors
    output["annotation_warnings"] = warnings
    output["annotation_complete"] = (
        output["hostility_relevance_normalized"].notna()
        & output["manual_hostility_normalized"].notna()
        & output["manual_hate_speech_normalized"].notna()
    )
    output["annotation_valid"] = output["annotation_errors"].eq("")
    output["annotation_ready_for_training"] = (
        output["annotation_complete"] & output["annotation_valid"]
    )

    # Stable targets retain the independent human binary decisions.
    output["y_hostility"] = output["manual_hostility_normalized"]
    output["y_hate_speech"] = output["manual_hate_speech_normalized"]
    output["y_hostility_multiclass"] = output[
        "hostility_relevance_normalized"
    ]
    output["y_severe_or_identity"] = _nullable_int(severe_or_identity)

    # Backward-compatible aliases used by older analysis exports.
    output["y_hostility_intensity"] = output["y_hostility_multiclass"]
    output["manual_hate_speech_binary"] = output["y_hate_speech"]

    any_value_entered = pd.Series(False, index=output.index)
    for column in CANONICAL_LABEL_COLUMNS:
        any_value_entered = any_value_entered | output[column].map(
            lambda value: not _is_missing(value)
        )
    invalid_entered = any_value_entered & ~output["annotation_valid"]

    summary = pd.DataFrame(
        [
            {"metric": "rows", "value": len(output)},
            {
                "metric": "rows_with_any_label",
                "value": int(any_value_entered.sum()),
            },
            {
                "metric": "complete_annotations",
                "value": int(output["annotation_complete"].sum()),
            },
            {
                "metric": "valid_complete_annotations",
                "value": int(output["annotation_ready_for_training"].sum()),
            },
            {
                "metric": "rows_with_errors",
                "value": int(invalid_entered.sum()),
            },
            {
                "metric": "rows_with_warnings",
                "value": int(output["annotation_warnings"].ne("").sum()),
            },
            {
                "metric": "hostility_relevance_filled",
                "value": int(
                    output["hostility_relevance_normalized"].notna().sum()
                ),
            },
            {
                "metric": "manual_hostility_filled",
                "value": int(
                    output["manual_hostility_normalized"].notna().sum()
                ),
            },
            {
                "metric": "manual_hate_speech_filled",
                "value": int(
                    output["manual_hate_speech_normalized"].notna().sum()
                ),
            },
        ]
    )

    issue_columns = [
        column
        for column in [
            "review_id",
            "tweet_id",
            "event_id",
            "hostility_relevance",
            "manual_hostility",
            "manual_hate_speech",
            "annotation_errors",
            "annotation_warnings",
        ]
        if column in output.columns
    ]
    issues = output[
        output["annotation_errors"].ne("")
        | output["annotation_warnings"].ne("")
    ][issue_columns].copy()

    level_distribution = (
        output["hostility_relevance_normalized"]
        .value_counts(dropna=False)
        .rename_axis("hostility_relevance")
        .reset_index(name="n_rows")
    )
    hostility_distribution = (
        output["manual_hostility_normalized"]
        .value_counts(dropna=False)
        .rename_axis("manual_hostility")
        .reset_index(name="n_rows")
    )
    hate_distribution = (
        output["manual_hate_speech_normalized"]
        .value_counts(dropna=False)
        .rename_axis("manual_hate_speech")
        .reset_index(name="n_rows")
    )

    diagnostics = {
        "summary": summary,
        "issues": issues,
        "level_distribution": level_distribution,
        "intensity_distribution": level_distribution.copy(),
        "hostility_distribution": hostility_distribution,
        "hate_speech_distribution": hate_distribution,
    }

    if strict and int(invalid_entered.sum()) > 0:
        raise ValueError(
            "Manual annotations contain "
            f"{int(invalid_entered.sum())} invalid rows"
        )

    return output, diagnostics

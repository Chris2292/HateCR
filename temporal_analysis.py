"""Temporal analysis helpers for the formal HateCR media-anchored corpus."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch


PRIMARY_PHASES = [
    "pre_24h",
    "during",
    "post_0_6h",
    "post_6_24h",
    "post_24_72h",
]

PHASE_LABELS = {
    "pre_24h": "Antes\n(-24 a 0 h)",
    "during": "Durante",
    "post_0_6h": "Después\n(0-6 h)",
    "post_6_24h": "Después\n(6-24 h)",
    "post_24_72h": "Después\n(24-72 h)",
    "outside": "Fuera del rango",
}

PHASE_DURATION_HOURS = {
    "pre_24h": 24.0,
    "post_0_6h": 6.0,
    "post_6_24h": 18.0,
    "post_24_72h": 48.0,
}


def _timestamp_to_utc(value: object, timezone: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        return pd.NaT
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(timezone)
    else:
        timestamp = timestamp.tz_convert(timezone)
    return timestamp.tz_convert("UTC")


def _validate_columns(frame: pd.DataFrame, required: Sequence[str], name: str) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise KeyError(f"Faltan columnas en {name}: {missing}")


def load_event_schedule(
    events_path: Union[str, Path],
    active_only: bool = True,
    formal_only: bool = True,
) -> pd.DataFrame:
    """Load event boundaries and normalize every timestamp to UTC."""
    path = Path(events_path)
    if not path.exists():
        raise FileNotFoundError(f"No existe la configuración de eventos: {path}")

    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rows = []
    for event in config.get("events", []):
        if active_only and not bool(event.get("active", False)):
            continue
        if formal_only and not bool(event.get("formal", False)):
            continue

        timezone = str(event.get("timezone") or "America/Costa_Rica")
        collection_window = event.get("collection_window") or {}
        required = ["event_id", "event_name", "event_start_local", "event_end_local"]
        missing = [field for field in required if not event.get(field)]
        if missing:
            raise ValueError(
                f"El evento {event.get('event_id', '<sin id>')} carece de {missing}"
            )

        event_start_utc = _timestamp_to_utc(event["event_start_local"], timezone)
        event_end_utc = _timestamp_to_utc(event["event_end_local"], timezone)
        if event_end_utc <= event_start_utc:
            raise ValueError(f"Ventana inválida para {event['event_id']}")

        collection_start = collection_window.get("start")
        collection_end = collection_window.get("end")
        rows.append(
            {
                "event_id": str(event["event_id"]),
                "event_name": str(event["event_name"]),
                "formal_order": int(event.get("formal_order", 999)),
                "event_date": str(event.get("event_date") or ""),
                "event_type": str(event.get("event_type") or ""),
                "timezone": timezone,
                "event_start_utc": event_start_utc,
                "event_end_utc": event_end_utc,
                "collection_start_utc": (
                    _timestamp_to_utc(collection_start, timezone)
                    if collection_start
                    else pd.NaT
                ),
                "collection_end_utc": (
                    _timestamp_to_utc(collection_end, timezone)
                    if collection_end
                    else pd.NaT
                ),
                "event_duration_hours": (
                    event_end_utc - event_start_utc
                ).total_seconds()
                / 3600.0,
            }
        )

    if not rows:
        raise ValueError("No se encontraron eventos activos y formales")

    schedule = pd.DataFrame(rows).sort_values("formal_order").reset_index(drop=True)
    if schedule["event_id"].duplicated().any():
        duplicated = schedule.loc[
            schedule["event_id"].duplicated(keep=False), "event_id"
        ].tolist()
        raise ValueError(f"event_id duplicados en configuración: {duplicated}")
    return schedule


def event_window_overlaps(schedule: pd.DataFrame) -> pd.DataFrame:
    """Return pairwise overlaps between configured collection windows."""
    _validate_columns(
        schedule,
        [
            "event_id",
            "event_name",
            "formal_order",
            "collection_start_utc",
            "collection_end_utc",
        ],
        "schedule",
    )
    rows = []
    ordered = schedule.sort_values("formal_order").reset_index(drop=True)
    for left_index in range(len(ordered)):
        for right_index in range(left_index + 1, len(ordered)):
            left = ordered.iloc[left_index]
            right = ordered.iloc[right_index]
            if pd.isna(left["collection_start_utc"]) or pd.isna(
                left["collection_end_utc"]
            ):
                continue
            if pd.isna(right["collection_start_utc"]) or pd.isna(
                right["collection_end_utc"]
            ):
                continue
            overlap_start = max(
                left["collection_start_utc"], right["collection_start_utc"]
            )
            overlap_end = min(
                left["collection_end_utc"], right["collection_end_utc"]
            )
            overlap_hours = max(
                0.0, (overlap_end - overlap_start).total_seconds() / 3600.0
            )
            if overlap_hours <= 0:
                continue
            rows.append(
                {
                    "event_id_a": left["event_id"],
                    "event_name_a": left["event_name"],
                    "event_id_b": right["event_id"],
                    "event_name_b": right["event_name"],
                    "overlap_start_utc": overlap_start,
                    "overlap_end_utc": overlap_end,
                    "overlap_hours": overlap_hours,
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "event_id_a",
            "event_name_a",
            "event_id_b",
            "event_name_b",
            "overlap_start_utc",
            "overlap_end_utc",
            "overlap_hours",
        ],
    )


def prepare_temporal_corpus(
    corpus: pd.DataFrame,
    schedule: pd.DataFrame,
    prediction_column: str = "ml_hostility_pred_v2_balanced",
    created_at_column: str = "created_at",
    display_timezone: str = "America/Costa_Rica",
    pre_event_hours: float = 24.0,
    max_post_event_hours: float = 72.0,
) -> pd.DataFrame:
    """Attach event-relative time and a mutually exclusive temporal phase."""
    _validate_columns(
        corpus,
        ["tweet_id", "event_id", created_at_column, prediction_column],
        "corpus",
    )
    _validate_columns(
        schedule,
        [
            "event_id",
            "event_name",
            "formal_order",
            "event_start_utc",
            "event_end_utc",
            "event_duration_hours",
        ],
        "schedule",
    )

    work = corpus.copy()
    work["tweet_id"] = work["tweet_id"].astype("string")
    work["created_at_utc"] = pd.to_datetime(
        work[created_at_column], errors="coerce", utc=True
    )
    work["created_at_cr"] = work["created_at_utc"].dt.tz_convert(display_timezone)
    work["hostility_pred"] = pd.to_numeric(
        work[prediction_column], errors="coerce"
    ).fillna(0).astype(int)

    schedule_columns = [
        "event_id",
        "event_name",
        "formal_order",
        "event_date",
        "event_type",
        "timezone",
        "event_start_utc",
        "event_end_utc",
        "collection_start_utc",
        "collection_end_utc",
        "event_duration_hours",
    ]
    schedule_for_merge = schedule[schedule_columns].rename(
        columns={"event_name": "event_name_config"}
    )
    work = work.merge(schedule_for_merge, on="event_id", how="left")
    if "event_name" not in work.columns:
        work["event_name"] = work["event_name_config"]
    else:
        work["event_name"] = work["event_name"].fillna(work["event_name_config"])

    work["hours_from_event_start"] = (
        work["created_at_utc"] - work["event_start_utc"]
    ).dt.total_seconds() / 3600.0
    work["hours_from_event_end"] = (
        work["created_at_utc"] - work["event_end_utc"]
    ).dt.total_seconds() / 3600.0

    pre = work["hours_from_event_start"].ge(-pre_event_hours) & work[
        "hours_from_event_start"
    ].lt(0)
    during = work["created_at_utc"].ge(work["event_start_utc"]) & work[
        "created_at_utc"
    ].le(work["event_end_utc"])
    post_0_6 = work["hours_from_event_end"].gt(0) & work[
        "hours_from_event_end"
    ].le(6)
    post_6_24 = work["hours_from_event_end"].gt(6) & work[
        "hours_from_event_end"
    ].le(24)
    post_24_72 = work["hours_from_event_end"].gt(24) & work[
        "hours_from_event_end"
    ].le(max_post_event_hours)

    work["temporal_phase"] = np.select(
        [pre, during, post_0_6, post_6_24, post_24_72],
        PRIMARY_PHASES,
        default="outside",
    )
    work["within_primary_temporal_window"] = work["temporal_phase"].ne("outside")

    missing_timestamp = work["created_at_utc"].isna()
    missing_event = work["event_start_utc"].isna() | work["event_end_utc"].isna()
    before_window = work["hours_from_event_start"].lt(-pre_event_hours)
    after_window = work["hours_from_event_end"].gt(max_post_event_hours)
    work["outside_reason"] = np.select(
        [missing_timestamp, missing_event, before_window, after_window],
        [
            "missing_created_at",
            "event_not_configured",
            "before_pre_event_window",
            "after_post_event_window",
        ],
        default="",
    )
    work["pre_event_hours_config"] = float(pre_event_hours)
    work["max_post_event_hours_config"] = float(max_post_event_hours)
    return work


def _add_wilson_interval(
    frame: pd.DataFrame,
    positive_column: str,
    total_column: str,
    prefix: str = "hostility",
    z_value: float = 1.96,
) -> pd.DataFrame:
    result = frame.copy()
    positives = pd.to_numeric(result[positive_column], errors="coerce").to_numpy(
        dtype=float
    )
    totals = pd.to_numeric(result[total_column], errors="coerce").to_numpy(
        dtype=float
    )
    low = np.full(len(result), np.nan)
    high = np.full(len(result), np.nan)
    valid = totals > 0
    proportions = np.zeros(len(result), dtype=float)
    proportions[valid] = positives[valid] / totals[valid]
    denominator = 1 + (z_value**2) / totals[valid]
    centre = (
        proportions[valid] + (z_value**2) / (2 * totals[valid])
    ) / denominator
    margin = (
        z_value
        * np.sqrt(
            proportions[valid] * (1 - proportions[valid]) / totals[valid]
            + (z_value**2) / (4 * totals[valid] ** 2)
        )
        / denominator
    )
    low[valid] = 100 * (centre - margin)
    high[valid] = 100 * (centre + margin)
    result[f"{prefix}_ci95_low"] = low
    result[f"{prefix}_ci95_high"] = high
    return result


def summarize_temporal_phases(
    prepared: pd.DataFrame,
    min_phase_comments: int = 20,
) -> pd.DataFrame:
    """Summarize hostility and volume for before/during/after phases."""
    _validate_columns(
        prepared,
        [
            "tweet_id",
            "event_id",
            "event_name",
            "formal_order",
            "event_date",
            "event_duration_hours",
            "temporal_phase",
            "hostility_pred",
        ],
        "prepared corpus",
    )
    work = prepared.drop_duplicates("tweet_id").copy()
    event_meta = (
        work[
            [
                "event_id",
                "event_name",
                "formal_order",
                "event_date",
                "event_duration_hours",
            ]
        ]
        .drop_duplicates("event_id")
        .sort_values("formal_order")
    )
    phases = pd.DataFrame(
        {"temporal_phase": PRIMARY_PHASES, "phase_order": range(len(PRIMARY_PHASES))}
    )
    grid = event_meta.assign(_key=1).merge(phases.assign(_key=1), on="_key").drop(
        columns="_key"
    )

    observed = (
        work.loc[work["temporal_phase"].isin(PRIMARY_PHASES)]
        .groupby(["event_id", "temporal_phase"], as_index=False)
        .agg(
            comment_count=("tweet_id", "nunique"),
            hostile_count=("hostility_pred", "sum"),
        )
    )
    summary = grid.merge(observed, on=["event_id", "temporal_phase"], how="left")
    summary[["comment_count", "hostile_count"]] = summary[
        ["comment_count", "hostile_count"]
    ].fillna(0).astype(int)
    summary["hostility_pct"] = np.where(
        summary["comment_count"].gt(0),
        100 * summary["hostile_count"] / summary["comment_count"],
        np.nan,
    )
    summary["phase_duration_hours"] = summary["temporal_phase"].map(
        PHASE_DURATION_HOURS
    )
    during = summary["temporal_phase"].eq("during")
    summary.loc[during, "phase_duration_hours"] = summary.loc[
        during, "event_duration_hours"
    ]
    summary["comments_per_hour"] = np.where(
        summary["phase_duration_hours"].gt(0),
        summary["comment_count"] / summary["phase_duration_hours"],
        np.nan,
    )
    summary["low_n_flag"] = summary["comment_count"].lt(min_phase_comments)
    summary["min_phase_comments"] = int(min_phase_comments)
    summary = _add_wilson_interval(summary, "hostile_count", "comment_count")
    return summary.sort_values(["formal_order", "phase_order"]).reset_index(drop=True)


def summarize_post_event_bins(
    prepared: pd.DataFrame,
    bin_hours: int = 6,
    max_post_event_hours: int = 72,
    min_bin_comments: int = 20,
) -> pd.DataFrame:
    """Create fixed-width bins aligned to each event end."""
    if bin_hours <= 0 or max_post_event_hours <= 0:
        raise ValueError("bin_hours y max_post_event_hours deben ser positivos")
    if max_post_event_hours % bin_hours:
        raise ValueError("max_post_event_hours debe ser múltiplo de bin_hours")
    _validate_columns(
        prepared,
        [
            "tweet_id",
            "event_id",
            "event_name",
            "formal_order",
            "event_date",
            "hours_from_event_end",
            "hostility_pred",
        ],
        "prepared corpus",
    )

    work = prepared.drop_duplicates("tweet_id").copy()
    post = work.loc[
        work["hours_from_event_end"].gt(0)
        & work["hours_from_event_end"].le(max_post_event_hours)
    ].copy()
    post["bin_index"] = (
        np.ceil(post["hours_from_event_end"] / bin_hours).astype(int) - 1
    ).clip(lower=0)
    post["bin_start_hour"] = post["bin_index"] * bin_hours

    observed = (
        post.groupby(["event_id", "bin_start_hour"], as_index=False)
        .agg(
            comment_count=("tweet_id", "nunique"),
            hostile_count=("hostility_pred", "sum"),
        )
    )
    event_meta = (
        work[["event_id", "event_name", "formal_order", "event_date"]]
        .drop_duplicates("event_id")
        .sort_values("formal_order")
    )
    bin_starts = np.arange(0, max_post_event_hours, bin_hours, dtype=int)
    grid = event_meta.assign(_key=1).merge(
        pd.DataFrame({"bin_start_hour": bin_starts}).assign(_key=1), on="_key"
    ).drop(columns="_key")
    summary = grid.merge(observed, on=["event_id", "bin_start_hour"], how="left")
    summary[["comment_count", "hostile_count"]] = summary[
        ["comment_count", "hostile_count"]
    ].fillna(0).astype(int)
    summary["bin_end_hour"] = summary["bin_start_hour"] + bin_hours
    summary["bin_midpoint_hour"] = (
        summary["bin_start_hour"] + summary["bin_end_hour"]
    ) / 2.0
    summary["bin_label"] = summary.apply(
        lambda row: f"{int(row['bin_start_hour'])}-{int(row['bin_end_hour'])} h",
        axis=1,
    )
    summary["hostility_pct"] = np.where(
        summary["comment_count"].gt(0),
        100 * summary["hostile_count"] / summary["comment_count"],
        np.nan,
    )
    summary["comments_per_hour"] = summary["comment_count"] / float(bin_hours)
    summary["low_n_flag"] = summary["comment_count"].lt(min_bin_comments)
    summary["min_bin_comments"] = int(min_bin_comments)
    summary["bin_hours"] = int(bin_hours)
    summary = _add_wilson_interval(summary, "hostile_count", "comment_count")
    return summary.sort_values(["formal_order", "bin_start_hour"]).reset_index(
        drop=True
    )


def summarize_persistence(
    phase_summary: pd.DataFrame,
    post_bins: pd.DataFrame,
    min_phase_comments: int = 20,
    min_bin_comments: int = 20,
) -> pd.DataFrame:
    """Describe changes from baseline and the post-event weighted linear trend."""
    _validate_columns(
        phase_summary,
        [
            "event_id",
            "event_name",
            "formal_order",
            "temporal_phase",
            "comment_count",
            "hostile_count",
            "hostility_pct",
        ],
        "phase summary",
    )
    _validate_columns(
        post_bins,
        [
            "event_id",
            "bin_midpoint_hour",
            "comment_count",
            "hostile_count",
            "hostility_pct",
        ],
        "post-event bins",
    )

    rows = []
    events = (
        phase_summary[["event_id", "event_name", "formal_order", "event_date"]]
        .drop_duplicates("event_id")
        .sort_values("formal_order")
    )
    for _, event in events.iterrows():
        event_phases = phase_summary.loc[
            phase_summary["event_id"].eq(event["event_id"])
        ].set_index("temporal_phase")

        def phase_value(phase: str, column: str) -> float:
            if phase not in event_phases.index:
                return np.nan
            return event_phases.loc[phase, column]

        pre_n = int(phase_value("pre_24h", "comment_count"))
        during_n = int(phase_value("during", "comment_count"))
        post_0_6_n = int(phase_value("post_0_6h", "comment_count"))
        post_6_24_n = int(phase_value("post_6_24h", "comment_count"))
        late_n = int(phase_value("post_24_72h", "comment_count"))
        post_0_24_n = post_0_6_n + post_6_24_n
        post_0_24_hostile = int(phase_value("post_0_6h", "hostile_count")) + int(
            phase_value("post_6_24h", "hostile_count")
        )
        post_0_24_rate = (
            100 * post_0_24_hostile / post_0_24_n if post_0_24_n else np.nan
        )

        pre_rate = phase_value("pre_24h", "hostility_pct")
        during_rate = phase_value("during", "hostility_pct")
        post_0_6_rate = phase_value("post_0_6h", "hostility_pct")
        late_rate = phase_value("post_24_72h", "hostility_pct")

        event_bins = post_bins.loc[
            post_bins["event_id"].eq(event["event_id"])
            & post_bins["comment_count"].ge(min_bin_comments)
            & post_bins["hostility_pct"].notna()
        ].copy()
        slope = np.nan
        if len(event_bins) >= 3:
            slope = float(
                np.polyfit(
                    event_bins["bin_midpoint_hour"].to_numpy(dtype=float),
                    event_bins["hostility_pct"].to_numpy(dtype=float),
                    1,
                    w=np.sqrt(event_bins["comment_count"].to_numpy(dtype=float)),
                )[0]
            )

        reliability_reasons = []
        if pre_n < min_phase_comments:
            reliability_reasons.append("pre_event_low_n")
        if during_n < min_phase_comments:
            reliability_reasons.append("during_event_low_n")
        if post_0_24_n < min_phase_comments:
            reliability_reasons.append("post_0_24_low_n")
        if len(event_bins) < 3:
            reliability_reasons.append("fewer_than_3_valid_post_bins")

        rows.append(
            {
                "event_id": event["event_id"],
                "event_name": event["event_name"],
                "formal_order": int(event["formal_order"]),
                "event_date": event["event_date"],
                "pre_24h_n": pre_n,
                "pre_24h_hostility_pct": pre_rate,
                "during_n": during_n,
                "during_hostility_pct": during_rate,
                "post_0_6h_n": post_0_6_n,
                "post_0_6h_hostility_pct": post_0_6_rate,
                "post_0_24h_n": post_0_24_n,
                "post_0_24h_hostility_pct": post_0_24_rate,
                "post_24_72h_n": late_n,
                "post_24_72h_hostility_pct": late_rate,
                "during_minus_pre_pp": during_rate - pre_rate,
                "post_0_24h_minus_pre_pp": post_0_24_rate - pre_rate,
                "post_24_72h_minus_pre_pp": late_rate - pre_rate,
                "late_minus_early_post_pp": late_rate - post_0_24_rate,
                "post_trend_slope_pp_per_hour": slope,
                "post_trend_change_per_24h_pp": slope * 24 if pd.notna(slope) else np.nan,
                "valid_post_bins": int(len(event_bins)),
                "min_phase_comments": int(min_phase_comments),
                "min_bin_comments": int(min_bin_comments),
                "temporal_reliability": (
                    "adequate_descriptive_coverage"
                    if not reliability_reasons
                    else "limited: " + "|".join(reliability_reasons)
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("formal_order").reset_index(drop=True)


def build_temporal_coverage(
    prepared: pd.DataFrame,
    phase_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Build event-level diagnostics for temporal coverage."""
    _validate_columns(
        prepared,
        [
            "tweet_id",
            "event_id",
            "event_name",
            "formal_order",
            "temporal_phase",
            "outside_reason",
            "hours_from_event_start",
            "hours_from_event_end",
        ],
        "prepared corpus",
    )
    work = prepared.drop_duplicates("tweet_id")
    totals = (
        work.groupby(["event_id", "event_name", "formal_order"], as_index=False)
        .agg(
            total_comments=("tweet_id", "nunique"),
            primary_window_comments=(
                "within_primary_temporal_window",
                "sum",
            ),
            min_hours_from_start=("hours_from_event_start", "min"),
            max_hours_from_end=("hours_from_event_end", "max"),
        )
    )
    totals["outside_window_comments"] = (
        totals["total_comments"] - totals["primary_window_comments"]
    )
    low_phase = (
        phase_summary.groupby("event_id", as_index=False)["low_n_flag"]
        .sum()
        .rename(columns={"low_n_flag": "low_n_phase_count"})
    )
    coverage = totals.merge(low_phase, on="event_id", how="left")
    coverage["primary_window_pct"] = (
        100 * coverage["primary_window_comments"] / coverage["total_comments"]
    )
    return coverage.sort_values("formal_order").reset_index(drop=True)


def temporal_assignments_table(prepared: pd.DataFrame) -> pd.DataFrame:
    """Return a compact, non-textual row-level audit table."""
    columns = [
        "tweet_id",
        "event_id",
        "event_name",
        "formal_order",
        "created_at_utc",
        "created_at_cr",
        "event_start_utc",
        "event_end_utc",
        "hours_from_event_start",
        "hours_from_event_end",
        "temporal_phase",
        "within_primary_temporal_window",
        "outside_reason",
        "hostility_pred",
    ]
    _validate_columns(prepared, columns, "prepared corpus")
    return prepared[columns].copy().sort_values(
        ["formal_order", "created_at_utc", "tweet_id"]
    )


def _save_figure(fig: plt.Figure, output_path: Union[str, Path]) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def save_phase_heatmap(
    phase_summary: pd.DataFrame,
    output_path: Union[str, Path],
    min_phase_comments: int = 20,
) -> Path:
    """Save a rate heatmap with counts and low-sample warnings."""
    events = (
        phase_summary[["event_id", "event_name", "formal_order"]]
        .drop_duplicates("event_id")
        .sort_values("formal_order")
    )
    rate = phase_summary.pivot(
        index="event_id", columns="temporal_phase", values="hostility_pct"
    ).reindex(index=events["event_id"], columns=PRIMARY_PHASES)
    counts = phase_summary.pivot(
        index="event_id", columns="temporal_phase", values="comment_count"
    ).reindex(index=events["event_id"], columns=PRIMARY_PHASES)

    cmap = LinearSegmentedColormap.from_list(
        "hatecr_temporal", ["#F2EEE7", "#E7B17B", "#C64A34", "#742522"]
    )
    cmap.set_bad("#E2E0DB")
    values = np.ma.masked_invalid(rate.to_numpy(dtype=float))

    fig, ax = plt.subplots(figsize=(14, 8.5))
    fig.patch.set_facecolor("#FAFAF8")
    ax.set_facecolor("#FAFAF8")
    image = ax.imshow(values, cmap=cmap, vmin=25, vmax=60, aspect="auto")

    labels = [
        f"{int(row.formal_order)} · {textwrap.fill(row.event_name, 35)}"
        for row in events.itertuples(index=False)
    ]
    ax.set_yticks(np.arange(len(events)))
    ax.set_yticklabels(labels)
    ax.set_xticks(np.arange(len(PRIMARY_PHASES)))
    ax.set_xticklabels([PHASE_LABELS[phase] for phase in PRIMARY_PHASES])
    ax.tick_params(axis="both", length=0)

    for row_index in range(len(events)):
        for column_index in range(len(PRIMARY_PHASES)):
            n_value = int(counts.iloc[row_index, column_index])
            rate_value = rate.iloc[row_index, column_index]
            if n_value == 0 or pd.isna(rate_value):
                label = "sin datos"
                color = "#555555"
            else:
                marker = " †" if n_value < min_phase_comments else ""
                label = f"{rate_value:.1f}%{marker}\nn={n_value}".replace(".", ",")
                color = "white" if rate_value >= 48 else "#2F2F2F"
            ax.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                fontsize=10.5,
                fontweight="bold" if n_value >= min_phase_comments else "normal",
                color=color,
            )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.025)
    colorbar.set_label("Comentarios predichos como hostiles (%)")
    ax.set_title(
        "Hostilidad antes, durante y después de cada momento electoral\n"
        "Perfil balanceado v2",
        fontsize=19,
        pad=20,
    )
    fig.text(
        0.5,
        0.025,
        f"† Menos de {min_phase_comments} comentarios: tasa inestable. "
        "Los porcentajes son predicciones exploratorias y no demuestran un efecto causal del evento.",
        ha="center",
        fontsize=9.5,
        color="#5E5E5E",
    )
    fig.subplots_adjust(left=0.30, right=0.92, top=0.84, bottom=0.14)
    return _save_figure(fig, output_path)


def save_post_event_decay_figure(
    post_bins: pd.DataFrame,
    phase_summary: pd.DataFrame,
    output_path: Union[str, Path],
    min_bin_comments: int = 20,
    min_pre_comments: int = 20,
) -> Path:
    """Save six small multiples aligned to event end."""
    events = (
        phase_summary[["event_id", "event_name", "formal_order"]]
        .drop_duplicates("event_id")
        .sort_values("formal_order")
    )
    pre_summary = (
        phase_summary.loc[
            phase_summary["temporal_phase"].eq("pre_24h"),
            ["event_id", "hostility_pct", "comment_count"],
        ]
        .drop_duplicates("event_id")
        .set_index("event_id")
    )

    fig, axes = plt.subplots(3, 2, figsize=(15, 12), sharex=True, sharey=True)
    fig.patch.set_facecolor("#FAFAF8")
    axes_flat = axes.ravel()
    for ax, event in zip(axes_flat, events.itertuples(index=False)):
        ax.set_facecolor("#FAFAF8")
        event_bins = post_bins.loc[post_bins["event_id"].eq(event.event_id)].copy()
        valid = event_bins["comment_count"].ge(min_bin_comments)
        plotted_rates = event_bins["hostility_pct"].where(valid)
        ax.plot(
            event_bins["bin_midpoint_hour"],
            plotted_rates,
            color="#C64A34",
            marker="o",
            linewidth=2,
            markersize=5,
            label="Tasa post-evento",
        )
        low_n = event_bins["comment_count"].gt(0) & ~valid
        ax.scatter(
            event_bins.loc[low_n, "bin_midpoint_hour"],
            event_bins.loc[low_n, "hostility_pct"],
            color="#9D9991",
            marker="x",
            s=35,
            label=f"n < {min_bin_comments}",
        )
        pre_rate = (
            pre_summary.loc[event.event_id, "hostility_pct"]
            if event.event_id in pre_summary.index
            else np.nan
        )
        pre_count = (
            int(pre_summary.loc[event.event_id, "comment_count"])
            if event.event_id in pre_summary.index
            else 0
        )
        if pd.notna(pre_rate) and pre_count >= min_pre_comments:
            ax.axhline(
                pre_rate,
                color="#3B6571",
                linestyle="--",
                linewidth=1.5,
                label="Nivel previo (24 h)",
            )
        elif pre_count:
            ax.text(
                0.98,
                0.05,
                f"Base previa insuficiente (n={pre_count})",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=8.5,
                color="#6A6761",
            )
        ax.set_title(
            f"{int(event.formal_order)} · {textwrap.fill(event.event_name, 34)}",
            fontsize=12,
            pad=10,
        )
        ax.set_xlim(0, 72)
        ax.set_ylim(20, 70)
        ax.set_xticks(np.arange(0, 73, 12))
        ax.grid(True, color="#D9D9D6", linewidth=0.7, alpha=0.65)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    for ax in axes[:, 0]:
        ax.set_ylabel("Hostilidad predicha (%)")
    for ax in axes[-1, :]:
        ax.set_xlabel("Horas desde el final del evento")

    handles = [
        Patch(facecolor="#C64A34", label="Tasa post-evento (n suficiente)"),
        Patch(facecolor="#3B6571", label="Nivel previo de 24 h (n suficiente)"),
        Patch(facecolor="#9D9991", label=f"Intervalo con n < {min_bin_comments}"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(
        "Persistencia de la hostilidad durante las 72 horas posteriores\n"
        "Intervalos de 6 horas · perfil balanceado v2",
        fontsize=19,
        y=0.98,
    )
    fig.text(
        0.5,
        0.025,
        "Las líneas se interrumpen cuando el intervalo tiene pocos comentarios. "
        "La pendiente es descriptiva; no representa una tasa causal de decaimiento.",
        ha="center",
        fontsize=9.5,
        color="#5E5E5E",
    )
    fig.subplots_adjust(left=0.09, right=0.97, top=0.89, bottom=0.11, hspace=0.36)
    return _save_figure(fig, output_path)


def save_phase_change_figure(
    persistence: pd.DataFrame,
    output_path: Union[str, Path],
) -> Path:
    """Save changes in percentage points relative to the pre-event baseline."""
    ordered = persistence.sort_values("formal_order").reset_index(drop=True)
    y = np.arange(len(ordered))
    height = 0.23
    series = [
        ("during_minus_pre_pp", "Durante", "#C64A34", -height),
        ("post_0_24h_minus_pre_pp", "Después 0-24 h", "#D99855", 0.0),
        ("post_24_72h_minus_pre_pp", "Después 24-72 h", "#3B6571", height),
    ]

    fig, ax = plt.subplots(figsize=(14, 8.5))
    fig.patch.set_facecolor("#FAFAF8")
    ax.set_facecolor("#FAFAF8")
    insufficient = ordered["temporal_reliability"].astype(str).str.startswith(
        "limited:"
    )
    for column, label, color, offset in series:
        values = ordered[column].to_numpy(dtype=float)
        values = np.where(insufficient.to_numpy(), np.nan, values)
        bars = ax.barh(y + offset, values, height=height, color=color, label=label)
        for bar, value in zip(bars, values):
            if pd.isna(value):
                continue
            horizontal = "left" if value >= 0 else "right"
            text_x = value + (0.25 if value >= 0 else -0.25)
            ax.text(
                text_x,
                bar.get_y() + bar.get_height() / 2,
                f"{value:+.1f}".replace(".", ","),
                va="center",
                ha=horizontal,
                fontsize=9,
                color="#333333",
            )

    for index in np.flatnonzero(insufficient.to_numpy()):
        ax.text(
            0.35,
            y[index],
            "cobertura temporal insuficiente",
            va="center",
            ha="left",
            fontsize=9.5,
            color="#6A6761",
            fontstyle="italic",
        )

    labels = [
        f"{int(row.formal_order)} · {textwrap.fill(row.event_name, 34)}"
        for row in ordered.itertuples(index=False)
    ]
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_xlabel("Cambio respecto a las 24 h previas (puntos porcentuales)")
    ax.grid(True, axis="x", color="#D9D9D6", linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=3,
        frameon=False,
    )
    ax.set_title(
        "Cambio temporal de la hostilidad respecto al nivel previo\n"
        "Perfil balanceado v2",
        fontsize=19,
        pad=20,
    )
    fig.text(
        0.5,
        0.018,
        "Valores positivos indican más hostilidad predicha que en las 24 horas previas; "
        "valores negativos indican menos. Las comparaciones con cobertura insuficiente se omiten. "
        "Análisis descriptivo, no causal.",
        ha="center",
        fontsize=9.5,
        color="#5E5E5E",
    )
    fig.subplots_adjust(left=0.31, right=0.97, top=0.84, bottom=0.20)
    return _save_figure(fig, output_path)

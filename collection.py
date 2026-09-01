from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from tqdm.auto import tqdm

from src.search_queries import DEFAULT_BASE_QUERY_TERMS, dedupe_keep_order, get_terms_from_group
from src.x_api import (
    XApiUrls,
    count_x_posts_full_archive,
    get_quote_tweets_url,
    parse_datetime_utc,
    recent_window_is_eligible,
    safe_json,
    summarize_api_error,
    to_rfc3339_utc,
)

SOURCE_POST_CANDIDATE_COLUMNS = [
    "source_type",
    "source_universe",
    "anchor_type",
    "anchor_handle",
    "event_id",
    "event_name",
    "event_date",
    "media_id",
    "media_name",
    "media_handle",
    "search_mode",
    "term_batch_id",
    "term_count",
    "term_group_names",
    "search_modes_found",
    "query",
    "query_start_time",
    "query_end_time",
    "endpoint_used",
    "tweet_id",
    "id",
    "source_post_id",
    "text",
    "source_post_text",
    "created_at",
    "source_post_created_at",
    "author_id",
    "conversation_id",
    "lang",
    "public_metrics",
    "source_post_public_metrics",
    "reply_count",
    "retweet_count",
    "like_count",
    "quote_count",
    "source_post_url",
    "collected_at",
]

SOURCE_POST_AUDIT_COLUMNS = [
    "event_id",
    "media_id",
    "media_handle",
    "search_mode",
    "term_batch_id",
    "term_count",
    "term_group_names",
    "query",
    "start_time",
    "end_time",
    "endpoint_attempted",
    "endpoint_used",
    "status",
    "status_code",
    "n_rows",
    "error_summary",
]

FULL_ARCHIVE_COUNT_COLUMNS = [
    "event_id",
    "event_name",
    "formal_order",
    "media_id",
    "media_name",
    "media_handle",
    "search_mode",
    "term_count",
    "term_group_names",
    "query",
    "start_time",
    "end_time",
    "granularity",
    "endpoint",
    "endpoint_url",
    "status",
    "status_code",
    "bucket_count",
    "total_tweet_count",
    "error_summary",
    "seconds",
    "collected_at",
]

SOURCE_POST_REVIEW_COLUMNS = [
    "id",
    "source_post_id",
    "source_post_url",
    "event_id",
    "media_id",
    "media_handle",
    "source_post_created_at",
    "reply_count",
    "quote_count",
    "retweet_count",
    "like_count",
    "engagement_score",
    "keyword_hits",
    "source_post_text",
    "manual_keep",
    "semiauto_keep",
    "final_keep",
    "selection_reason",
    "event_name",
    "media_name",
    "search_modes_found",
]

REPLIES_CLEAN_COLUMNS = [
    "source_type",
    "source_universe",
    "anchor_type",
    "anchor_post_id",
    "anchor_media_id",
    "anchor_media_handle",
    "event_id",
    "event_name",
    "formal_event_memberships",
    "formal_event_names",
    "formal_event_count",
    "media_id",
    "media_name",
    "media_handle",
    "source_post_id",
    "tweet_id",
    "reply_id",
    "text",
    "reply_text",
    "created_at",
    "reply_created_at",
    "reply_author_id_hash",
    "lang",
    "public_metrics",
    "conversation_id",
    "in_reply_to_user_id",
    "referenced_tweets",
    "query",
    "collected_at",
]

REPLIES_STATS_COLUMNS = [
    "event_id",
    "formal_event_memberships",
    "formal_event_count",
    "media_id",
    "media_handle",
    "source_post_id",
    "query",
    "start_time",
    "end_time",
    "endpoint_attempted",
    "endpoint_used",
    "status",
    "status_code",
    "n_rows_api",
    "n_rows_kept",
    "error_summary",
    "seconds",
]

QUOTE_CLEAN_COLUMNS = [
    "source_type",
    "source_universe",
    "anchor_type",
    "anchor_post_id",
    "anchor_media_id",
    "anchor_media_handle",
    "source_post_id",
    "event_id",
    "event_name",
    "formal_event_memberships",
    "formal_event_names",
    "formal_event_count",
    "media_id",
    "media_name",
    "media_handle",
    "tweet_id",
    "text",
    "created_at",
    "author_id",
    "conversation_id",
    "referenced_tweets",
    "public_metrics",
    "lang",
    "query",
    "collected_at",
]

QUOTE_STATS_COLUMNS = [
    "event_id",
    "formal_event_memberships",
    "formal_event_count",
    "media_id",
    "media_handle",
    "source_post_id",
    "endpoint_used",
    "status",
    "status_code",
    "n_rows_api",
    "n_rows_kept",
    "error_summary",
    "seconds",
]

REPLY_COLLECTION_MANIFEST_COLUMNS = [
    "collection_layer",
    "source_type",
    "source_universe",
    "anchor_type",
    "anchor_handle",
    "eligible_for_collection",
    "selected_for_collection",
    "selection_reason",
    "selection_rank_event_media",
    "collection_batch_id",
    "source_post_id",
    "source_post_url",
    "source_post_created_at",
    "primary_event_id",
    "primary_event_order",
    "event_id",
    "event_name",
    "formal_event_memberships",
    "formal_event_names",
    "formal_event_count",
    "media_id",
    "media_name",
    "media_handle",
    "source_post_text",
    "reply_count",
    "quote_count",
    "retweet_count",
    "like_count",
    "engagement_score",
    "query",
    "endpoint_type",
    "endpoint_url",
    "start_time",
    "end_time",
    "reply_window_hours",
    "planned_max_replies",
    "planned_max_pages",
    "planned_results_per_page",
    "estimated_api_requests",
    "plan_created_at",
]

QUOTE_COLLECTION_MANIFEST_COLUMNS = [
    "collection_layer",
    "source_type",
    "source_universe",
    "anchor_type",
    "anchor_handle",
    "eligible_for_collection",
    "selected_for_collection",
    "selection_reason",
    "selection_rank_event_media",
    "collection_batch_id",
    "source_post_id",
    "source_post_url",
    "source_post_created_at",
    "primary_event_id",
    "primary_event_order",
    "event_id",
    "event_name",
    "formal_event_memberships",
    "formal_event_names",
    "formal_event_count",
    "media_id",
    "media_name",
    "media_handle",
    "source_post_text",
    "reply_count",
    "quote_count",
    "retweet_count",
    "like_count",
    "engagement_score",
    "query",
    "endpoint_type",
    "endpoint_url",
    "planned_max_quotes",
    "planned_max_pages",
    "planned_results_per_page",
    "estimated_api_requests",
    "plan_created_at",
]

COLLECTION_PLAN_SUMMARY_COLUMNS = [
    "collection_layer",
    "event_id",
    "media_id",
    "media_handle",
    "manifest_posts",
    "eligible_posts",
    "selected_posts",
    "reported_interactions",
    "planned_interaction_cap",
    "estimated_api_requests",
]

ORGANIC_POST_COLUMNS = [
    "source_type",
    "source_universe",
    "anchor_type",
    "anchor_post_id",
    "anchor_media_id",
    "anchor_media_handle",
    "event_id",
    "event_name",
    "tweet_id",
    "text",
    "created_at",
    "author_id",
    "conversation_id",
    "lang",
    "reply_count",
    "quote_count",
    "retweet_count",
    "like_count",
    "public_metrics",
    "query",
    "collected_at",
]

CANDIDATE_INTERACTION_COLUMNS = [
    "source_type",
    "source_universe",
    "anchor_type",
    "anchor_post_id",
    "anchor_media_id",
    "anchor_media_handle",
    "target_actor_id",
    "target_actor_name",
    "target_handle",
    "event_id",
    "event_name",
    "tweet_id",
    "text",
    "created_at",
    "author_id",
    "conversation_id",
    "lang",
    "public_metrics",
    "query",
    "query_mode",
    "collected_at",
]

ORGANIC_AUDIT_COLUMNS = [
    "source_type",
    "event_id",
    "query",
    "start_time",
    "end_time",
    "endpoint_attempted",
    "endpoint_used",
    "status",
    "status_code",
    "n_rows",
    "error_summary",
    "seconds",
]

CANDIDATE_AUDIT_COLUMNS = [
    "source_type",
    "event_id",
    "target_actor_id",
    "target_handle",
    "query_mode",
    "query",
    "start_time",
    "end_time",
    "endpoint_attempted",
    "endpoint_used",
    "status",
    "status_code",
    "n_rows",
    "error_summary",
    "seconds",
]

UNIFIED_CORPUS_COLUMNS = [
    "corpus_id",
    "source_type",
    "source_type_originals",
    "source_universe",
    "anchor_type",
    "anchor_post_id",
    "anchor_media_id",
    "anchor_media_handle",
    "event_id",
    "event_name",
    "source_post_id",
    "parent_post_id",
    "target_actor_id",
    "target_handle",
    "media_id",
    "media_handle",
    "tweet_id",
    "text",
    "text_norm",
    "created_at",
    "author_id",
    "author_id_hash",
    "conversation_id",
    "lang",
    "reply_count",
    "quote_count",
    "retweet_count",
    "like_count",
    "query",
    "collected_at",
]

MEDIA_ANCHORED_CORPUS_COLUMNS = [
    "corpus_id",
    "source_type",
    "source_universe",
    "anchor_type",
    "anchor_post_id",
    "anchor_media_id",
    "anchor_media_handle",
    "event_id",
    "event_name",
    "tweet_id",
    "text",
    "text_norm",
    "created_at",
    "author_id",
    "conversation_id",
    "lang",
    "reply_count",
    "quote_count",
    "retweet_count",
    "like_count",
    "query",
    "collected_at",
]

SOURCE_POST_FINAL_COLUMNS = [
    "source_type",
    "source_universe",
    "anchor_type",
    "anchor_handle",
    "event_id",
    "event_name",
    "event_date",
    "media_id",
    "media_name",
    "media_handle",
    "tweet_id",
    "id",
    "source_post_id",
    "text",
    "source_post_text",
    "created_at",
    "source_post_created_at",
    "author_id",
    "conversation_id",
    "lang",
    "public_metrics",
    "source_post_public_metrics",
    "reply_count",
    "retweet_count",
    "like_count",
    "quote_count",
    "source_post_url",
    "collected_at",
    "keyword_hits",
    "engagement_score",
    "semiauto_keep",
    "manual_keep",
    "final_keep",
    "selection_reason",
    "search_mode",
    "term_batch_id",
    "term_count",
    "term_group_names",
    "search_modes_found",
]


def normalize_source_post_id(value: Any) -> str | None:
    if value is None:
        return None

    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None

    if re.fullmatch(r"\d+", text):
        return text

    if re.fullmatch(r"\d+\.0+", text):
        return text.split(".")[0]

    if re.fullmatch(r"\d+(\.\d+)?e\+\d+", text.lower()):
        try:
            return f"{float(text):.0f}"
        except Exception:
            return None

    return None


def _normalize_text_for_matching(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"\s+", " ", text)
    return text


def _coerce_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def _ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = pd.NA
    return out[columns]


def _clean_optional_text(value: Any) -> str:
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "<na>"} else text


def prepare_source_posts_for_interaction_collection(
    source_posts_df: pd.DataFrame,
    event_order: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Return one media source post per X id while preserving event memberships."""
    if source_posts_df.empty:
        return source_posts_df.copy()

    work = source_posts_df.copy()
    id_col = next(
        (col for col in ["source_post_id", "tweet_id", "id"] if col in work.columns),
        None,
    )
    if id_col is None:
        raise ValueError("Source posts require source_post_id, tweet_id, or id")

    work["source_post_id"] = work[id_col].map(normalize_source_post_id)
    invalid_id_count = int(work["source_post_id"].isna().sum())
    if invalid_id_count:
        raise ValueError(f"Source posts contain {invalid_id_count} invalid X ids")

    if "source_type" in work.columns:
        source_types = {
            _clean_optional_text(value)
            for value in work["source_type"].dropna().tolist()
        }
        source_types.discard("")
        if source_types - {"media_source_post"}:
            raise ValueError(
                "Interaction collection requires media_source_post rows only: "
                f"{sorted(source_types)}"
            )

    for required in ["event_id", "media_id", "media_handle"]:
        if required not in work.columns:
            raise ValueError(f"Source posts missing required column: {required}")
        missing = work[required].map(_clean_optional_text).eq("")
        if missing.any():
            raise ValueError(
                f"Source posts contain {int(missing.sum())} missing values in {required}"
            )

    if "source_post_created_at" not in work.columns:
        work["source_post_created_at"] = work.get("created_at")
    if "source_post_text" not in work.columns:
        work["source_post_text"] = work.get("text")

    for metric in ["reply_count", "quote_count", "retweet_count", "like_count"]:
        if metric not in work.columns:
            work[metric] = 0
        work[metric] = _coerce_numeric(work[metric], default=0).clip(lower=0)

    if "engagement_score" not in work.columns:
        work["engagement_score"] = (
            work["reply_count"] * 3
            + work["quote_count"] * 2
            + work["retweet_count"]
            + work["like_count"] * 0.25
        )
    else:
        work["engagement_score"] = _coerce_numeric(
            work["engagement_score"], default=0
        )

    order_map = {str(key): int(value) for key, value in (event_order or {}).items()}
    fallback_order = max(order_map.values(), default=-1) + 1000
    work["_event_order"] = work["event_id"].map(
        lambda value: order_map.get(str(value), fallback_order)
    )
    work["_row_order"] = range(len(work))

    rows: list[dict[str, Any]] = []
    for source_post_id, group in work.groupby("source_post_id", sort=False, dropna=False):
        group_sorted = group.sort_values(["_event_order", "_row_order"])
        representative = group_sorted.iloc[0].to_dict()

        event_pairs: list[tuple[str, str]] = []
        seen_event_ids: set[str] = set()
        for _, event_row in group_sorted.iterrows():
            event_id = _clean_optional_text(event_row.get("event_id"))
            if not event_id or event_id in seen_event_ids:
                continue
            seen_event_ids.add(event_id)
            event_pairs.append(
                (event_id, _clean_optional_text(event_row.get("event_name")))
            )

        primary_event_id = event_pairs[0][0]
        primary_event_name = event_pairs[0][1]
        representative.update(
            {
                "source_post_id": str(source_post_id),
                "primary_event_id": primary_event_id,
                "primary_event_order": order_map.get(primary_event_id, fallback_order),
                "event_id": primary_event_id,
                "event_name": primary_event_name,
                "formal_event_memberships": "|".join(pair[0] for pair in event_pairs),
                "formal_event_names": "|".join(
                    pair[1] for pair in event_pairs if pair[1]
                ),
                "formal_event_count": len(event_pairs),
                "duplicate_across_formal_events": len(event_pairs) > 1,
            }
        )
        rows.append(representative)

    prepared = pd.DataFrame(rows).drop(columns=["_event_order", "_row_order"], errors="ignore")
    prepared = prepared.sort_values(
        ["primary_event_order", "media_id", "reply_count", "quote_count", "engagement_score"],
        ascending=[True, True, False, False, False],
    ).reset_index(drop=True)
    return prepared


def _apply_interaction_plan_selection(
    manifest: pd.DataFrame,
    interaction_count_col: str,
    min_interaction_count: int,
    max_posts_per_event_media: int,
    batch_size: int,
) -> pd.DataFrame:
    out = manifest.copy()
    interaction_counts = _coerce_numeric(out[interaction_count_col], default=0).clip(lower=0)
    created_at_ok = pd.to_datetime(
        out.get("source_post_created_at"), errors="coerce", utc=True
    ).notna()
    eligible = (interaction_counts >= max(0, int(min_interaction_count))) & created_at_ok
    out["eligible_for_collection"] = eligible
    out["selection_rank_event_media"] = pd.Series(pd.NA, index=out.index, dtype="Int64")

    eligible_sorted = out[eligible].sort_values(
        [
            "primary_event_order",
            "event_id",
            "media_id",
            interaction_count_col,
            "engagement_score",
            "source_post_created_at",
        ],
        ascending=[True, True, True, False, False, True],
    )
    ranks = (
        eligible_sorted.groupby(["event_id", "media_id"], dropna=False)
        .cumcount()
        .add(1)
        .astype("Int64")
    )
    out.loc[eligible_sorted.index, "selection_rank_event_media"] = ranks.to_numpy()

    group_cap = max(0, int(max_posts_per_event_media))
    if group_cap == 0:
        selected = eligible
    else:
        selected = eligible & out["selection_rank_event_media"].le(group_cap).fillna(False)
    out["selected_for_collection"] = selected.astype(bool)

    out["selection_reason"] = "selected"
    out.loc[~created_at_ok, "selection_reason"] = "missing_source_post_created_at"
    out.loc[
        created_at_ok & (interaction_counts < max(0, int(min_interaction_count))),
        "selection_reason",
    ] = f"below_min_{interaction_count_col}"
    out.loc[eligible & ~selected, "selection_reason"] = "over_event_media_cap"

    out["collection_batch_id"] = pd.NA
    selected_sorted = out[selected].sort_values(
        [
            "selection_rank_event_media",
            "media_id",
            "primary_event_order",
            "event_id",
        ]
    )
    batch_size_clean = max(1, int(batch_size))
    for position, row_index in enumerate(selected_sorted.index):
        out.at[row_index, "collection_batch_id"] = (
            f"batch_{(position // batch_size_clean) + 1:03d}"
        )
    return out


def build_reply_collection_manifest(
    source_posts_df: pd.DataFrame,
    search_all_url: str,
    event_order: dict[str, int] | None = None,
    reply_window_hours: int = 72,
    min_reply_count: int = 1,
    max_posts_per_event_media: int = 10,
    max_replies_per_post: int = 100,
    max_pages_per_post: int = 10,
    results_per_page: int = 100,
    batch_size: int = 50,
) -> pd.DataFrame:
    """Build an auditable reply plan without making API requests."""
    prepared = prepare_source_posts_for_interaction_collection(
        source_posts_df=source_posts_df,
        event_order=event_order,
    )
    if prepared.empty:
        return _ensure_columns(pd.DataFrame(), REPLY_COLLECTION_MANIFEST_COLUMNS)

    manifest = _apply_interaction_plan_selection(
        manifest=prepared,
        interaction_count_col="reply_count",
        min_interaction_count=min_reply_count,
        max_posts_per_event_media=max_posts_per_event_media,
        batch_size=batch_size,
    )

    hours = max(1, int(reply_window_hours))
    max_replies = max(1, int(max_replies_per_post))
    max_pages = max(1, int(max_pages_per_post))
    page_size = max(10, min(100, int(results_per_page)))
    created_at = pd.to_datetime(
        manifest["source_post_created_at"], errors="coerce", utc=True
    )

    manifest["collection_layer"] = "reply_to_media_post"
    manifest["query"] = manifest["source_post_id"].map(
        lambda source_post_id: f"conversation_id:{source_post_id} -is:retweet"
    )
    manifest["endpoint_type"] = "full_archive_search"
    manifest["endpoint_url"] = str(search_all_url).rstrip("/")
    manifest["start_time"] = created_at.map(
        lambda value: to_rfc3339_utc(value.to_pydatetime()) if pd.notna(value) else None
    )
    manifest["end_time"] = (created_at + pd.Timedelta(hours=hours)).map(
        lambda value: to_rfc3339_utc(value.to_pydatetime()) if pd.notna(value) else None
    )
    manifest["reply_window_hours"] = hours

    reported = _coerce_numeric(manifest["reply_count"], default=0).clip(lower=0).astype(int)
    planned = reported.clip(upper=max_replies).where(
        manifest["selected_for_collection"], 0
    )
    # conversation_id search can include the source post itself, so reserve one row.
    estimated_pages = planned.map(
        lambda count: min(max_pages, (int(count) + 1 + page_size - 1) // page_size)
        if int(count) > 0
        else 0
    )
    manifest["planned_max_replies"] = planned.astype(int)
    manifest["planned_max_pages"] = max_pages
    manifest["planned_results_per_page"] = page_size
    manifest["estimated_api_requests"] = estimated_pages.astype(int)
    manifest["plan_created_at"] = datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    return _ensure_columns(manifest, REPLY_COLLECTION_MANIFEST_COLUMNS)


def build_quote_collection_manifest(
    source_posts_df: pd.DataFrame,
    api_base_url: str,
    event_order: dict[str, int] | None = None,
    min_quote_count: int = 1,
    max_posts_per_event_media: int = 10,
    max_quotes_per_post: int = 100,
    max_pages_per_post: int = 10,
    results_per_page: int = 100,
    batch_size: int = 50,
) -> pd.DataFrame:
    """Build an auditable quote-tweet plan without making API requests."""
    prepared = prepare_source_posts_for_interaction_collection(
        source_posts_df=source_posts_df,
        event_order=event_order,
    )
    if prepared.empty:
        return _ensure_columns(pd.DataFrame(), QUOTE_COLLECTION_MANIFEST_COLUMNS)

    manifest = _apply_interaction_plan_selection(
        manifest=prepared,
        interaction_count_col="quote_count",
        min_interaction_count=min_quote_count,
        max_posts_per_event_media=max_posts_per_event_media,
        batch_size=batch_size,
    )
    max_quotes = max(1, int(max_quotes_per_post))
    max_pages = max(1, int(max_pages_per_post))
    page_size = max(10, min(100, int(results_per_page)))
    base_url = str(api_base_url).rstrip("/")

    manifest["collection_layer"] = "quote_of_media_post"
    manifest["query"] = manifest["source_post_id"].map(
        lambda source_post_id: f"quotes_of:{source_post_id}"
    )
    manifest["endpoint_type"] = "quote_tweets"
    manifest["endpoint_url"] = manifest["source_post_id"].map(
        lambda source_post_id: f"{base_url}/tweets/{source_post_id}/quote_tweets"
    )

    reported = _coerce_numeric(manifest["quote_count"], default=0).clip(lower=0).astype(int)
    planned = reported.clip(upper=max_quotes).where(
        manifest["selected_for_collection"], 0
    )
    estimated_pages = planned.map(
        lambda count: min(max_pages, (int(count) + page_size - 1) // page_size)
        if int(count) > 0
        else 0
    )
    manifest["planned_max_quotes"] = planned.astype(int)
    manifest["planned_max_pages"] = max_pages
    manifest["planned_results_per_page"] = page_size
    manifest["estimated_api_requests"] = estimated_pages.astype(int)
    manifest["plan_created_at"] = datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    return _ensure_columns(manifest, QUOTE_COLLECTION_MANIFEST_COLUMNS)


def summarize_interaction_collection_plan(
    reply_manifest_df: pd.DataFrame,
    quote_manifest_df: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize selected posts and estimated API use by layer/event/media."""
    summaries: list[pd.DataFrame] = []
    layer_specs = [
        (reply_manifest_df, "reply_count", "planned_max_replies"),
        (quote_manifest_df, "quote_count", "planned_max_quotes"),
    ]
    for manifest, reported_col, planned_col in layer_specs:
        if manifest.empty:
            continue
        work = manifest.copy()
        for column in [reported_col, planned_col, "estimated_api_requests"]:
            work[column] = _coerce_numeric(work[column], default=0)
        work["eligible_for_collection"] = work["eligible_for_collection"].fillna(False).astype(bool)
        work["selected_for_collection"] = work["selected_for_collection"].fillna(False).astype(bool)

        summary = (
            work.groupby(
                ["collection_layer", "event_id", "media_id", "media_handle"],
                dropna=False,
            )
            .agg(
                manifest_posts=("source_post_id", "nunique"),
                eligible_posts=("eligible_for_collection", "sum"),
                selected_posts=("selected_for_collection", "sum"),
                reported_interactions=(reported_col, "sum"),
                planned_interaction_cap=(planned_col, "sum"),
                estimated_api_requests=("estimated_api_requests", "sum"),
            )
            .reset_index()
        )
        summaries.append(summary)

    if not summaries:
        return _ensure_columns(pd.DataFrame(), COLLECTION_PLAN_SUMMARY_COLUMNS)
    result = pd.concat(summaries, ignore_index=True, sort=False)
    return _ensure_columns(result, COLLECTION_PLAN_SUMMARY_COLUMNS)


def _formal_event_context(post: Any, event_id: str) -> tuple[str, str, int]:
    memberships = _clean_optional_text(post.get("formal_event_memberships")) or event_id
    names = _clean_optional_text(post.get("formal_event_names"))
    try:
        event_count = int(float(post.get("formal_event_count")))
    except (TypeError, ValueError):
        event_count = len([value for value in memberships.split("|") if value])
    return memberships, names, max(1, event_count)


def _validate_media_anchored_query_spec(spec: dict[str, Any]) -> None:
    query = str(spec.get("query", "") or "")
    handle = str(spec.get("media_handle", "") or "").strip().lstrip("@")
    query_lower = query.lower()

    if not handle:
        raise ValueError(
            f"Query spec invalido para media_anchored: media_handle ausente. spec={spec}"
        )

    expected_token = f"from:{handle.lower()}"
    if expected_token not in query_lower:
        raise ValueError(
            "Regla metodologica violada: COLLECTION_SCOPE=media_anchored exige "
            f"`from:{handle}` en la query. query={query}"
        )


def _compute_wait_seconds(response: requests.Response, max_wait_seconds: int) -> int:
    reset_header = response.headers.get("x-rate-limit-reset")
    if reset_header and str(reset_header).isdigit():
        wait_seconds = max(1, int(reset_header) - int(time.time()) + 2)
    else:
        wait_seconds = max_wait_seconds
    return int(min(max_wait_seconds, wait_seconds))


def _request_search_page(
    endpoint_url: str,
    headers: dict[str, str],
    query: str,
    tweet_fields: str,
    start_time: str | None = None,
    end_time: str | None = None,
    max_results: int = 100,
    next_token: str | None = None,
    timeout_seconds: int = 60,
) -> requests.Response:
    params = {
        "query": query,
        "max_results": max(10, min(100, int(max_results))),
        "tweet.fields": tweet_fields,
    }
    if start_time:
        params["start_time"] = start_time
    if end_time:
        params["end_time"] = end_time
    if next_token:
        params["next_token"] = next_token

    return requests.get(endpoint_url, headers=headers, params=params, timeout=timeout_seconds)


def _run_search_endpoint(
    endpoint_url: str,
    endpoint_label: str,
    headers: dict[str, str],
    query: str,
    tweet_fields: str,
    start_time: str | None,
    end_time: str | None,
    max_results_per_page: int,
    max_pages: int,
    timeout_seconds: int,
    sleep_seconds: float,
    max_rate_limit_wait_seconds: int,
    max_429_retries: int,
) -> dict[str, Any]:
    tweets: list[dict[str, Any]] = []
    next_token: str | None = None
    last_status_code: int | None = None
    error_summary = ""
    pages_fetched = 0
    retries_429 = 0

    for _ in range(max_pages):
        try:
            response = _request_search_page(
                endpoint_url=endpoint_url,
                headers=headers,
                query=query,
                tweet_fields=tweet_fields,
                start_time=start_time,
                end_time=end_time,
                max_results=max_results_per_page,
                next_token=next_token,
                timeout_seconds=timeout_seconds,
            )
        except Exception as exc:
            return {
                "endpoint_label": endpoint_label,
                "endpoint_url": endpoint_url,
                "status_code": None,
                "error_summary": f"request_exception: {exc}",
                "tweets": tweets,
                "pages_fetched": pages_fetched,
            }

        last_status_code = response.status_code

        if response.status_code == 429:
            retries_429 += 1
            if retries_429 > max_429_retries:
                return {
                    "endpoint_label": endpoint_label,
                    "endpoint_url": endpoint_url,
                    "status_code": 429,
                    "error_summary": "rate_limit_retry_exceeded",
                    "tweets": tweets,
                    "pages_fetched": pages_fetched,
                }

            wait_seconds = _compute_wait_seconds(response, max_rate_limit_wait_seconds)
            time.sleep(wait_seconds)
            continue

        if response.status_code != 200:
            payload = safe_json(response)
            error_summary = summarize_api_error(payload)
            return {
                "endpoint_label": endpoint_label,
                "endpoint_url": endpoint_url,
                "status_code": response.status_code,
                "error_summary": error_summary,
                "tweets": tweets,
                "pages_fetched": pages_fetched,
            }

        payload = safe_json(response)
        data = payload.get("data", []) or []
        meta = payload.get("meta", {}) or {}
        tweets.extend(data)
        pages_fetched += 1

        next_token = meta.get("next_token")
        if not next_token:
            break

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    return {
        "endpoint_label": endpoint_label,
        "endpoint_url": endpoint_url,
        "status_code": last_status_code,
        "error_summary": error_summary,
        "tweets": tweets,
        "pages_fetched": pages_fetched,
    }


def collect_tweets_with_archive_fallback(
    query: str,
    headers: dict[str, str],
    urls: XApiUrls,
    use_full_archive: bool,
    start_time: str | None,
    end_time: str | None,
    tweet_fields: str,
    context: dict[str, Any],
    max_results_per_page: int = 100,
    max_pages: int = 5,
    timeout_seconds: int = 60,
    sleep_seconds: float = 0.5,
    max_rate_limit_wait_seconds: int = 30,
    max_429_retries: int = 3,
    recent_days: int = 7,
    allow_recent_fallback: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    recent_eligible = recent_window_is_eligible(
        start_time=start_time,
        end_time=end_time,
        recent_days=recent_days,
    )

    endpoint_attempted: list[str] = []

    if use_full_archive:
        endpoint_attempted.append("full_archive")
        full_archive_result = _run_search_endpoint(
            endpoint_url=urls.search_all_url,
            endpoint_label="full_archive",
            headers=headers,
            query=query,
            tweet_fields=tweet_fields,
            start_time=start_time,
            end_time=end_time,
            max_results_per_page=max_results_per_page,
            max_pages=max_pages,
            timeout_seconds=timeout_seconds,
            sleep_seconds=sleep_seconds,
            max_rate_limit_wait_seconds=max_rate_limit_wait_seconds,
            max_429_retries=max_429_retries,
        )

        full_status = full_archive_result.get("status_code")
        full_rows = full_archive_result.get("tweets", [])

        if full_status == 200:
            audit = {
                **context,
                "endpoint_attempted": "full_archive",
                "endpoint_used": "full_archive",
                "status": "ok",
                "status_code": 200,
                "n_rows": len(full_rows),
                "error_summary": "",
            }
            return full_rows, audit

        if full_status in {401, 403, 404}:
            if allow_recent_fallback and recent_eligible:
                endpoint_attempted.append("recent")
                recent_result = _run_search_endpoint(
                    endpoint_url=urls.search_recent_url,
                    endpoint_label="recent",
                    headers=headers,
                    query=query,
                    tweet_fields=tweet_fields,
                    start_time=start_time,
                    end_time=end_time,
                    max_results_per_page=max_results_per_page,
                    max_pages=max_pages,
                    timeout_seconds=timeout_seconds,
                    sleep_seconds=sleep_seconds,
                    max_rate_limit_wait_seconds=max_rate_limit_wait_seconds,
                    max_429_retries=max_429_retries,
                )

                recent_status = recent_result.get("status_code")
                recent_rows = recent_result.get("tweets", [])
                if recent_status == 200:
                    audit = {
                        **context,
                        "endpoint_attempted": "->".join(endpoint_attempted),
                        "endpoint_used": "recent",
                        "status": "ok_fallback_recent",
                        "status_code": 200,
                        "n_rows": len(recent_rows),
                        "error_summary": full_archive_result.get("error_summary", ""),
                    }
                    return recent_rows, audit

                audit = {
                    **context,
                    "endpoint_attempted": "->".join(endpoint_attempted),
                    "endpoint_used": "none",
                    "status": "error_recent_after_full_archive_failure",
                    "status_code": recent_status,
                    "n_rows": 0,
                    "error_summary": recent_result.get("error_summary", ""),
                }
                return [], audit

            audit = {
                **context,
                "endpoint_attempted": "->".join(endpoint_attempted),
                "endpoint_used": "none",
                "status": "skipped_no_full_archive",
                "status_code": full_status,
                "n_rows": 0,
                "error_summary": full_archive_result.get("error_summary", ""),
            }
            return [], audit

        audit = {
            **context,
            "endpoint_attempted": "->".join(endpoint_attempted),
            "endpoint_used": "none",
            "status": "error_full_archive",
            "status_code": full_status,
            "n_rows": 0,
            "error_summary": full_archive_result.get("error_summary", ""),
        }
        return [], audit

    # use_full_archive == False
    if recent_eligible:
        endpoint_attempted.append("recent")
        recent_result = _run_search_endpoint(
            endpoint_url=urls.search_recent_url,
            endpoint_label="recent",
            headers=headers,
            query=query,
            tweet_fields=tweet_fields,
            start_time=start_time,
            end_time=end_time,
            max_results_per_page=max_results_per_page,
            max_pages=max_pages,
            timeout_seconds=timeout_seconds,
            sleep_seconds=sleep_seconds,
            max_rate_limit_wait_seconds=max_rate_limit_wait_seconds,
            max_429_retries=max_429_retries,
        )

        recent_status = recent_result.get("status_code")
        recent_rows = recent_result.get("tweets", [])
        if recent_status == 200:
            audit = {
                **context,
                "endpoint_attempted": "recent",
                "endpoint_used": "recent",
                "status": "ok",
                "status_code": 200,
                "n_rows": len(recent_rows),
                "error_summary": "",
            }
            return recent_rows, audit

        audit = {
            **context,
            "endpoint_attempted": "recent",
            "endpoint_used": "none",
            "status": "error_recent",
            "status_code": recent_status,
            "n_rows": 0,
            "error_summary": recent_result.get("error_summary", ""),
        }
        return [], audit

    audit = {
        **context,
        "endpoint_attempted": "none",
        "endpoint_used": "none",
        "status": "skipped_no_full_archive",
        "status_code": None,
        "n_rows": 0,
        "error_summary": "recent_window_not_eligible",
    }
    return [], audit


def collect_full_archive_query_counts(
    query_specs: list[dict[str, Any]],
    headers: dict[str, str],
    urls: XApiUrls,
    granularity: str = "hour",
    timeout_seconds: int = 60,
    max_rate_limit_wait_seconds: int = 60,
    max_429_retries: int = 3,
) -> pd.DataFrame:
    """Collect one Full-Archive total count row for each query specification."""
    rows: list[dict[str, Any]] = []
    iterable = tqdm(query_specs, total=len(query_specs), desc="Full-Archive Counts")

    for idx, spec in enumerate(iterable, start=1):
        started = time.perf_counter()
        try:
            buckets, audit = count_x_posts_full_archive(
                query=str(spec.get("query", "")),
                start_time=str(spec.get("start_time", "")),
                end_time=str(spec.get("end_time", "")),
                granularity=granularity,
                timeout_seconds=timeout_seconds,
                max_rate_limit_wait_seconds=max_rate_limit_wait_seconds,
                max_429_retries=max_429_retries,
                headers=headers,
                base_url=urls.base_url,
            )
        except Exception as exc:
            buckets = []
            audit = {
                "endpoint": "full_archive_counts",
                "endpoint_url": urls.counts_all_url,
                "status": "exception",
                "status_code": None,
                "total_tweet_count": 0,
                "error_summary": str(exc)[:400],
            }

        seconds = round(time.perf_counter() - started, 3)
        row = {
            "event_id": spec.get("event_id"),
            "event_name": spec.get("event_name"),
            "formal_order": spec.get("formal_order"),
            "media_id": spec.get("media_id"),
            "media_name": spec.get("media_name"),
            "media_handle": spec.get("media_handle"),
            "search_mode": spec.get("search_mode"),
            "term_count": spec.get("term_count"),
            "term_group_names": spec.get("term_group_names"),
            "query": spec.get("query"),
            "start_time": spec.get("start_time"),
            "end_time": spec.get("end_time"),
            "granularity": granularity,
            "endpoint": audit.get("endpoint"),
            "endpoint_url": audit.get("endpoint_url"),
            "status": audit.get("status"),
            "status_code": audit.get("status_code"),
            "bucket_count": len(buckets),
            "total_tweet_count": audit.get("total_tweet_count", 0),
            "error_summary": audit.get("error_summary", ""),
            "seconds": seconds,
            "collected_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        rows.append(row)
        print(
            f"[{idx}/{len(query_specs)}] event={spec.get('event_id')} "
            f"media={spec.get('media_handle')} count={row['total_tweet_count']} "
            f"status={row['status']} seconds={seconds}"
        )

    return _ensure_columns(pd.DataFrame(rows), FULL_ARCHIVE_COUNT_COLUMNS)


def _flatten_source_post_record(
    spec: dict[str, Any],
    tweet: dict[str, Any],
    endpoint_used: str,
) -> dict[str, Any]:
    public_metrics = tweet.get("public_metrics", {}) or {}
    tweet_id = tweet.get("id")

    return {
        "source_type": "media_source_post",
        "source_universe": "costa_rican_media",
        "anchor_type": "media_account",
        "anchor_handle": spec.get("media_handle"),
        "event_id": spec.get("event_id"),
        "event_name": spec.get("event_name"),
        "event_date": spec.get("event_date"),
        "media_id": spec.get("media_id"),
        "media_name": spec.get("media_name"),
        "media_handle": spec.get("media_handle"),
        "search_mode": spec.get("search_mode"),
        "term_batch_id": spec.get("term_batch_id"),
        "term_count": spec.get("term_count"),
        "term_group_names": spec.get("term_group_names"),
        "query": spec.get("query"),
        "query_start_time": spec.get("start_time"),
        "query_end_time": spec.get("end_time"),
        "endpoint_used": endpoint_used,
        "tweet_id": tweet_id,
        "id": tweet_id,
        "source_post_id": tweet_id,
        "text": tweet.get("text"),
        "source_post_text": tweet.get("text"),
        "created_at": tweet.get("created_at"),
        "source_post_created_at": tweet.get("created_at"),
        "author_id": tweet.get("author_id"),
        "conversation_id": tweet.get("conversation_id"),
        "lang": tweet.get("lang"),
        "public_metrics": json.dumps(public_metrics, ensure_ascii=False),
        "source_post_public_metrics": json.dumps(public_metrics, ensure_ascii=False),
        "reply_count": public_metrics.get("reply_count", 0),
        "retweet_count": public_metrics.get("retweet_count", 0),
        "like_count": public_metrics.get("like_count", 0),
        "quote_count": public_metrics.get("quote_count", 0),
        "source_post_url": f"https://x.com/{spec.get('media_handle')}/status/{tweet_id}" if tweet_id else None,
        "collected_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def collect_source_post_candidates(
    query_specs: list[dict[str, Any]],
    headers: dict[str, str],
    urls: XApiUrls,
    use_full_archive: bool,
    max_results_per_page: int = 100,
    max_pages_per_query: int = 5,
    timeout_seconds: int = 60,
    sleep_seconds: float = 0.5,
    max_rate_limit_wait_seconds: int = 30,
    max_429_retries: int = 3,
    recent_days: int = 7,
    allow_recent_fallback: bool = False,
    collection_scope: str = "media_anchored",
    enable_media_source_posts: bool = True,
    dedupe_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tweet_fields = "id,text,created_at,author_id,conversation_id,public_metrics,lang"

    candidate_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    scope = str(collection_scope or "media_anchored").strip().lower()
    if not enable_media_source_posts:
        for spec in query_specs:
            audit_rows.append(
                {
                    "event_id": spec.get("event_id"),
                    "media_id": spec.get("media_id"),
                    "media_handle": spec.get("media_handle"),
                    "search_mode": spec.get("search_mode"),
                    "term_batch_id": spec.get("term_batch_id"),
                    "term_count": spec.get("term_count"),
                    "term_group_names": spec.get("term_group_names"),
                    "query": spec.get("query"),
                    "start_time": spec.get("start_time"),
                    "end_time": spec.get("end_time"),
                    "endpoint_attempted": "none",
                    "endpoint_used": "none",
                    "status": "omitted_by_configuration",
                    "status_code": None,
                    "n_rows": 0,
                    "error_summary": "ENABLE_MEDIA_SOURCE_POSTS=false",
                }
            )
        candidates_df = _ensure_columns(pd.DataFrame(candidate_rows), SOURCE_POST_CANDIDATE_COLUMNS)
        audit_df = _ensure_columns(pd.DataFrame(audit_rows), SOURCE_POST_AUDIT_COLUMNS)
        return candidates_df, audit_df

    for idx, spec in tqdm(
        list(enumerate(query_specs, start=1)),
        total=len(query_specs),
        desc="Collecting source post candidates",
    ):
        if scope == "media_anchored":
            _validate_media_anchored_query_spec(spec)

        query_started = time.perf_counter()
        context = {
            "event_id": spec.get("event_id"),
            "media_id": spec.get("media_id"),
            "media_handle": spec.get("media_handle"),
            "search_mode": spec.get("search_mode"),
            "term_batch_id": spec.get("term_batch_id"),
            "term_count": spec.get("term_count"),
            "term_group_names": spec.get("term_group_names"),
            "query": spec.get("query"),
            "start_time": spec.get("start_time"),
            "end_time": spec.get("end_time"),
        }

        tweets, audit = collect_tweets_with_archive_fallback(
            query=spec.get("query", ""),
            headers=headers,
            urls=urls,
            use_full_archive=use_full_archive,
            start_time=spec.get("start_time"),
            end_time=spec.get("end_time"),
            tweet_fields=tweet_fields,
            context=context,
            max_results_per_page=max_results_per_page,
            max_pages=max_pages_per_query,
            timeout_seconds=timeout_seconds,
            sleep_seconds=sleep_seconds,
            max_rate_limit_wait_seconds=max_rate_limit_wait_seconds,
            max_429_retries=max_429_retries,
            recent_days=recent_days,
            allow_recent_fallback=allow_recent_fallback,
        )

        audit_rows.append(audit)
        elapsed = round(time.perf_counter() - query_started, 3)
        print(
            f"[{idx}/{len(query_specs)}] media={spec.get('media_handle')} "
            f"mode={spec.get('search_mode')} rows={len(tweets)} seconds={elapsed} "
            f"status={audit.get('status')}"
        )

        for tweet in tweets:
            candidate_rows.append(_flatten_source_post_record(spec, tweet, endpoint_used=audit.get("endpoint_used")))

    candidates_df = pd.DataFrame(candidate_rows)
    audit_df = pd.DataFrame(audit_rows)

    candidates_df = _ensure_columns(candidates_df, SOURCE_POST_CANDIDATE_COLUMNS)
    audit_df = _ensure_columns(audit_df, SOURCE_POST_AUDIT_COLUMNS)

    if candidates_df.empty:
        return candidates_df, audit_df

    for metric_col in ["reply_count", "retweet_count", "like_count", "quote_count"]:
        candidates_df[metric_col] = _coerce_numeric(candidates_df[metric_col], default=0).astype(float)

    # Keep trace of search metadata when the same post appears in multiple queries.
    modes_by_id = (
        candidates_df.groupby("id", dropna=False)["search_mode"]
        .agg(lambda s: "|".join(sorted({str(x) for x in s if pd.notna(x)})))
        .rename("search_modes_found")
    )
    batches_by_id = (
        candidates_df.groupby("id", dropna=False)["term_batch_id"]
        .agg(lambda s: "|".join(sorted({str(x) for x in s if pd.notna(x)})))
        .rename("term_batch_id")
    )

    dedupe_subset = dedupe_columns or ["id"]
    missing_dedupe_columns = [c for c in dedupe_subset if c not in candidates_df.columns]
    if missing_dedupe_columns:
        raise ValueError(f"Missing candidate dedupe columns: {missing_dedupe_columns}")

    candidates_df = (
        candidates_df.sort_values(
            ["reply_count", "quote_count", "retweet_count", "like_count"],
            ascending=False,
        )
        .drop_duplicates(subset=dedupe_subset, keep="first")
        .reset_index(drop=True)
    )

    candidates_df = candidates_df.drop(columns=["search_modes_found"], errors="ignore")
    candidates_df = candidates_df.drop(columns=["term_batch_id"], errors="ignore")

    candidates_df = candidates_df.merge(
        modes_by_id,
        left_on="id",
        right_index=True,
        how="left",
    )
    candidates_df = candidates_df.merge(
        batches_by_id,
        left_on="id",
        right_index=True,
        how="left",
    )

    candidates_df = _ensure_columns(candidates_df, SOURCE_POST_CANDIDATE_COLUMNS)
    return candidates_df, audit_df


def build_event_term_index(
    events: list[dict[str, Any]],
    term_groups: dict[str, Any],
    base_query_terms: list[str] | None = None,
) -> dict[str, list[str]]:
    base_terms = base_query_terms or DEFAULT_BASE_QUERY_TERMS
    index: dict[str, list[str]] = {}

    for event in events:
        event_id = str(event.get("event_id"))
        profile = event.get("search_profile", {}) if isinstance(event, dict) else {}
        include_groups = profile.get("include_groups", []) or []
        optional_groups = profile.get("optional_groups", []) or []

        event_terms: list[str] = []
        for group_name in include_groups + optional_groups:
            event_terms.extend(get_terms_from_group(term_groups, str(group_name)))

        all_terms = dedupe_keep_order(list(base_terms) + event_terms)
        index[event_id] = [_normalize_text_for_matching(term) for term in all_terms]

    return index


def _compute_keyword_hits(text: Any, terms: list[str]) -> int:
    normalized_text = _normalize_text_for_matching(text)
    if not normalized_text:
        return 0
    return int(sum(1 for term in terms if term and term in normalized_text))


def _coerce_manual_keep(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None

    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "si", "s"}:
        return 1
    if text in {"0", "false", "f", "no", "n"}:
        return 0

    try:
        n = int(float(text))
        return 1 if n > 0 else 0
    except Exception:
        return None


def select_source_posts_semiautomatic(
    candidates_df: pd.DataFrame,
    events: list[dict[str, Any]],
    term_groups: dict[str, Any],
    base_query_terms: list[str] | None = None,
    min_reply_count: int = 5,
    min_quote_count: int = 1,
    top_posts_per_event_media: int = 10,
    manual_flags_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if candidates_df.empty:
        empty_source = pd.DataFrame(columns=SOURCE_POST_FINAL_COLUMNS)
        empty_review = pd.DataFrame(columns=SOURCE_POST_REVIEW_COLUMNS)
        return empty_source, empty_review

    work = candidates_df.copy()
    work = _ensure_columns(
        work,
        [
            *SOURCE_POST_CANDIDATE_COLUMNS,
            "search_modes_found",
        ],
    )

    for metric_col in ["reply_count", "like_count", "quote_count", "retweet_count"]:
        if metric_col not in work.columns:
            work[metric_col] = 0
        work[metric_col] = _coerce_numeric(work[metric_col], default=0)

    event_terms_index = build_event_term_index(
        events=events,
        term_groups=term_groups,
        base_query_terms=base_query_terms,
    )

    work["keyword_hits"] = work.apply(
        lambda row: _compute_keyword_hits(
            row.get("text", ""),
            event_terms_index.get(str(row.get("event_id")), []),
        ),
        axis=1,
    )

    work["engagement_score"] = (
        work["reply_count"] * 3
        + work["quote_count"] * 2
        + work["retweet_count"]
        + work["like_count"] * 0.25
    )

    work["semiauto_keep"] = 0
    work["selection_reason"] = "not_selected"

    top_n = max(1, int(top_posts_per_event_media))
    min_reply = max(0, int(min_reply_count))
    min_quote = max(0, int(min_quote_count))

    grouped = work.groupby(["event_id", "media_id"], dropna=False)
    for (event_id, media_id), group in grouped:
        group_sorted = group.sort_values(
            ["reply_count", "quote_count", "engagement_score", "created_at"],
            ascending=[False, False, False, True],
        )

        eligible = group_sorted[
            (group_sorted["reply_count"] >= min_reply)
            | (group_sorted["quote_count"] >= min_quote)
        ]
        selected_index: list[int] = []

        if len(eligible) >= top_n:
            selected = eligible.head(top_n)
            selected_index.extend(selected.index.tolist())
            work.loc[selected.index, "selection_reason"] = "min_reply_or_quote_count"
        elif len(eligible) > 0:
            selected_index.extend(eligible.index.tolist())
            remaining_slots = top_n - len(eligible)
            fallback = group_sorted[~group_sorted.index.isin(eligible.index)].head(remaining_slots)
            selected_index.extend(fallback.index.tolist())
            work.loc[eligible.index, "selection_reason"] = "min_reply_or_quote_count"
            work.loc[fallback.index, "selection_reason"] = "fallback_engagement"
        else:
            fallback = group_sorted.head(top_n)
            selected_index.extend(fallback.index.tolist())
            work.loc[fallback.index, "selection_reason"] = "fallback_engagement"

        if selected_index:
            work.loc[selected_index, "semiauto_keep"] = 1

    if "source_post_text" not in work.columns:
        work["source_post_text"] = work.get("text")
    if "source_post_created_at" not in work.columns:
        work["source_post_created_at"] = work.get("created_at")

    review_cols = SOURCE_POST_REVIEW_COLUMNS.copy()

    for col in review_cols:
        if col not in work.columns:
            work[col] = pd.NA

    if "search_modes_found" not in work.columns:
        work["search_modes_found"] = pd.NA

    work["manual_keep"] = pd.NA

    if manual_flags_path and manual_flags_path.exists():
        manual_df = pd.read_csv(manual_flags_path)
        manual_key = "id" if "id" in manual_df.columns else "source_post_id" if "source_post_id" in manual_df.columns else None
        if manual_key is not None and "manual_keep" in manual_df.columns:
            manual_df = manual_df[[manual_key, "manual_keep"]].copy()
            manual_df["manual_keep"] = manual_df["manual_keep"].map(_coerce_manual_keep)
            work = work.merge(
                manual_df,
                on=manual_key,
                how="left",
                suffixes=("", "_manual"),
            )
            if "manual_keep_manual" in work.columns:
                work["manual_keep"] = work["manual_keep_manual"].combine_first(work["manual_keep"])
                work = work.drop(columns=["manual_keep_manual"])

    work["final_keep"] = work["semiauto_keep"]
    manual_mask = work["manual_keep"].notna()
    work.loc[manual_mask, "final_keep"] = work.loc[manual_mask, "manual_keep"].astype(int)

    source_posts = work[work["final_keep"] == 1].copy()
    source_posts = source_posts.sort_values(
        ["event_id", "media_id", "reply_count", "engagement_score"],
        ascending=[True, True, False, False],
    )
    source_posts = source_posts.drop_duplicates(subset=["id"], keep="first").reset_index(drop=True)

    if "source_post_id" not in source_posts.columns:
        source_posts["source_post_id"] = source_posts["id"]

    review_df = work[SOURCE_POST_REVIEW_COLUMNS].copy()

    source_posts = _ensure_columns(source_posts, SOURCE_POST_FINAL_COLUMNS)
    review_df = _ensure_columns(review_df, SOURCE_POST_REVIEW_COLUMNS)

    return source_posts, review_df


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _save_replies_checkpoint(
    clean_rows: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    stats_rows: list[dict[str, Any]],
    interim_dir: Path,
) -> None:
    interim_dir.mkdir(parents=True, exist_ok=True)

    clean_df = pd.DataFrame(clean_rows)
    if not clean_df.empty and "reply_id" in clean_df.columns:
        clean_df = clean_df.drop_duplicates(subset=["reply_id"], keep="first")
    clean_df.to_csv(interim_dir / "replies_checkpoint_clean.csv", index=False)

    _write_jsonl(interim_dir / "replies_checkpoint_raw.jsonl", raw_rows)

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(interim_dir / "replies_checkpoint_stats.csv", index=False)


def collect_replies_for_source_posts(
    source_posts_df: pd.DataFrame,
    headers: dict[str, str],
    urls: XApiUrls,
    hash_salt: str,
    use_full_archive: bool,
    reply_window_hours: int = 72,
    max_replies_per_post: int = 50,
    max_pages_per_post: int = 5,
    max_results_per_page: int = 100,
    timeout_seconds: int = 60,
    sleep_seconds: float = 0.2,
    max_rate_limit_wait_seconds: int = 30,
    max_429_retries: int = 3,
    recent_days: int = 7,
    allow_recent_fallback: bool = False,
    checkpoint_every: int = 10,
    interim_dir: Path | None = None,
    raw_dir: Path | None = None,
    stop_on_error: bool = False,
    stop_on_credits_depleted: bool = True,
) -> tuple[pd.DataFrame, list[dict[str, Any]], pd.DataFrame]:
    if source_posts_df.empty:
        return pd.DataFrame(), [], pd.DataFrame()

    interim_dir = interim_dir or Path.cwd()
    interim_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = raw_dir or interim_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = interim_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    work = source_posts_df.copy()
    work["source_post_id"] = work["source_post_id"].map(normalize_source_post_id)
    work = work[work["source_post_id"].notna()].copy()
    if work.empty:
        return pd.DataFrame(), [], pd.DataFrame()

    tweet_fields = (
        "id,text,created_at,author_id,conversation_id,"
        "in_reply_to_user_id,referenced_tweets,public_metrics,lang"
    )

    all_clean_rows: list[dict[str, Any]] = []
    all_raw_rows: list[dict[str, Any]] = []
    stats_rows: list[dict[str, Any]] = []
    seen_reply_ids: set[str] = set()

    iterable = tqdm(work.reset_index(drop=True).iterrows(), total=len(work), desc="Collecting replies")

    for idx, post in iterable:
        post_started = time.perf_counter()

        source_post_id = str(post.get("source_post_id"))
        event_id = str(post.get("event_id", ""))
        event_name = post.get("event_name")
        formal_event_memberships, formal_event_names, formal_event_count = (
            _formal_event_context(post, event_id)
        )
        media_id = str(post.get("media_id", ""))
        media_name = post.get("media_name")
        media_handle = str(post.get("media_handle", ""))

        source_created_at = post.get("source_post_created_at")
        source_dt = parse_datetime_utc(source_created_at)

        if source_dt is None:
            status = "skipped_missing_source_post_created_at"
            stats_rows.append(
                {
                    "event_id": event_id,
                    "formal_event_memberships": formal_event_memberships,
                    "formal_event_count": formal_event_count,
                    "media_id": media_id,
                    "media_handle": media_handle,
                    "source_post_id": source_post_id,
                    "query": f"conversation_id:{source_post_id} -is:retweet",
                    "start_time": None,
                    "end_time": None,
                    "endpoint_attempted": "none",
                    "endpoint_used": "none",
                    "status": status,
                    "status_code": None,
                    "n_rows_api": 0,
                    "n_rows_kept": 0,
                    "error_summary": "missing_source_post_created_at",
                    "seconds": round(time.perf_counter() - post_started, 3),
                }
            )
            continue

        start_time = to_rfc3339_utc(source_dt)
        end_time = to_rfc3339_utc(source_dt + timedelta(hours=max(1, int(reply_window_hours))))

        query = f"conversation_id:{source_post_id} -is:retweet"

        context = {
            "event_id": event_id,
            "media_id": media_id,
            "media_handle": media_handle,
            "query": query,
            "start_time": start_time,
            "end_time": end_time,
        }

        page_size = max(10, min(100, int(max_results_per_page)))
        target_reply_count = max(1, int(max_replies_per_post))
        # Full-Archive conversation search can return the source post in addition to replies.
        pages_needed = max(1, (target_reply_count + 1 + page_size - 1) // page_size)
        effective_max_pages = min(max(1, int(max_pages_per_post)), pages_needed)

        tweets, audit = collect_tweets_with_archive_fallback(
            query=query,
            headers=headers,
            urls=urls,
            use_full_archive=use_full_archive,
            start_time=start_time,
            end_time=end_time,
            tweet_fields=tweet_fields,
            context=context,
            max_results_per_page=page_size,
            max_pages=effective_max_pages,
            timeout_seconds=timeout_seconds,
            sleep_seconds=sleep_seconds,
            max_rate_limit_wait_seconds=max_rate_limit_wait_seconds,
            max_429_retries=max_429_retries,
            recent_days=recent_days,
            allow_recent_fallback=allow_recent_fallback,
        )

        keep_count = 0
        for reply in tweets:
            reply_id = normalize_source_post_id(reply.get("id"))
            if not reply_id:
                continue

            if reply_id == source_post_id:
                continue

            if str(reply.get("conversation_id")) != source_post_id:
                continue

            if reply_id in seen_reply_ids:
                continue

            seen_reply_ids.add(reply_id)

            public_metrics = reply.get("public_metrics", {}) or {}
            reply_author_id_hash = hashlib.sha256(
                f"{hash_salt}:{reply.get('author_id')}".encode("utf-8")
            ).hexdigest()
            reply_raw = dict(reply)
            reply_raw.pop("author_id", None)
            reply_raw["author_id_hash"] = reply_author_id_hash
            all_clean_rows.append(
                {
                    "source_type": "reply_to_media_post",
                    "source_universe": "media_anchored",
                    "anchor_type": "media_source_post",
                    "anchor_post_id": source_post_id,
                    "anchor_media_id": media_id,
                    "anchor_media_handle": media_handle,
                    "event_id": event_id,
                    "event_name": event_name,
                    "formal_event_memberships": formal_event_memberships,
                    "formal_event_names": formal_event_names,
                    "formal_event_count": formal_event_count,
                    "media_id": media_id,
                    "media_name": media_name,
                    "media_handle": media_handle,
                    "source_post_id": source_post_id,
                    "tweet_id": reply_id,
                    "reply_id": reply_id,
                    "text": reply.get("text"),
                    "reply_text": reply.get("text"),
                    "created_at": reply.get("created_at"),
                    "reply_created_at": reply.get("created_at"),
                    "reply_author_id_hash": reply_author_id_hash,
                    "lang": reply.get("lang"),
                    "public_metrics": json.dumps(public_metrics, ensure_ascii=False),
                    "conversation_id": reply.get("conversation_id"),
                    "in_reply_to_user_id": reply.get("in_reply_to_user_id"),
                    "referenced_tweets": json.dumps(reply.get("referenced_tweets", []), ensure_ascii=False),
                    "query": query,
                    "collected_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
            )

            all_raw_rows.append(
                {
                    "event_id": event_id,
                    "formal_event_memberships": formal_event_memberships,
                    "formal_event_names": formal_event_names,
                    "formal_event_count": formal_event_count,
                    "media_id": media_id,
                    "source_post_id": source_post_id,
                    "query": query,
                    "start_time": start_time,
                    "end_time": end_time,
                    "collected_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "reply_raw": reply_raw,
                }
            )

            keep_count += 1
            if keep_count >= max(1, int(max_replies_per_post)):
                break

        seconds = round(time.perf_counter() - post_started, 3)

        status = audit.get("status", "unknown")
        stats_rows.append(
            {
                "event_id": event_id,
                "formal_event_memberships": formal_event_memberships,
                "formal_event_count": formal_event_count,
                "media_id": media_id,
                "media_handle": media_handle,
                "source_post_id": source_post_id,
                "query": query,
                "start_time": start_time,
                "end_time": end_time,
                "endpoint_attempted": audit.get("endpoint_attempted"),
                "endpoint_used": audit.get("endpoint_used"),
                "status": status,
                "status_code": audit.get("status_code"),
                "n_rows_api": audit.get("n_rows", 0),
                "n_rows_kept": keep_count,
                "error_summary": audit.get("error_summary", ""),
                "seconds": seconds,
            }
        )

        print(
            f"[{idx + 1}/{len(work)}] source_post_id={source_post_id} "
            f"replies={keep_count} seconds={seconds} status={status}"
        )

        if audit.get("status_code") == 402 and stop_on_credits_depleted:
            _save_replies_checkpoint(
                clean_rows=all_clean_rows,
                raw_rows=all_raw_rows,
                stats_rows=stats_rows,
                interim_dir=checkpoint_dir,
            )
            print(
                "[STOP] X API devolvio HTTP 402; se guardo el checkpoint "
                "y se detuvo el batch para evitar solicitudes adicionales."
            )
            break

        if status.startswith("error") and stop_on_error:
            _save_replies_checkpoint(
                clean_rows=all_clean_rows,
                raw_rows=all_raw_rows,
                stats_rows=stats_rows,
                interim_dir=checkpoint_dir,
            )
            raise RuntimeError(f"Failed source_post_id={source_post_id}: {audit}")

        if checkpoint_every > 0 and ((idx + 1) % checkpoint_every == 0):
            _save_replies_checkpoint(
                clean_rows=all_clean_rows,
                raw_rows=all_raw_rows,
                stats_rows=stats_rows,
                interim_dir=checkpoint_dir,
            )

    clean_df = pd.DataFrame(all_clean_rows)
    if not clean_df.empty and "reply_id" in clean_df.columns:
        clean_df = clean_df.drop_duplicates(subset=["reply_id"], keep="first").reset_index(drop=True)
    clean_df = _ensure_columns(clean_df, REPLIES_CLEAN_COLUMNS)

    stats_df = _ensure_columns(pd.DataFrame(stats_rows), REPLIES_STATS_COLUMNS)

    clean_df.to_csv(interim_dir / "replies_clean.csv", index=False)
    _write_jsonl(raw_dir / "replies_raw.jsonl", all_raw_rows)
    stats_df.to_csv(interim_dir / "replies_stats.csv", index=False)

    return clean_df, all_raw_rows, stats_df


def _save_quotes_checkpoint(
    clean_rows: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    stats_rows: list[dict[str, Any]],
    checkpoint_dir: Path,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    clean_df = _ensure_columns(pd.DataFrame(clean_rows), QUOTE_CLEAN_COLUMNS)
    if not clean_df.empty and "tweet_id" in clean_df.columns:
        clean_df = clean_df.drop_duplicates(subset=["tweet_id"], keep="first")
    clean_df.to_csv(checkpoint_dir / "quote_posts_checkpoint_clean.csv", index=False)
    _write_jsonl(checkpoint_dir / "quote_posts_checkpoint_raw.jsonl", raw_rows)
    stats_df = _ensure_columns(pd.DataFrame(stats_rows), QUOTE_STATS_COLUMNS)
    stats_df.to_csv(checkpoint_dir / "quote_posts_checkpoint_stats.csv", index=False)


def _save_generic_query_checkpoint(
    rows: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
    checkpoint_dir: Path,
    clean_name: str,
    audit_name: str,
    clean_columns: list[str] | None = None,
    audit_columns: list[str] | None = None,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    clean_df = pd.DataFrame(rows)
    if clean_columns:
        clean_df = _ensure_columns(clean_df, clean_columns)
    clean_df.to_csv(checkpoint_dir / clean_name, index=False)
    audit_df = pd.DataFrame(audit_rows)
    if audit_columns:
        audit_df = _ensure_columns(audit_df, audit_columns)
    audit_df.to_csv(checkpoint_dir / audit_name, index=False)


def collect_quote_tweets_for_source_posts(
    source_posts_df: pd.DataFrame,
    headers: dict[str, str],
    urls: XApiUrls,
    min_quote_count: int = 1,
    max_quotes_per_post: int = 100,
    max_pages_per_post: int = 10,
    max_results_per_page: int = 100,
    timeout_seconds: int = 60,
    sleep_seconds: float = 0.2,
    max_rate_limit_wait_seconds: int = 30,
    max_429_retries: int = 3,
    checkpoint_every: int = 10,
    interim_dir: Path | None = None,
    raw_dir: Path | None = None,
    stop_on_error: bool = False,
) -> tuple[pd.DataFrame, list[dict[str, Any]], pd.DataFrame]:
    if source_posts_df.empty:
        return (
            _ensure_columns(pd.DataFrame(), QUOTE_CLEAN_COLUMNS),
            [],
            _ensure_columns(pd.DataFrame(), QUOTE_STATS_COLUMNS),
        )

    interim_dir = interim_dir or Path.cwd()
    interim_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = raw_dir or interim_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = interim_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    work = source_posts_df.copy()
    work["source_post_id"] = work["source_post_id"].map(normalize_source_post_id)
    work = work[work["source_post_id"].notna()].copy()
    if work.empty:
        return (
            _ensure_columns(pd.DataFrame(), QUOTE_CLEAN_COLUMNS),
            [],
            _ensure_columns(pd.DataFrame(), QUOTE_STATS_COLUMNS),
        )

    if "quote_count" not in work.columns:
        work["quote_count"] = 0
    work["quote_count"] = _coerce_numeric(work["quote_count"], default=0)
    work = work[work["quote_count"] >= max(0, int(min_quote_count))].copy()

    if work.empty:
        empty_clean = _ensure_columns(pd.DataFrame(), QUOTE_CLEAN_COLUMNS)
        empty_stats = _ensure_columns(pd.DataFrame(), QUOTE_STATS_COLUMNS)
        empty_clean.to_csv(interim_dir / "quote_posts_clean.csv", index=False)
        _write_jsonl(raw_dir / "quote_posts_raw.jsonl", [])
        empty_stats.to_csv(interim_dir / "quote_posts_stats.csv", index=False)
        return empty_clean, [], empty_stats

    tweet_fields = (
        "id,text,created_at,author_id,conversation_id,"
        "in_reply_to_user_id,referenced_tweets,public_metrics,lang"
    )

    clean_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    stats_rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    iterable = tqdm(work.reset_index(drop=True).iterrows(), total=len(work), desc="Collecting quote tweets")
    for idx, post in iterable:
        started = time.perf_counter()
        source_post_id = str(post.get("source_post_id"))
        event_id = str(post.get("event_id", ""))
        event_name = post.get("event_name")
        formal_event_memberships, formal_event_names, formal_event_count = (
            _formal_event_context(post, event_id)
        )
        media_id = str(post.get("media_id", ""))
        media_name = post.get("media_name")
        media_handle = str(post.get("media_handle", ""))

        endpoint_url = get_quote_tweets_url(source_post_id, base_url=urls.base_url)
        query = f"quotes_of:{source_post_id}"
        next_token: str | None = None
        retries_429 = 0
        status = "ok"
        status_code: int | None = None
        error_summary = ""
        rows_api = 0
        rows_kept = 0
        pages_fetched = 0

        page_limit = max(1, int(max_pages_per_post))
        while pages_fetched < page_limit:
            params = {
                "max_results": max(10, min(100, int(max_results_per_page))),
                "tweet.fields": tweet_fields,
            }
            if next_token:
                params["pagination_token"] = next_token

            try:
                response = requests.get(
                    endpoint_url,
                    headers=headers,
                    params=params,
                    timeout=timeout_seconds,
                )
            except Exception as exc:
                status = "request_exception"
                error_summary = str(exc)
                break

            status_code = response.status_code

            if status_code == 429:
                retries_429 += 1
                if retries_429 > max_429_retries:
                    status = "rate_limit_retry_exceeded"
                    error_summary = "rate_limit_retry_exceeded"
                    break
                wait_seconds = _compute_wait_seconds(response, max_rate_limit_wait_seconds)
                time.sleep(wait_seconds)
                continue

            if status_code != 200:
                status = "error_quote_endpoint"
                error_summary = summarize_api_error(safe_json(response))
                break

            payload = safe_json(response)
            data = payload.get("data", []) or []
            meta = payload.get("meta", {}) or {}
            rows_api += len(data)
            pages_fetched += 1
            retries_429 = 0

            for tweet in data:
                quote_id = normalize_source_post_id(tweet.get("id"))
                if not quote_id:
                    continue
                if quote_id in seen_ids:
                    continue

                seen_ids.add(quote_id)
                public_metrics = tweet.get("public_metrics", {}) or {}
                clean_rows.append(
                    {
                        "source_type": "quote_of_media_post",
                        "source_universe": "media_anchored",
                        "anchor_type": "media_source_post",
                        "anchor_post_id": source_post_id,
                        "anchor_media_id": media_id,
                        "anchor_media_handle": media_handle,
                        "source_post_id": source_post_id,
                        "event_id": event_id,
                        "event_name": event_name,
                        "formal_event_memberships": formal_event_memberships,
                        "formal_event_names": formal_event_names,
                        "formal_event_count": formal_event_count,
                        "media_id": media_id,
                        "media_name": media_name,
                        "media_handle": media_handle,
                        "tweet_id": quote_id,
                        "text": tweet.get("text"),
                        "created_at": tweet.get("created_at"),
                        "author_id": tweet.get("author_id"),
                        "conversation_id": tweet.get("conversation_id"),
                        "referenced_tweets": json.dumps(tweet.get("referenced_tweets", []), ensure_ascii=False),
                        "public_metrics": json.dumps(public_metrics, ensure_ascii=False),
                        "lang": tweet.get("lang"),
                        "query": query,
                        "collected_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    }
                )
                raw_rows.append(
                    {
                        "event_id": event_id,
                        "formal_event_memberships": formal_event_memberships,
                        "formal_event_names": formal_event_names,
                        "formal_event_count": formal_event_count,
                        "media_id": media_id,
                        "source_post_id": source_post_id,
                        "query": query,
                        "endpoint_url": endpoint_url,
                        "collected_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "quote_raw": tweet,
                    }
                )
                rows_kept += 1
                if rows_kept >= max(1, int(max_quotes_per_post)):
                    break

            if rows_kept >= max(1, int(max_quotes_per_post)):
                break

            next_token = meta.get("next_token")
            if not next_token:
                break
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

        seconds = round(time.perf_counter() - started, 3)
        if status == "ok" and status_code not in {None, 200}:
            status = "error_quote_endpoint"
        if status.startswith("error") and stop_on_error:
            raise RuntimeError(
                f"Failed quote collection source_post_id={source_post_id} "
                f"status={status} code={status_code} detail={error_summary}"
            )

        stats_rows.append(
            {
                "event_id": event_id,
                "formal_event_memberships": formal_event_memberships,
                "formal_event_count": formal_event_count,
                "media_id": media_id,
                "media_handle": media_handle,
                "source_post_id": source_post_id,
                "endpoint_used": "quote_tweets",
                "status": status,
                "status_code": status_code,
                "n_rows_api": rows_api,
                "n_rows_kept": rows_kept,
                "error_summary": error_summary,
                "seconds": seconds,
            }
        )
        print(
            f"[{idx + 1}/{len(work)}] source_post_id={source_post_id} "
            f"quotes={rows_kept} seconds={seconds} status={status}"
        )

        if checkpoint_every > 0 and ((idx + 1) % checkpoint_every == 0):
            _save_quotes_checkpoint(
                clean_rows=clean_rows,
                raw_rows=raw_rows,
                stats_rows=stats_rows,
                checkpoint_dir=checkpoint_dir,
            )

    clean_df = _ensure_columns(pd.DataFrame(clean_rows), QUOTE_CLEAN_COLUMNS)
    if not clean_df.empty:
        clean_df = clean_df.drop_duplicates(subset=["tweet_id"], keep="first").reset_index(drop=True)
    stats_df = _ensure_columns(pd.DataFrame(stats_rows), QUOTE_STATS_COLUMNS)

    clean_df.to_csv(interim_dir / "quote_posts_clean.csv", index=False)
    _write_jsonl(raw_dir / "quote_posts_raw.jsonl", raw_rows)
    stats_df.to_csv(interim_dir / "quote_posts_stats.csv", index=False)

    return clean_df, raw_rows, stats_df


def _flatten_search_tweet_common(tweet: dict[str, Any]) -> dict[str, Any]:
    public_metrics = tweet.get("public_metrics", {}) or {}
    return {
        "tweet_id": normalize_source_post_id(tweet.get("id")),
        "text": tweet.get("text"),
        "created_at": tweet.get("created_at"),
        "author_id": tweet.get("author_id"),
        "conversation_id": tweet.get("conversation_id"),
        "lang": tweet.get("lang"),
        "reply_count": public_metrics.get("reply_count", 0),
        "quote_count": public_metrics.get("quote_count", 0),
        "retweet_count": public_metrics.get("retweet_count", 0),
        "like_count": public_metrics.get("like_count", 0),
        "public_metrics": json.dumps(public_metrics, ensure_ascii=False),
        "referenced_tweets": json.dumps(tweet.get("referenced_tweets", []), ensure_ascii=False),
    }


def collect_organic_event_posts(
    query_specs: list[dict[str, Any]],
    headers: dict[str, str],
    urls: XApiUrls,
    use_full_archive: bool,
    max_results_per_query: int = 500,
    max_pages_per_query: int = 10,
    timeout_seconds: int = 60,
    sleep_seconds: float = 0.2,
    max_rate_limit_wait_seconds: int = 30,
    max_429_retries: int = 3,
    recent_days: int = 7,
    allow_recent_fallback: bool = False,
    checkpoint_every: int = 10,
    interim_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    interim_dir = interim_dir or Path.cwd()
    interim_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = interim_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    tweet_fields = (
        "id,text,created_at,author_id,conversation_id,"
        "in_reply_to_user_id,referenced_tweets,public_metrics,lang"
    )

    iterable = tqdm(query_specs, total=len(query_specs), desc="Collecting organic event posts")
    for idx, spec in enumerate(iterable, start=1):
        started = time.perf_counter()
        context = {
            "event_id": spec.get("event_id"),
            "media_id": None,
            "media_handle": None,
            "query": spec.get("query"),
            "start_time": spec.get("start_time"),
            "end_time": spec.get("end_time"),
        }

        tweets, audit = collect_tweets_with_archive_fallback(
            query=spec.get("query", ""),
            headers=headers,
            urls=urls,
            use_full_archive=use_full_archive,
            start_time=spec.get("start_time"),
            end_time=spec.get("end_time"),
            tweet_fields=tweet_fields,
            context=context,
            max_results_per_page=max(10, min(500, int(max_results_per_query))),
            max_pages=max_pages_per_query,
            timeout_seconds=timeout_seconds,
            sleep_seconds=sleep_seconds,
            max_rate_limit_wait_seconds=max_rate_limit_wait_seconds,
            max_429_retries=max_429_retries,
            recent_days=recent_days,
            allow_recent_fallback=allow_recent_fallback,
        )

        audit_rows.append(
            {
                **audit,
                "source_type": "organic_event_post",
                "seconds": round(time.perf_counter() - started, 3),
            }
        )
        for tweet in tweets:
            common = _flatten_search_tweet_common(tweet)
            if not common.get("tweet_id"):
                continue
            rows.append(
                {
                    "source_type": "organic_event_post",
                    "source_universe": "open_x_search",
                    "anchor_type": "none",
                    "anchor_post_id": pd.NA,
                    "anchor_media_id": pd.NA,
                    "anchor_media_handle": pd.NA,
                    "event_id": spec.get("event_id"),
                    "event_name": spec.get("event_name"),
                    "tweet_id": common["tweet_id"],
                    "text": common["text"],
                    "created_at": common["created_at"],
                    "author_id": common["author_id"],
                    "conversation_id": common["conversation_id"],
                    "lang": common["lang"],
                    "reply_count": common["reply_count"],
                    "quote_count": common["quote_count"],
                    "retweet_count": common["retweet_count"],
                    "like_count": common["like_count"],
                    "public_metrics": common["public_metrics"],
                    "query": spec.get("query"),
                    "collected_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
            )

        print(
            f"[{idx}/{len(query_specs)}] event={spec.get('event_id')} "
            f"rows={audit.get('n_rows', 0)} status={audit.get('status')} "
            f"seconds={round(time.perf_counter() - started, 3)}"
        )
        if checkpoint_every > 0 and (idx % checkpoint_every == 0):
            _save_generic_query_checkpoint(
                rows=rows,
                audit_rows=audit_rows,
                checkpoint_dir=checkpoint_dir,
                clean_name="organic_event_posts_checkpoint.csv",
                audit_name="organic_event_posts_audit_checkpoint.csv",
                clean_columns=ORGANIC_POST_COLUMNS,
                audit_columns=ORGANIC_AUDIT_COLUMNS,
            )

    posts_df = _ensure_columns(pd.DataFrame(rows), ORGANIC_POST_COLUMNS)
    if not posts_df.empty:
        posts_df = posts_df.drop_duplicates(subset=["tweet_id"], keep="first").reset_index(drop=True)

    audit_df = _ensure_columns(pd.DataFrame(audit_rows), ORGANIC_AUDIT_COLUMNS)
    posts_df.to_csv(interim_dir / "organic_event_posts.csv", index=False)
    audit_df.to_csv(interim_dir / "organic_event_posts_audit.csv", index=False)
    return posts_df, audit_df


def collect_candidate_interactions(
    query_specs: list[dict[str, Any]],
    headers: dict[str, str],
    urls: XApiUrls,
    use_full_archive: bool,
    max_results_per_query: int = 500,
    max_pages_per_query: int = 10,
    timeout_seconds: int = 60,
    sleep_seconds: float = 0.2,
    max_rate_limit_wait_seconds: int = 30,
    max_429_retries: int = 3,
    recent_days: int = 7,
    allow_recent_fallback: bool = False,
    checkpoint_every: int = 10,
    interim_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    interim_dir = interim_dir or Path.cwd()
    interim_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = interim_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    tweet_fields = (
        "id,text,created_at,author_id,conversation_id,"
        "in_reply_to_user_id,referenced_tweets,public_metrics,lang"
    )

    iterable = tqdm(query_specs, total=len(query_specs), desc="Collecting candidate interactions")
    for idx, spec in enumerate(iterable, start=1):
        started = time.perf_counter()
        context = {
            "event_id": spec.get("event_id"),
            "media_id": None,
            "media_handle": spec.get("target_handle"),
            "query": spec.get("query"),
            "start_time": spec.get("start_time"),
            "end_time": spec.get("end_time"),
        }

        tweets, audit = collect_tweets_with_archive_fallback(
            query=spec.get("query", ""),
            headers=headers,
            urls=urls,
            use_full_archive=use_full_archive,
            start_time=spec.get("start_time"),
            end_time=spec.get("end_time"),
            tweet_fields=tweet_fields,
            context=context,
            max_results_per_page=max(10, min(500, int(max_results_per_query))),
            max_pages=max_pages_per_query,
            timeout_seconds=timeout_seconds,
            sleep_seconds=sleep_seconds,
            max_rate_limit_wait_seconds=max_rate_limit_wait_seconds,
            max_429_retries=max_429_retries,
            recent_days=recent_days,
            allow_recent_fallback=allow_recent_fallback,
        )

        audit_rows.append(
            {
                **audit,
                "source_type": "reply_or_mention_to_candidate",
                "target_actor_id": spec.get("target_actor_id"),
                "target_handle": spec.get("target_handle"),
                "query_mode": spec.get("query_mode"),
                "seconds": round(time.perf_counter() - started, 3),
            }
        )
        for tweet in tweets:
            common = _flatten_search_tweet_common(tweet)
            if not common.get("tweet_id"):
                continue
            rows.append(
                {
                    "source_type": "reply_or_mention_to_candidate",
                    "source_universe": "open_x_search",
                    "anchor_type": "candidate_account",
                    "anchor_post_id": common.get("conversation_id"),
                    "anchor_media_id": pd.NA,
                    "anchor_media_handle": pd.NA,
                    "target_actor_id": spec.get("target_actor_id"),
                    "target_actor_name": spec.get("target_actor_name"),
                    "target_handle": spec.get("target_handle"),
                    "event_id": spec.get("event_id"),
                    "event_name": spec.get("event_name"),
                    "tweet_id": common["tweet_id"],
                    "text": common["text"],
                    "created_at": common["created_at"],
                    "author_id": common["author_id"],
                    "conversation_id": common["conversation_id"],
                    "lang": common["lang"],
                    "public_metrics": common["public_metrics"],
                    "query": spec.get("query"),
                    "query_mode": spec.get("query_mode"),
                    "collected_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
            )

        print(
            f"[{idx}/{len(query_specs)}] actor={spec.get('target_handle')} "
            f"mode={spec.get('query_mode')} rows={audit.get('n_rows', 0)} "
            f"status={audit.get('status')} seconds={round(time.perf_counter() - started, 3)}"
        )
        if checkpoint_every > 0 and (idx % checkpoint_every == 0):
            _save_generic_query_checkpoint(
                rows=rows,
                audit_rows=audit_rows,
                checkpoint_dir=checkpoint_dir,
                clean_name="candidate_interactions_checkpoint.csv",
                audit_name="candidate_interactions_audit_checkpoint.csv",
                clean_columns=CANDIDATE_INTERACTION_COLUMNS,
                audit_columns=CANDIDATE_AUDIT_COLUMNS,
            )

    interactions_df = _ensure_columns(pd.DataFrame(rows), CANDIDATE_INTERACTION_COLUMNS)
    if not interactions_df.empty:
        interactions_df = interactions_df.drop_duplicates(subset=["tweet_id"], keep="first").reset_index(drop=True)
    audit_df = _ensure_columns(pd.DataFrame(audit_rows), CANDIDATE_AUDIT_COLUMNS)

    interactions_df.to_csv(interim_dir / "candidate_interactions.csv", index=False)
    audit_df.to_csv(interim_dir / "candidate_interactions_audit.csv", index=False)
    return interactions_df, audit_df


def build_unified_political_corpus(
    replies_df: pd.DataFrame | None = None,
    quote_posts_df: pd.DataFrame | None = None,
    source_posts_df: pd.DataFrame | None = None,
    organic_posts_df: pd.DataFrame | None = None,
    candidate_interactions_df: pd.DataFrame | None = None,
    include_media_source_posts: bool = False,
    include_organic: bool = False,
    include_candidate: bool = False,
    output_path: Path | None = None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    if replies_df is not None and not replies_df.empty:
        tmp = replies_df.copy()
        tmp["source_type"] = tmp.get("source_type", "reply_to_media_post")
        tmp["source_universe"] = tmp.get("source_universe", "media_anchored")
        tmp["anchor_type"] = tmp.get("anchor_type", "media_source_post")
        tmp["anchor_post_id"] = tmp.get("anchor_post_id", tmp.get("source_post_id"))
        tmp["anchor_media_id"] = tmp.get("anchor_media_id", tmp.get("media_id"))
        tmp["anchor_media_handle"] = tmp.get("anchor_media_handle", tmp.get("media_handle"))
        tmp["tweet_id"] = tmp.get("tweet_id", tmp.get("reply_id"))
        tmp["text"] = tmp.get("text", tmp.get("reply_text"))
        tmp["created_at"] = tmp.get("created_at", tmp.get("reply_created_at"))
        tmp["author_id_hash"] = tmp.get("reply_author_id_hash")
        frames.append(tmp)

    if quote_posts_df is not None and not quote_posts_df.empty:
        tmp = quote_posts_df.copy()
        tmp["source_type"] = tmp.get("source_type", "quote_of_media_post")
        tmp["source_universe"] = tmp.get("source_universe", "media_anchored")
        tmp["anchor_type"] = tmp.get("anchor_type", "media_source_post")
        tmp["anchor_post_id"] = tmp.get("anchor_post_id", tmp.get("source_post_id"))
        tmp["anchor_media_id"] = tmp.get("anchor_media_id", tmp.get("media_id"))
        tmp["anchor_media_handle"] = tmp.get("anchor_media_handle", tmp.get("media_handle"))
        frames.append(tmp)

    if include_media_source_posts and source_posts_df is not None and not source_posts_df.empty:
        tmp = source_posts_df.copy()
        tmp["source_type"] = tmp.get("source_type", "media_source_post")
        tmp["source_universe"] = tmp.get("source_universe", "costa_rican_media")
        tmp["anchor_type"] = tmp.get("anchor_type", "media_account")
        tmp["anchor_post_id"] = tmp.get("source_post_id", tmp.get("tweet_id"))
        tmp["anchor_media_id"] = tmp.get("media_id")
        tmp["anchor_media_handle"] = tmp.get("media_handle")
        tmp["tweet_id"] = tmp.get("tweet_id", tmp.get("source_post_id"))
        tmp["text"] = tmp.get("text", tmp.get("source_post_text"))
        tmp["created_at"] = tmp.get("created_at", tmp.get("source_post_created_at"))
        tmp["public_metrics"] = tmp.get("public_metrics", tmp.get("source_post_public_metrics"))
        frames.append(tmp)

    if include_organic and organic_posts_df is not None and not organic_posts_df.empty:
        tmp = organic_posts_df.copy()
        tmp["source_type"] = tmp.get("source_type", "organic_event_post")
        tmp["source_universe"] = tmp.get("source_universe", "open_x_search")
        tmp["anchor_type"] = tmp.get("anchor_type", "none")
        tmp["anchor_post_id"] = tmp.get("anchor_post_id", pd.NA)
        tmp["anchor_media_id"] = tmp.get("anchor_media_id", pd.NA)
        tmp["anchor_media_handle"] = tmp.get("anchor_media_handle", pd.NA)
        frames.append(tmp)

    if include_candidate and candidate_interactions_df is not None and not candidate_interactions_df.empty:
        tmp = candidate_interactions_df.copy()
        tmp["source_type"] = tmp.get("source_type", "reply_or_mention_to_candidate")
        tmp["source_universe"] = tmp.get("source_universe", "open_x_search")
        tmp["anchor_type"] = tmp.get("anchor_type", "candidate_account")
        tmp["anchor_post_id"] = tmp.get("anchor_post_id", tmp.get("conversation_id"))
        tmp["anchor_media_id"] = tmp.get("anchor_media_id", pd.NA)
        tmp["anchor_media_handle"] = tmp.get("anchor_media_handle", pd.NA)
        frames.append(tmp)

    if not frames:
        corpus = _ensure_columns(pd.DataFrame(), UNIFIED_CORPUS_COLUMNS)
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            corpus.to_csv(output_path, index=False)
        return corpus

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["tweet_id"] = combined.get("tweet_id", pd.Series([None] * len(combined))).map(normalize_source_post_id)
    combined = combined[combined["tweet_id"].notna()].copy()
    combined["text"] = combined.get("text", "").fillna("").astype(str)
    combined["text_norm"] = combined["text"].map(_normalize_text_for_matching)

    for metric_col in ["reply_count", "quote_count", "retweet_count", "like_count"]:
        if metric_col not in combined.columns:
            combined[metric_col] = 0
        combined[metric_col] = _coerce_numeric(combined[metric_col], default=0)

    source_types_by_tweet = (
        combined.groupby("tweet_id", dropna=False)["source_type"]
        .agg(lambda s: "|".join(sorted({str(x) for x in s if pd.notna(x)})))
        .rename("source_type_originals")
    )

    combined = combined.sort_values(
        ["tweet_id", "reply_count", "quote_count", "retweet_count", "like_count"],
        ascending=[True, False, False, False, False],
    )
    combined = combined.drop_duplicates(subset=["tweet_id"], keep="first").reset_index(drop=True)
    combined = combined.merge(
        source_types_by_tweet,
        left_on="tweet_id",
        right_index=True,
        how="left",
        suffixes=("", "_agg"),
    )
    combined["source_type_originals"] = combined["source_type_originals"].fillna(combined.get("source_type"))
    combined["corpus_id"] = [f"hc_{idx + 1:08d}" for idx in range(len(combined))]

    corpus = _ensure_columns(combined, UNIFIED_CORPUS_COLUMNS)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        corpus.to_csv(output_path, index=False)
    return corpus


def build_media_anchored_interactions_corpus(
    replies_df: pd.DataFrame | None = None,
    quote_posts_df: pd.DataFrame | None = None,
    source_posts_df: pd.DataFrame | None = None,
    include_media_source_posts: bool = False,
    output_path: Path | None = None,
) -> pd.DataFrame:
    corpus = build_unified_political_corpus(
        replies_df=replies_df,
        quote_posts_df=quote_posts_df,
        source_posts_df=source_posts_df,
        include_media_source_posts=include_media_source_posts,
        include_organic=False,
        include_candidate=False,
        output_path=None,
    )
    media_corpus = _ensure_columns(corpus, MEDIA_ANCHORED_CORPUS_COLUMNS)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        media_corpus.to_csv(output_path, index=False)
    return media_corpus


def source_posts_diagnostics(source_posts_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if source_posts_df.empty:
        empty = pd.DataFrame()
        return {
            "posts_by_event_media": empty,
            "top_30_by_reply_count": empty,
            "top_30_by_quote_count": empty,
            "reply_count_distribution": empty,
            "quote_count_distribution": empty,
        }

    work = source_posts_df.copy()
    work["reply_count"] = _coerce_numeric(work.get("reply_count", pd.Series([0] * len(work))), default=0)
    work["quote_count"] = _coerce_numeric(work.get("quote_count", pd.Series([0] * len(work))), default=0)

    posts_by_event_media = (
        work.groupby(["event_id", "media_id"], dropna=False)
        .size()
        .reset_index(name="n_posts")
        .sort_values("n_posts", ascending=False)
    )

    top_30_by_reply_count = work.sort_values("reply_count", ascending=False).head(30)
    top_30_by_quote_count = work.sort_values("quote_count", ascending=False).head(30)

    bins = [-1, 0, 1, 5, 10, 20, 50, 100, 10_000_000]
    labels = ["0", "1", "2-5", "6-10", "11-20", "21-50", "51-100", "100+"]
    work["reply_count_bin"] = pd.cut(work["reply_count"], bins=bins, labels=labels)
    work["quote_count_bin"] = pd.cut(work["quote_count"], bins=bins, labels=labels)
    reply_count_distribution = (
        work.groupby("reply_count_bin", dropna=False)
        .size()
        .reset_index(name="n_posts")
    )
    quote_count_distribution = (
        work.groupby("quote_count_bin", dropna=False)
        .size()
        .reset_index(name="n_posts")
    )

    return {
        "posts_by_event_media": posts_by_event_media,
        "top_30_by_reply_count": top_30_by_reply_count,
        "top_30_by_quote_count": top_30_by_quote_count,
        "reply_count_distribution": reply_count_distribution,
        "quote_count_distribution": quote_count_distribution,
    }


def replies_diagnostics(
    replies_df: pd.DataFrame,
    source_posts_df: pd.DataFrame,
    stats_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    empty = pd.DataFrame()

    if replies_df.empty:
        return {
            "replies_by_post": empty,
            "posts_high_replycount_low_collected": empty,
            "endpoint_audit": stats_df,
            "avg_seconds": pd.DataFrame(),
            "errors_by_endpoint_status": pd.DataFrame(),
        }

    replies_by_post = (
        replies_df.groupby(["event_id", "media_id", "source_post_id"], dropna=False)
        .size()
        .reset_index(name="n_replies_collected")
    )

    source_ref = source_posts_df.copy()
    source_ref["source_post_id"] = source_ref["source_post_id"].astype(str)
    source_ref["reply_count"] = _coerce_numeric(source_ref.get("reply_count", pd.Series([0] * len(source_ref))), default=0)

    gaps = source_ref.merge(
        replies_by_post,
        on=["event_id", "media_id", "source_post_id"],
        how="left",
    )
    gaps["n_replies_collected"] = _coerce_numeric(gaps["n_replies_collected"], default=0)
    gaps["reply_collection_gap"] = gaps["reply_count"] - gaps["n_replies_collected"]

    posts_high_replycount_low_collected = gaps[
        (gaps["reply_count"] >= 10) & (gaps["n_replies_collected"] <= 2)
    ].sort_values("reply_count", ascending=False)

    if not stats_df.empty and "seconds" in stats_df.columns:
        avg_seconds = pd.DataFrame(
            [{"avg_seconds_per_post": _coerce_numeric(stats_df["seconds"], default=0).mean()}]
        )
    else:
        avg_seconds = pd.DataFrame()

    if not stats_df.empty:
        errors_by_endpoint_status = (
            stats_df.assign(status_code_str=stats_df["status_code"].astype(str))
            .groupby(["endpoint_used", "status", "status_code_str"], dropna=False)
            .size()
            .reset_index(name="n_posts")
            .sort_values("n_posts", ascending=False)
        )
    else:
        errors_by_endpoint_status = pd.DataFrame()

    return {
        "replies_by_post": replies_by_post,
        "posts_high_replycount_low_collected": posts_high_replycount_low_collected,
        "endpoint_audit": stats_df,
        "avg_seconds": avg_seconds,
        "errors_by_endpoint_status": errors_by_endpoint_status,
    }


def cross_layer_diagnostics(
    source_posts_df: pd.DataFrame | None = None,
    replies_df: pd.DataFrame | None = None,
    quote_posts_df: pd.DataFrame | None = None,
    organic_posts_df: pd.DataFrame | None = None,
    candidate_interactions_df: pd.DataFrame | None = None,
    source_posts_audit_df: pd.DataFrame | None = None,
    organic_audit_df: pd.DataFrame | None = None,
    candidate_audit_df: pd.DataFrame | None = None,
    replies_stats_df: pd.DataFrame | None = None,
    quote_stats_df: pd.DataFrame | None = None,
    corpus_df: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    empty = pd.DataFrame()

    queries_by_source_type_rows: list[dict[str, Any]] = []
    for source_name, audit_df in [
        ("media_source_post", source_posts_audit_df),
        ("organic_event_post", organic_audit_df),
        ("reply_or_mention_to_candidate", candidate_audit_df),
    ]:
        if audit_df is not None and not audit_df.empty:
            queries_by_source_type_rows.append(
                {"source_type": source_name, "n_queries": len(audit_df)}
            )
    queries_by_source_type = pd.DataFrame(queries_by_source_type_rows)

    posts_by_event_media = empty
    if source_posts_df is not None and not source_posts_df.empty:
        posts_by_event_media = (
            source_posts_df.groupby(["event_id", "media_id"], dropna=False)
            .size()
            .reset_index(name="n_posts")
            .sort_values("n_posts", ascending=False)
        )

    replies_by_post = empty
    if replies_df is not None and not replies_df.empty:
        replies_by_post = (
            replies_df.groupby(["event_id", "media_id", "source_post_id"], dropna=False)
            .size()
            .reset_index(name="n_replies")
            .sort_values("n_replies", ascending=False)
        )

    quotes_by_post = empty
    if quote_posts_df is not None and not quote_posts_df.empty:
        quotes_by_post = (
            quote_posts_df.groupby(["event_id", "media_id", "source_post_id"], dropna=False)
            .size()
            .reset_index(name="n_quotes")
            .sort_values("n_quotes", ascending=False)
        )

    organic_by_event = empty
    if organic_posts_df is not None and not organic_posts_df.empty:
        organic_by_event = (
            organic_posts_df.groupby(["event_id"], dropna=False)
            .size()
            .reset_index(name="n_posts")
            .sort_values("n_posts", ascending=False)
        )

    interactions_by_actor = empty
    if candidate_interactions_df is not None and not candidate_interactions_df.empty:
        interactions_by_actor = (
            candidate_interactions_df.groupby(["target_actor_id", "target_handle"], dropna=False)
            .size()
            .reset_index(name="n_posts")
            .sort_values("n_posts", ascending=False)
        )

    corpus_by_source_type = empty
    if corpus_df is not None and not corpus_df.empty:
        corpus_by_source_type = (
            corpus_df.groupby(["source_type"], dropna=False)
            .size()
            .reset_index(name="n_rows")
            .sort_values("n_rows", ascending=False)
        )

    corpus_by_anchor_media = empty
    if corpus_df is not None and not corpus_df.empty:
        anchor_col = "anchor_media_handle" if "anchor_media_handle" in corpus_df.columns else "media_handle"
        corpus_by_anchor_media = (
            corpus_df.groupby([anchor_col], dropna=False)
            .size()
            .reset_index(name="n_rows")
            .sort_values("n_rows", ascending=False)
        )

    anchored_integrity_rows: list[dict[str, Any]] = []
    if replies_df is not None and not replies_df.empty:
        anchor_series = replies_df.get("anchor_media_handle", pd.Series([None] * len(replies_df))).astype("string")
        missing = int(anchor_series.isna().sum() + (anchor_series.str.strip() == "").sum())
        anchored_integrity_rows.append(
            {"check": "replies_anchor_media_handle_not_null", "missing_rows": missing}
        )
    if quote_posts_df is not None and not quote_posts_df.empty:
        anchor_series = quote_posts_df.get("anchor_media_handle", pd.Series([None] * len(quote_posts_df))).astype("string")
        missing = int(anchor_series.isna().sum() + (anchor_series.str.strip() == "").sum())
        anchored_integrity_rows.append(
            {"check": "quotes_anchor_media_handle_not_null", "missing_rows": missing}
        )
    if corpus_df is not None and not corpus_df.empty:
        n_organic = int((corpus_df.get("source_type", pd.Series([], dtype="object")) == "organic_event_post").sum())
        anchored_integrity_rows.append(
            {"check": "organic_rows_in_main_corpus", "missing_rows": n_organic}
        )
    anchored_integrity = pd.DataFrame(anchored_integrity_rows)

    all_error_frames: list[pd.DataFrame] = []
    for audit_df in [source_posts_audit_df, organic_audit_df, candidate_audit_df, replies_stats_df, quote_stats_df]:
        if audit_df is None or audit_df.empty:
            continue
        frame = audit_df.copy()
        if "endpoint_used" not in frame.columns:
            frame["endpoint_used"] = pd.NA
        if "status" not in frame.columns:
            frame["status"] = pd.NA
        if "status_code" not in frame.columns:
            frame["status_code"] = pd.NA
        if "seconds" not in frame.columns:
            frame["seconds"] = pd.NA
        all_error_frames.append(frame[["endpoint_used", "status", "status_code", "seconds"]])

    errors_by_endpoint_status = empty
    avg_seconds = empty
    if all_error_frames:
        merged = pd.concat(all_error_frames, ignore_index=True)
        errors_by_endpoint_status = (
            merged.assign(status_code_str=merged["status_code"].astype(str))
            .groupby(["endpoint_used", "status", "status_code_str"], dropna=False)
            .size()
            .reset_index(name="n_rows")
            .sort_values("n_rows", ascending=False)
        )
        avg_seconds = pd.DataFrame(
            [{"avg_seconds": _coerce_numeric(merged["seconds"], default=0).mean()}]
        )

    return {
        "queries_by_source_type": queries_by_source_type,
        "posts_by_event_media": posts_by_event_media,
        "replies_by_post": replies_by_post,
        "quotes_by_post": quotes_by_post,
        "organic_by_event": organic_by_event,
        "interactions_by_actor": interactions_by_actor,
        "corpus_by_source_type": corpus_by_source_type,
        "corpus_by_anchor_media": corpus_by_anchor_media,
        "anchored_integrity": anchored_integrity,
        "errors_by_endpoint_status": errors_by_endpoint_status,
        "avg_seconds": avg_seconds,
    }


def build_omitted_layer_audit(
    layer_name: str,
    reason: str,
    events: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    event_rows = events or [{}]
    rows: list[dict[str, Any]] = []
    for event in event_rows:
        rows.append(
            {
                "source_type": layer_name,
                "event_id": event.get("event_id"),
                "query": pd.NA,
                "start_time": event.get("collection_window", {}).get("start"),
                "end_time": event.get("collection_window", {}).get("end"),
                "endpoint_attempted": "none",
                "endpoint_used": "none",
                "status": "omitted_by_configuration",
                "status_code": None,
                "n_rows": 0,
                "error_summary": reason,
                "seconds": 0.0,
            }
        )
    return pd.DataFrame(rows)

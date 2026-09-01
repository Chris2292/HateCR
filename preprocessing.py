from __future__ import annotations

import ast
import hashlib
import json
import re
import unicodedata
from datetime import timezone
from pathlib import Path
from typing import Any

import pandas as pd


URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
MENTION_RE = re.compile(r"(?<!\w)@[A-Za-z0-9_]{1,30}")
HASHTAG_RE = re.compile(r"#([\w_]+)")
WHITESPACE_RE = re.compile(r"\s+")
ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\ufeff\u2060]")

METRIC_KEYS = [
    "reply_count",
    "quote_count",
    "retweet_count",
    "like_count",
    "bookmark_count",
    "impression_count",
]

FORMAL_REPLY_REQUIRED_COLUMNS = [
    "source_type",
    "source_universe",
    "anchor_type",
    "anchor_post_id",
    "anchor_media_id",
    "anchor_media_handle",
    "event_id",
    "source_post_id",
    "reply_id",
    "reply_text",
    "reply_created_at",
    "reply_author_id_hash",
    "lang",
    "public_metrics",
    "conversation_id",
    "query",
    "collected_at",
]

UNCERTAIN_LANGUAGE_CODES = {"und", "qme", "qam", "qht", "zxx"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_formal_reply_batches(
    formal_collection_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load materialized formal batches without traversing retry/checkpoint folders."""
    batch_paths = sorted(
        formal_collection_dir.glob("batch_[0-9][0-9][0-9]/replies_clean.csv")
    )
    if not batch_paths:
        return pd.DataFrame(), pd.DataFrame()

    frames: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, Any]] = []
    id_dtypes = {
        "tweet_id": "string",
        "reply_id": "string",
        "source_post_id": "string",
        "anchor_post_id": "string",
        "conversation_id": "string",
        "reply_author_id_hash": "string",
    }

    for path in batch_paths:
        batch_id = path.parent.name
        frame = pd.read_csv(path, dtype=id_dtypes)
        missing = [col for col in FORMAL_REPLY_REQUIRED_COLUMNS if col not in frame.columns]
        if missing:
            raise ValueError(f"{path} no contiene columnas requeridas: {missing}")

        frame["collection_batch_id"] = batch_id
        pending_path = path.parent / "replies_pending.csv"
        pending_count = 0
        if pending_path.exists():
            pending_count = len(pd.read_csv(pending_path, dtype="string"))
        batch_status = "partial" if pending_count else "complete"
        frame["collection_batch_status"] = batch_status
        frames.append(frame)

        manifest_rows.append(
            {
                "collection_batch_id": batch_id,
                "batch_status": batch_status,
                "path": str(path),
                "sha256": _sha256_file(path),
                "rows": len(frame),
                "unique_reply_ids": frame["reply_id"].nunique(dropna=True),
                "duplicate_reply_ids_within_batch": int(frame["reply_id"].duplicated().sum()),
                "source_posts_with_replies": frame["source_post_id"].nunique(dropna=True),
                "pending_source_posts": pending_count,
                "modified_at_utc": pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC").isoformat(),
            }
        )

    combined = pd.concat(frames, ignore_index=True, sort=False)
    manifest = pd.DataFrame(manifest_rows)
    return combined, manifest


def normalize_tweet_id(value: Any) -> str | None:
    if value is None:
        return None

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
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


def parse_json_like_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value

    if value is None:
        return {}

    if isinstance(value, float) and pd.isna(value):
        return {}

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "{}", "[]"}:
        return {}

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
        return {}
    except Exception:
        pass

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, dict):
            return parsed
        return {}
    except Exception:
        return {}


def normalize_text(
    value: Any,
    lowercase: bool = True,
    replace_urls: bool = True,
    replace_mentions: bool = True,
    normalize_hashtags: bool = True,
) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKC", text)
    text = ZERO_WIDTH_RE.sub(" ", text)

    if replace_urls:
        text = URL_RE.sub(" <url> ", text)

    if replace_mentions:
        text = MENTION_RE.sub(" <user> ", text)

    if normalize_hashtags:
        text = HASHTAG_RE.sub(r" \1 ", text)

    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cc")
    text = WHITESPACE_RE.sub(" ", text).strip()

    if lowercase:
        text = text.lower()

    return text


def strip_accents(value: Any) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", text)


def _select_text_series(df: pd.DataFrame) -> pd.Series:
    for col in ["text", "reply_text", "source_post_text"]:
        if col in df.columns:
            return df[col]
    return pd.Series([""] * len(df), index=df.index, dtype="object")


def enrich_public_metrics(df: pd.DataFrame, metrics_col: str = "public_metrics") -> pd.DataFrame:
    out = df.copy()

    if metrics_col in out.columns:
        metrics_series = out[metrics_col].map(parse_json_like_dict)
    else:
        metrics_series = pd.Series([{}] * len(out), index=out.index)

    for key in METRIC_KEYS:
        extracted = pd.to_numeric(metrics_series.map(lambda d: d.get(key, None)), errors="coerce")
        if key in out.columns:
            out[key] = pd.to_numeric(out[key], errors="coerce")
            out[key] = out[key].fillna(extracted).fillna(0)
        else:
            out[key] = extracted.fillna(0)

    return out


def build_author_hash(
    author_id: Any,
    hash_salt: str | None = None,
) -> str | None:
    if author_id is None:
        return None

    author_text = str(author_id).strip()
    if not author_text or author_text.lower() in {"nan", "none", "null"}:
        return None

    salt = (hash_salt or "").strip()
    if not salt:
        return None

    return hashlib.sha256(f"{salt}:{author_text}".encode("utf-8")).hexdigest()


def clean_interactions_corpus(
    corpus_df: pd.DataFrame,
    keep_langs: list[str] | None = None,
    hash_salt: str | None = None,
    min_text_chars: int = 3,
    filter_by_lang: bool = True,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    if corpus_df.empty:
        empty_df = pd.DataFrame()
        return empty_df, {
            "summary": pd.DataFrame([{"metric": "input_rows", "value": 0}]),
            "by_source_type": pd.DataFrame(),
            "by_lang": pd.DataFrame(),
            "missing_core": pd.DataFrame(),
        }

    keep_langs_set = {str(x).strip().lower() for x in (keep_langs or ["es", "und"]) if str(x).strip()}

    work = corpus_df.copy()
    n_input = len(work)

    if "tweet_id" not in work.columns and "reply_id" in work.columns:
        work["tweet_id"] = work["reply_id"]

    work["tweet_id"] = work.get("tweet_id", pd.Series([None] * len(work), index=work.index)).map(normalize_tweet_id)

    if "source_post_id" in work.columns:
        work["source_post_id"] = work["source_post_id"].map(normalize_tweet_id)

    text_series = _select_text_series(work).fillna("").astype(str)
    work["text_original"] = text_series

    work["text_norm"] = work["text_original"].map(normalize_text)
    work["text_norm_no_accents"] = work["text_norm"].map(strip_accents)
    work["text_norm_hash"] = work["text_norm"].map(
        lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    )
    work["text_char_len"] = work["text_norm"].map(len)
    work["text_token_len"] = work["text_norm"].map(lambda x: len(x.split()) if x else 0)
    substantive_text = (
        work["text_norm"]
        .str.replace("<url>", " ", regex=False)
        .str.replace("<user>", " ", regex=False)
        .str.replace(r"[^\wáéíóúüñ]+", " ", regex=True)
        .str.strip()
    )
    work["has_substantive_text"] = substantive_text.str.len().ge(2)

    created_series = work.get("created_at", pd.Series([None] * len(work), index=work.index))
    if "reply_created_at" in work.columns:
        created_series = created_series.fillna(work["reply_created_at"])

    created_dt = pd.to_datetime(created_series, errors="coerce", utc=True)
    work["created_at_dt"] = created_dt
    work["created_at"] = created_dt.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    work.loc[work["created_at_dt"].isna(), "created_at"] = pd.NA

    work["year"] = work["created_at_dt"].dt.year
    work["month"] = work["created_at_dt"].dt.month
    work["day"] = work["created_at_dt"].dt.day
    work["hour_utc"] = work["created_at_dt"].dt.hour

    lang_series = work.get("lang", pd.Series([None] * len(work), index=work.index))
    work["lang"] = lang_series.astype("string").str.strip().str.lower().fillna("unknown")
    work["is_lang_kept"] = work["lang"].isin(keep_langs_set)
    work["lang_group"] = work["lang"].map(
        lambda value: (
            "spanish_detected"
            if value == "es"
            else "language_uncertain"
            if value in UNCERTAIN_LANGUAGE_CODES
            else "other_language_label"
        )
    )

    work = enrich_public_metrics(work, metrics_col="public_metrics")

    if "author_id_hash" not in work.columns:
        work["author_id_hash"] = pd.NA

    if "reply_author_id_hash" in work.columns:
        work["author_id_hash"] = work["author_id_hash"].fillna(work["reply_author_id_hash"])

    if "author_id" not in work.columns:
        work["author_id"] = pd.NA

    if hash_salt:
        missing_hash = work["author_id_hash"].isna()
        work.loc[missing_hash, "author_id_hash"] = work.loc[missing_hash, "author_id"].map(
            lambda x: build_author_hash(x, hash_salt=hash_salt)
        )

    if "source_type" not in work.columns:
        work["source_type"] = "unknown"
    work["source_type"] = work["source_type"].fillna("unknown").astype(str)

    if "corpus_id" not in work.columns:
        work["corpus_id"] = work["source_type"] + ":" + work["tweet_id"].fillna("")
    else:
        missing_corpus_id = work["corpus_id"].isna() | work["corpus_id"].astype(str).str.strip().eq("")
        work.loc[missing_corpus_id, "corpus_id"] = (
            work.loc[missing_corpus_id, "source_type"]
            + ":"
            + work.loc[missing_corpus_id, "tweet_id"].fillna("")
        )

    missing_tweet_ids = int(work["tweet_id"].isna().sum())
    short_text_rows = int((work["text_char_len"] < max(0, int(min_text_chars))).sum())
    language_excluded_rows = int((~work["is_lang_kept"]).sum()) if filter_by_lang else 0

    work = work[work["tweet_id"].notna()].copy()
    work = work[work["text_char_len"] >= max(0, int(min_text_chars))].copy()
    if filter_by_lang:
        work = work[work["is_lang_kept"]].copy()

    n_before_dedup = len(work)
    work = work.sort_values(["tweet_id", "created_at_dt"], ascending=[True, True])
    work = work.drop_duplicates(subset=["tweet_id"], keep="first").reset_index(drop=True)
    n_after_dedup = len(work)

    work["cleaned_at"] = pd.Timestamp.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    by_source_type = (
        work.groupby("source_type", dropna=False)
        .size()
        .reset_index(name="n_rows")
        .sort_values("n_rows", ascending=False)
    )

    by_lang = (
        work.groupby("lang", dropna=False)
        .size()
        .reset_index(name="n_rows")
        .sort_values("n_rows", ascending=False)
    )

    core_cols = ["tweet_id", "text_original", "text_norm", "created_at", "source_type", "event_id", "lang"]
    missing_core_rows: list[dict[str, Any]] = []
    for col in core_cols:
        if col not in work.columns:
            missing_core_rows.append({"column": col, "missing_count": len(work), "missing_pct": 1.0})
            continue

        miss = int(work[col].isna().sum())
        missing_core_rows.append(
            {
                "column": col,
                "missing_count": miss,
                "missing_pct": (miss / len(work)) if len(work) else 0.0,
            }
        )

    summary = pd.DataFrame(
        [
            {"metric": "input_rows", "value": n_input},
            {"metric": "missing_tweet_ids_removed", "value": missing_tweet_ids},
            {"metric": "short_text_rows_removed", "value": short_text_rows},
            {"metric": "language_rows_removed", "value": language_excluded_rows},
            {"metric": "filter_by_lang", "value": bool(filter_by_lang)},
            {"metric": "rows_after_filters_before_dedup", "value": n_before_dedup},
            {"metric": "rows_after_dedup", "value": n_after_dedup},
            {"metric": "duplicates_removed", "value": max(0, n_before_dedup - n_after_dedup)},
            {"metric": "rows_removed_total", "value": max(0, n_input - n_after_dedup)},
        ]
    )

    diagnostics = {
        "summary": summary,
        "by_source_type": by_source_type,
        "by_lang": by_lang,
        "missing_core": pd.DataFrame(missing_core_rows),
    }

    return work, diagnostics


def build_replies_analysis_view(clean_df: pd.DataFrame) -> pd.DataFrame:
    if clean_df.empty:
        return pd.DataFrame(
            columns=[
                "reply_id",
                "reply_text",
                "event_id",
                "media_account",
                "media_handle",
                "source_type",
                "created_at",
                "lang",
                "reply_count",
                "quote_count",
                "retweet_count",
                "like_count",
            ]
        )

    work = clean_df.copy()

    if "source_type" in work.columns:
        replies = work[work["source_type"] == "reply_to_media_post"].copy()
        if replies.empty:
            replies = work.copy()
    else:
        replies = work.copy()

    replies["reply_id"] = replies.get("tweet_id")
    replies["reply_text"] = replies.get("text_original", replies.get("text", ""))

    media_account = replies.get("media_handle")
    if media_account is None:
        media_account = replies.get("media_id")
    replies["media_account"] = media_account.fillna("unknown") if hasattr(media_account, "fillna") else "unknown"

    for col in ["reply_count", "quote_count", "retweet_count", "like_count"]:
        if col not in replies.columns:
            replies[col] = 0

    output_cols = [
        "reply_id",
        "reply_text",
        "text_norm",
        "text_norm_no_accents",
        "text_norm_hash",
        "event_id",
        "event_name",
        "formal_event_memberships",
        "formal_event_names",
        "formal_event_count",
        "collection_batch_id",
        "collection_batch_status",
        "source_post_id",
        "anchor_media_id",
        "anchor_media_handle",
        "media_account",
        "media_handle",
        "source_type",
        "source_universe",
        "anchor_type",
        "created_at",
        "lang",
        "lang_group",
        "is_lang_kept",
        "reply_author_id_hash",
        "author_id_hash",
        "conversation_id",
        "reply_count",
        "quote_count",
        "retweet_count",
        "like_count",
        "bookmark_count",
        "impression_count",
        "text_char_len",
        "text_token_len",
        "has_substantive_text",
        "query",
        "collected_at",
        "cleaned_at",
    ]

    for col in output_cols:
        if col not in replies.columns:
            replies[col] = pd.NA

    replies = replies[output_cols].drop_duplicates(subset=["reply_id"], keep="first").reset_index(drop=True)
    return replies


def load_interim_layers(interim_dir: Path) -> dict[str, pd.DataFrame]:
    mapping = {
        "replies": interim_dir / "replies_clean.csv",
        "quotes": interim_dir / "quote_posts_clean.csv",
        "organic": interim_dir / "organic_event_posts.csv",
        "candidate": interim_dir / "candidate_interactions.csv",
    }

    layers: dict[str, pd.DataFrame] = {}
    for key, path in mapping.items():
        if path.exists():
            layers[key] = pd.read_csv(path)
        else:
            layers[key] = pd.DataFrame()

    return layers


def combine_layers_fallback(layers: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    if not layers:
        return pd.DataFrame()

    replies = layers.get("replies", pd.DataFrame())
    if not replies.empty:
        tmp = replies.copy()
        tmp["source_type"] = tmp.get("source_type", "reply_to_media_post")
        tmp["tweet_id"] = tmp.get("tweet_id", tmp.get("reply_id"))
        tmp["text"] = tmp.get("text", tmp.get("reply_text"))
        frames.append(tmp)

    quotes = layers.get("quotes", pd.DataFrame())
    if not quotes.empty:
        tmp = quotes.copy()
        tmp["source_type"] = tmp.get("source_type", "quote_of_media_post")
        frames.append(tmp)

    organic = layers.get("organic", pd.DataFrame())
    if not organic.empty:
        tmp = organic.copy()
        tmp["source_type"] = tmp.get("source_type", "organic_event_post")
        frames.append(tmp)

    candidate = layers.get("candidate", pd.DataFrame())
    if not candidate.empty:
        tmp = candidate.copy()
        tmp["source_type"] = tmp.get("source_type", "reply_or_mention_to_candidate")
        frames.append(tmp)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True, sort=False)

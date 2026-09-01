from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.preprocessing import normalize_text, strip_accents


TOKEN_RE = re.compile(r"[a-zA-Z]+")
TRUTHY_VALUES = {"1", "true", "t", "yes", "y", "si", "s"}

DEFAULT_SPANISH_STOPWORDS = {
    "a",
    "al",
    "algo",
    "ante",
    "como",
    "con",
    "contra",
    "cual",
    "cuando",
    "de",
    "del",
    "desde",
    "donde",
    "dos",
    "el",
    "ella",
    "ellas",
    "ellos",
    "en",
    "entre",
    "era",
    "es",
    "esa",
    "ese",
    "eso",
    "esta",
    "este",
    "esto",
    "fue",
    "ha",
    "han",
    "hasta",
    "hay",
    "la",
    "las",
    "le",
    "les",
    "lo",
    "los",
    "mas",
    "me",
    "mi",
    "muy",
    "ni",
    "no",
    "nos",
    "o",
    "para",
    "pero",
    "por",
    "porque",
    "que",
    "se",
    "ser",
    "si",
    "sin",
    "son",
    "su",
    "sus",
    "te",
    "tiene",
    "tu",
    "un",
    "una",
    "uno",
    "user",
    "url",
    "y",
    "ya",
    "yo",
}


def atomic_to_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    df.to_csv(temporary, index=False)
    temporary.replace(path)
    return path


def safe_read_csv(
    path: Path,
    name: str | None = None,
    dtype: Any = None,
) -> pd.DataFrame:
    if not path.exists():
        print(f"[WARN] Missing {name or path.name}: {path}")
        return pd.DataFrame()
    errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            frame = pd.read_csv(path, dtype=dtype, encoding=encoding)
            print(
                f"[OK] {name or path.name}: {len(frame):,} rows "
                f"(encoding={encoding})"
            )
            return frame
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
        except Exception as exc:
            print(f"[WARN] Could not load {name or path.name}: {exc}")
            return pd.DataFrame()
    print(f"[WARN] Could not decode {name or path.name}: {' | '.join(errors)}")
    return pd.DataFrame()


def load_formal_reply_stats(formal_collection_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    paths = sorted(
        formal_collection_dir.glob("batch_[0-9][0-9][0-9]/replies_stats.csv")
    )
    for path in paths:
        frame = pd.read_csv(
            path,
            dtype={"source_post_id": "string", "status_code": "string"},
        )
        frame["collection_batch_id"] = path.parent.name
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def summarize_by_group(
    df: pd.DataFrame,
    group_columns: str | list[str],
    tweet_col: str = "tweet_id",
    author_col: str = "reply_author_id_hash",
) -> pd.DataFrame:
    groups = [group_columns] if isinstance(group_columns, str) else list(group_columns)
    if df.empty or any(column not in df.columns for column in groups):
        return pd.DataFrame(columns=groups + ["n_rows", "n_unique_tweets", "n_unique_authors"])

    aggregations: dict[str, tuple[str, str]] = {"n_rows": (groups[0], "size")}
    if tweet_col in df.columns:
        aggregations["n_unique_tweets"] = (tweet_col, "nunique")
    if author_col in df.columns:
        aggregations["n_unique_authors"] = (author_col, "nunique")

    result = (
        df.groupby(groups, dropna=False)
        .agg(**aggregations)
        .reset_index()
        .sort_values("n_rows", ascending=False)
        .reset_index(drop=True)
    )
    return result


def normalize_for_matching(value: Any) -> str:
    text = strip_accents(normalize_text(value))
    return " ".join(TOKEN_RE.findall(text.lower()))


def tokenize_for_eda(
    value: Any,
    stopwords: set[str] | None = None,
    min_length: int = 3,
) -> list[str]:
    stopword_set = DEFAULT_SPANISH_STOPWORDS if stopwords is None else stopwords
    tokens = TOKEN_RE.findall(normalize_for_matching(value))
    return [
        token
        for token in tokens
        if len(token) >= max(1, int(min_length)) and token not in stopword_set
    ]


def count_ngrams(
    texts: Iterable[Any],
    n: int,
    top_n: int = 100,
    stopwords: set[str] | None = None,
    min_token_length: int = 3,
) -> pd.DataFrame:
    ngram_size = max(1, int(n))
    counter: Counter[str] = Counter()
    for text in texts:
        tokens = tokenize_for_eda(
            text,
            stopwords=stopwords,
            min_length=min_token_length,
        )
        if len(tokens) < ngram_size:
            continue
        counter.update(
            " ".join(tokens[index : index + ngram_size])
            for index in range(len(tokens) - ngram_size + 1)
        )
    column = "term" if ngram_size == 1 else f"{ngram_size}gram"
    return pd.DataFrame(counter.most_common(max(1, int(top_n))), columns=[column, "count"])


def prepare_lexicon_index(
    lexicon_df: pd.DataFrame,
) -> tuple[dict[int, dict[str, dict[str, Any]]], pd.DataFrame]:
    required = {"term", "category", "source", "include_in_classification"}
    if lexicon_df.empty or not required.issubset(lexicon_df.columns):
        return {}, pd.DataFrame()

    work = lexicon_df.copy()
    include = (
        work["include_in_classification"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(TRUTHY_VALUES)
    )
    work = work[include].copy()
    work["match_term"] = work["term"].map(normalize_for_matching)
    work = work[work["match_term"].ne("")].copy()
    work["token_count"] = work["match_term"].str.split().str.len()

    index: dict[int, dict[str, dict[str, Any]]] = {}
    metadata_rows: list[dict[str, Any]] = []
    for match_term, part in work.groupby("match_term", sort=False):
        token_count = int(part["token_count"].iloc[0])
        categories = sorted(
            set(part["category"].dropna().astype(str).str.strip()) - {""}
        )
        sources = sorted(
            set(part["source"].dropna().astype(str).str.strip()) - {""}
        )
        metadata = {
            "term": match_term,
            "token_count": token_count,
            "categories": categories,
            "sources": sources,
        }
        index.setdefault(token_count, {})[match_term] = metadata
        metadata_rows.append(
            {
                "match_term": match_term,
                "token_count": token_count,
                "categories": "|".join(categories),
                "sources": "|".join(sources),
                "lexicon_rows": len(part),
            }
        )

    return index, pd.DataFrame(metadata_rows)


def match_text_to_lexicon(
    value: Any,
    lexicon_index: dict[int, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    tokens = TOKEN_RE.findall(normalize_for_matching(value))
    found_terms: set[str] = set()
    found_categorized_terms: set[str] = set()
    found_uncategorized_terms: set[str] = set()
    found_categories: set[str] = set()
    found_sources: set[str] = set()

    for ngram_size, term_map in lexicon_index.items():
        if len(tokens) < ngram_size:
            continue
        for index in range(len(tokens) - ngram_size + 1):
            phrase = " ".join(tokens[index : index + ngram_size])
            metadata = term_map.get(phrase)
            if metadata is None:
                continue
            found_terms.add(phrase)
            if any(category != "uncategorized" for category in metadata["categories"]):
                found_categorized_terms.add(phrase)
            else:
                found_uncategorized_terms.add(phrase)
            found_categories.update(metadata["categories"])
            found_sources.update(metadata["sources"])

    terms = sorted(found_terms)
    categories = sorted(found_categories)
    sources = sorted(found_sources)
    return {
        "lexicon_hit_count": len(terms),
        "categorized_lexicon_hit_count": len(found_categorized_terms),
        "uncategorized_lexicon_hit_count": len(found_uncategorized_terms),
        "lexicon_terms_found": "|".join(terms),
        "lexicon_categories_found": "|".join(categories),
        "lexicon_sources_found": "|".join(sources),
        "has_lexicon_match": bool(terms),
        "has_categorized_lexicon_match": bool(found_categorized_terms),
    }


def apply_lexicon_matches(
    df: pd.DataFrame,
    lexicon_df: pd.DataFrame,
    text_col: str = "text_norm_no_accents",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = df.copy()
    index, lexicon_metadata = prepare_lexicon_index(lexicon_df)
    empty_defaults = {
        "lexicon_hit_count": 0,
        "categorized_lexicon_hit_count": 0,
        "uncategorized_lexicon_hit_count": 0,
        "lexicon_terms_found": "",
        "lexicon_categories_found": "",
        "lexicon_sources_found": "",
        "has_lexicon_match": False,
        "has_categorized_lexicon_match": False,
    }
    if not index or text_col not in output.columns:
        for column, default in empty_defaults.items():
            output[column] = default
        return output, lexicon_metadata

    matches = output[text_col].fillna("").map(
        lambda value: match_text_to_lexicon(value, index)
    )
    match_frame = pd.DataFrame(matches.tolist(), index=output.index)
    for column, default in empty_defaults.items():
        output[column] = match_frame.get(column, default)
    return output, lexicon_metadata

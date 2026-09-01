from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Iterable, Optional, Sequence, Set

import pandas as pd


DEFAULT_SPANISH_STOPWORDS = {
    "al",
    "algo",
    "ante",
    "asi",
    "como",
    "con",
    "cual",
    "cuando",
    "de",
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
    "eso",
    "esta",
    "este",
    "esto",
    "fue",
    "ha",
    "han",
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
    "si",
    "sin",
    "son",
    "su",
    "sus",
    "te",
    "tiene",
    "un",
    "una",
    "uno",
    "ya",
    "yo",
}


def normalize_lexical_text(text: object) -> str:
    if text is None:
        return ""
    try:
        if pd.isna(text):
            return ""
    except (TypeError, ValueError):
        pass
    value = str(text or "").lower()
    value = re.sub(r"https?://\S+|www\.\S+", " ", value)
    value = re.sub(r"@\w+", " ", value)
    value = re.sub(r"#(\w+)", r"\1", value)
    value = unicodedata.normalize("NFD", value)
    value = "".join(
        character
        for character in value
        if unicodedata.category(character) != "Mn"
    )
    value = re.sub(r"[^a-z0-9ñ\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def tokenize_lexical_text(
    text: object,
    stopwords: Optional[Iterable[str]] = None,
    min_length: int = 3,
) -> list:
    stopword_set = (
        set(DEFAULT_SPANISH_STOPWORDS)
        if stopwords is None
        else set(stopwords)
    )
    return [
        token
        for token in normalize_lexical_text(text).split()
        if len(token) >= min_length
        and not token.isdigit()
        and token not in stopword_set
    ]


def parse_pipe_values(value: object) -> list:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return [
        normalized
        for normalized in (
            normalize_lexical_text(part)
            for part in str(value).split("|")
        )
        if normalized
    ]


def build_categorized_lexicon_index(
    lexicon: pd.DataFrame,
    excluded_categories: Optional[Set[str]] = None,
) -> pd.DataFrame:
    excluded = excluded_categories or {"uncategorized", "unknown", ""}
    required = {"term", "category", "source"}
    missing = required.difference(lexicon.columns)
    if missing:
        raise KeyError(f"Missing lexicon columns: {sorted(missing)}")

    working = lexicon.copy()
    working["category"] = working["category"].fillna("").astype(str).str.strip()
    working["source"] = working["source"].fillna("").astype(str).str.strip()
    working = working[~working["category"].str.lower().isin(excluded)].copy()

    term_frames = []
    for column in ["term", "lemma"]:
        if column not in working.columns:
            continue
        part = working[[column, "category", "source"]].copy()
        part["term_norm"] = part[column].map(normalize_lexical_text)
        term_frames.append(part[["term_norm", "category", "source"]])

    if not term_frames:
        return pd.DataFrame(
            columns=["term_norm", "categories", "sources"]
        )
    terms = pd.concat(term_frames, ignore_index=True)
    terms = terms[terms["term_norm"].ne("")].drop_duplicates()
    return (
        terms.groupby("term_norm", as_index=False)
        .agg(
            categories=(
                "category",
                lambda values: "|".join(sorted(set(values))),
            ),
            sources=(
                "source",
                lambda values: "|".join(sorted(set(values))),
            ),
        )
        .sort_values("term_norm")
        .reset_index(drop=True)
    )


def explode_categorized_matches(
    corpus: pd.DataFrame,
    term_index: pd.DataFrame,
    match_column: str = "lexicon_terms_found",
    id_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    if match_column not in corpus.columns:
        raise KeyError(f"Missing match column: {match_column}")
    term_set = set(term_index["term_norm"].dropna().astype(str))
    selected_ids = [
        column for column in (id_columns or []) if column in corpus.columns
    ]
    rows = []
    for _, row in corpus.iterrows():
        matched_terms = {
            term
            for term in parse_pipe_values(row[match_column])
            if term in term_set
        }
        for term in matched_terms:
            output = {column: row[column] for column in selected_ids}
            output["term_norm"] = term
            rows.append(output)

    columns = selected_ids + ["term_norm"]
    exploded = pd.DataFrame(rows, columns=columns)
    if exploded.empty:
        return exploded.merge(term_index, on="term_norm", how="left")
    return exploded.merge(term_index, on="term_norm", how="left")


def build_manual_term_validation(
    exploded_manual: pd.DataFrame,
    document_column: str = "tweet_id",
    hostility_column: str = "manual_hostility",
    hate_column: str = "manual_hate_speech",
    total_hostility_rate: Optional[float] = None,
) -> pd.DataFrame:
    required = {
        document_column,
        "term_norm",
        hostility_column,
        hate_column,
    }
    missing = required.difference(exploded_manual.columns)
    if missing:
        raise KeyError(f"Missing manual validation columns: {sorted(missing)}")

    working = exploded_manual.drop_duplicates(
        [document_column, "term_norm"]
    ).copy()
    working[hostility_column] = pd.to_numeric(
        working[hostility_column], errors="coerce"
    )
    working[hate_column] = pd.to_numeric(
        working[hate_column], errors="coerce"
    )
    validation = (
        working.groupby("term_norm", as_index=False)
        .agg(
            document_count=(document_column, "nunique"),
            hostile_document_count=(hostility_column, "sum"),
            hate_document_count=(hate_column, "sum"),
            categories=("categories", "first"),
            sources=("sources", "first"),
        )
    )
    validation["hostility_precision"] = (
        validation["hostile_document_count"]
        / validation["document_count"]
    )
    validation["hate_precision"] = (
        validation["hate_document_count"]
        / validation["document_count"]
    )
    base_rate = total_hostility_rate
    if base_rate is None:
        base_rate = float(working[hostility_column].mean())
    validation["hostility_lift"] = (
        validation["hostility_precision"] / base_rate
        if base_rate
        else 0.0
    )
    return validation.sort_values(
        ["hostility_precision", "document_count"],
        ascending=[False, False],
    ).reset_index(drop=True)


def ngram_document_frequency(
    documents: pd.DataFrame,
    text_column: str,
    document_column: str,
    n: int,
    required_tokens: Optional[Set[str]] = None,
    stopwords: Optional[Iterable[str]] = None,
    min_count: int = 1,
) -> pd.DataFrame:
    if n < 1:
        raise ValueError("n must be at least 1")
    required = set(required_tokens or [])
    counts: Counter = Counter()
    for _, row in documents[[document_column, text_column]].iterrows():
        tokens = tokenize_lexical_text(row[text_column], stopwords=stopwords)
        document_ngrams = set()
        for index in range(len(tokens) - n + 1):
            parts = tuple(tokens[index : index + n])
            if required and not required.intersection(parts):
                continue
            document_ngrams.add(" ".join(parts))
        counts.update(document_ngrams)

    rows = [
        {"ngram": value, "document_count": count, "n": n}
        for value, count in counts.items()
        if count >= min_count
    ]
    return pd.DataFrame(
        rows, columns=["ngram", "document_count", "n"]
    ).sort_values(
        ["document_count", "ngram"], ascending=[False, True]
    ).reset_index(drop=True)


def contrast_ngrams_by_binary_label(
    documents: pd.DataFrame,
    text_column: str,
    document_column: str,
    label_column: str,
    n: int,
    stopwords: Optional[Iterable[str]] = None,
    min_total_documents: int = 3,
    smoothing: float = 0.5,
) -> pd.DataFrame:
    required = {text_column, document_column, label_column}
    missing = required.difference(documents.columns)
    if missing:
        raise KeyError(f"Missing ngram contrast columns: {sorted(missing)}")

    working = documents.drop_duplicates(document_column).copy()
    working[label_column] = pd.to_numeric(
        working[label_column], errors="coerce"
    )
    working = working[working[label_column].isin([0, 1])]
    positive_total = int((working[label_column] == 1).sum())
    negative_total = int((working[label_column] == 0).sum())

    positive_counts: Counter = Counter()
    negative_counts: Counter = Counter()
    for _, row in working.iterrows():
        tokens = tokenize_lexical_text(row[text_column], stopwords=stopwords)
        document_ngrams = {
            " ".join(tokens[index : index + n])
            for index in range(len(tokens) - n + 1)
        }
        if int(row[label_column]) == 1:
            positive_counts.update(document_ngrams)
        else:
            negative_counts.update(document_ngrams)

    rows = []
    for ngram in set(positive_counts).union(negative_counts):
        positive = positive_counts[ngram]
        negative = negative_counts[ngram]
        total = positive + negative
        if total < min_total_documents:
            continue
        positive_rate = positive / positive_total if positive_total else 0.0
        negative_rate = negative / negative_total if negative_total else 0.0
        log_odds = math.log(
            (positive + smoothing)
            / (positive_total - positive + smoothing)
        ) - math.log(
            (negative + smoothing)
            / (negative_total - negative + smoothing)
        )
        rows.append(
            {
                "ngram": ngram,
                "n": n,
                "hostile_document_count": positive,
                "non_hostile_document_count": negative,
                "total_document_count": total,
                "hostile_document_rate": positive_rate,
                "non_hostile_document_rate": negative_rate,
                "hostility_rate_difference": positive_rate - negative_rate,
                "log_odds_hostility": log_odds,
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "ngram",
            "n",
            "hostile_document_count",
            "non_hostile_document_count",
            "total_document_count",
            "hostile_document_rate",
            "non_hostile_document_rate",
            "hostility_rate_difference",
            "log_odds_hostility",
        ],
    ).sort_values(
        ["log_odds_hostility", "hostile_document_count"],
        ascending=[False, False],
    ).reset_index(drop=True)

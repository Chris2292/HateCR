from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping
import re
import unicodedata

import pandas as pd

FINAL_LEXICON_COLUMNS = [
    "term",
    "lemma",
    "category",
    "target_type",
    "severity",
    "context_dependency",
    "source",
    "notes",
    "include_in_queries",
    "include_in_classification",
]

DEFAULT_ALIASES = {
    "term": ["term", "token", "lexeme", "word", "palabra", "expresion", "expression"],
    "lemma": ["lemma", "lema"],
    "category": ["category", "categoria", "label", "class", "group"],
    "target_type": ["target_type", "target", "objetivo", "target_group"],
    "severity": ["severity", "weight", "score", "intensity", "severidad"],
    "context_dependency": [
        "context_dependency",
        "context",
        "depends_on_context",
        "dependencia_contexto",
    ],
    "source": ["source", "fuente", "origin"],
    "notes": ["notes", "note", "observaciones", "comments"],
    "include_in_queries": ["include_in_queries", "include_query", "query_flag"],
    "include_in_classification": [
        "include_in_classification",
        "include_classification",
        "classification_flag",
    ],
}


def find_project_root(start: Path | str | None = None) -> Path:
    """Find the project root based on expected folder markers."""
    start_path = Path(start).resolve() if start else Path.cwd().resolve()
    for candidate in [start_path, *start_path.parents]:
        if (candidate / "config").exists() and (candidate / "src").exists():
            return candidate
    return start_path


def normalize_text(value: Any) -> str:
    """Normalize text for lexicon processing."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""

    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_column_name(name: Any) -> str:
    """Normalize column names to a canonical comparison form."""
    text = normalize_text(name)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def list_raw_lexicon_files(raw_dir: Path | str) -> list[Path]:
    """List supported raw lexicon files in a directory."""
    raw_path = Path(raw_dir)
    supported_ext = {".csv", ".tsv", ".txt", ".xlsx"}
    files = [p for p in raw_path.iterdir() if p.is_file() and p.suffix.lower() in supported_ext]
    return sorted(files)


def _read_lexicon_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")

    if suffix == ".txt":
        df = pd.read_csv(path, sep="\t")
        if df.shape[1] == 1:
            return pd.read_csv(path)
        return df

    if suffix == ".xlsx":
        return pd.read_excel(path)

    raise ValueError(f"Unsupported lexicon file format: {path}")


def load_raw_lexicon_files(raw_dir: Path | str) -> dict[str, pd.DataFrame]:
    """Load raw lexicon files and return them keyed by file stem."""
    raw_path = Path(raw_dir)
    files = list_raw_lexicon_files(raw_path)

    loaded: dict[str, pd.DataFrame] = {}
    for file_path in files:
        source_name = file_path.stem
        loaded[source_name] = _read_lexicon_file(file_path)

    return loaded


def infer_column(columns: Iterable[str], aliases: Iterable[str]) -> str | None:
    """Infer a column by trying normalized aliases."""
    normalized_to_original = {normalize_column_name(c): c for c in columns}
    for alias in aliases:
        candidate = normalize_column_name(alias)
        if candidate in normalized_to_original:
            return normalized_to_original[candidate]
    return None


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default

    if isinstance(value, bool):
        return value

    text = normalize_text(value)
    if text in {"1", "true", "t", "yes", "y", "si", "s"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    return default


def standardize_raw_lexicon(
    df: pd.DataFrame,
    source: str,
    schema: Mapping[str, str] | None = None,
    default_category: str = "uncategorized",
    default_target_type: str = "unknown",
    default_severity: float = 1.0,
    default_context_dependency: str = "medium",
    include_in_queries_default: bool = False,
    include_in_classification_default: bool = True,
) -> pd.DataFrame:
    """Convert a raw lexicon dataframe into the HateCR processing schema."""
    schema = dict(schema or {})
    local_df = df.copy()

    original_columns = list(local_df.columns)
    normalized_columns = {normalize_column_name(c): c for c in original_columns}

    def resolve_field(field: str, required: bool = False) -> str | None:
        if field in schema:
            configured = schema[field]
            if configured in local_df.columns:
                return configured
            configured_norm = normalize_column_name(configured)
            if configured_norm in normalized_columns:
                return normalized_columns[configured_norm]
            if required:
                raise ValueError(
                    f"Schema for source '{source}' references missing column '{configured}'"
                )
            return None

        inferred = infer_column(local_df.columns, DEFAULT_ALIASES.get(field, []))
        if required and inferred is None:
            raise ValueError(
                f"Could not infer required field '{field}' for source '{source}'. "
                f"Available columns: {original_columns}"
            )
        return inferred

    term_col = resolve_field("term", required=True)
    lemma_col = resolve_field("lemma", required=False)
    category_col = resolve_field("category", required=False)
    target_type_col = resolve_field("target_type", required=False)
    severity_col = resolve_field("severity", required=False)
    context_col = resolve_field("context_dependency", required=False)
    source_col = resolve_field("source", required=False)
    notes_col = resolve_field("notes", required=False)
    include_queries_col = resolve_field("include_in_queries", required=False)
    include_class_col = resolve_field("include_in_classification", required=False)

    out = pd.DataFrame()
    out["term"] = local_df[term_col].map(normalize_text)

    if lemma_col:
        out["lemma"] = local_df[lemma_col].map(normalize_text)
    else:
        out["lemma"] = ""

    if category_col:
        out["raw_category"] = local_df[category_col].map(normalize_text)
        out["category"] = out["raw_category"].replace("", pd.NA).fillna(default_category)
    else:
        out["raw_category"] = ""
        out["category"] = default_category

    if target_type_col:
        out["target_type"] = local_df[target_type_col].map(normalize_text).replace("", pd.NA)
        out["target_type"] = out["target_type"].fillna(default_target_type)
    else:
        out["target_type"] = default_target_type

    if severity_col:
        out["severity"] = pd.to_numeric(local_df[severity_col], errors="coerce").fillna(default_severity)
    else:
        out["severity"] = float(default_severity)

    if context_col:
        out["context_dependency"] = local_df[context_col].map(normalize_text).replace("", pd.NA)
        out["context_dependency"] = out["context_dependency"].fillna(default_context_dependency)
    else:
        out["context_dependency"] = default_context_dependency

    if source_col:
        out["source"] = local_df[source_col].map(normalize_text).replace("", pd.NA).fillna(source)
    else:
        out["source"] = source

    if notes_col:
        out["notes"] = local_df[notes_col].fillna("").astype(str).str.strip()
    else:
        out["notes"] = ""

    if include_queries_col:
        out["include_in_queries"] = local_df[include_queries_col].map(
            lambda x: _coerce_bool(x, include_in_queries_default)
        )
    else:
        out["include_in_queries"] = include_in_queries_default

    if include_class_col:
        out["include_in_classification"] = local_df[include_class_col].map(
            lambda x: _coerce_bool(x, include_in_classification_default)
        )
    else:
        out["include_in_classification"] = include_in_classification_default

    out["term"] = out["term"].map(normalize_text)
    out["lemma"] = out["lemma"].map(normalize_text)

    out = out[out["term"] != ""].reset_index(drop=True)
    return out


def load_spacy_spanish_model(model_candidates: Iterable[str] = ("es_core_news_md", "es_core_news_sm")):
    """Load a Spanish spaCy model without downloading anything automatically."""
    import spacy

    errors = []
    for model_name in model_candidates:
        try:
            return spacy.load(model_name)
        except Exception as exc:  # pragma: no cover - depends on local env
            errors.append(f"{model_name}: {exc}")

    raise OSError(
        "Could not load a Spanish spaCy model. "
        "Install one manually, for example: `python -m spacy download es_core_news_md`. "
        f"Attempted models: {list(model_candidates)}. Errors: {errors}"
    )


def lemmatize_terms(
    df: pd.DataFrame,
    term_col: str = "term",
    lemma_col: str = "lemma",
    nlp: Any | None = None,
) -> pd.DataFrame:
    """Lemmatize terms in Spanish using spaCy."""
    if nlp is None:
        nlp = load_spacy_spanish_model()

    local_df = df.copy()
    local_df[term_col] = local_df[term_col].map(normalize_text)

    unique_terms = local_df[term_col].dropna().astype(str).unique().tolist()
    lemma_map: dict[str, str] = {}

    for doc in nlp.pipe(unique_terms, batch_size=256):
        lemmas = []
        for token in doc:
            if token.is_space:
                continue
            lemma = normalize_text(token.lemma_)
            lemmas.append(lemma if lemma else normalize_text(token.text))

        normalized_doc = normalize_text(doc.text)
        lemma_map[normalized_doc] = normalize_text(" ".join(lemmas))

    local_df[lemma_col] = local_df[term_col].map(lambda text: lemma_map.get(normalize_text(text), text))
    local_df[lemma_col] = local_df[lemma_col].replace("", pd.NA).fillna(local_df[term_col])

    return local_df


def union_lexicon_sources(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Union a list of processed lexicon dataframes."""
    frame_list = [f.copy() for f in frames if f is not None and not f.empty]
    if not frame_list:
        return pd.DataFrame(columns=FINAL_LEXICON_COLUMNS + ["raw_category"])
    return pd.concat(frame_list, ignore_index=True)


def deduplicate_lexicon(
    df: pd.DataFrame,
    subset: list[str] | None = None,
    keep: str = "first",
) -> pd.DataFrame:
    """Drop duplicates from a lexicon dataframe."""
    if subset is None:
        subset = ["term", "lemma", "category", "source"]
    return df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)


def assign_categories(
    df: pd.DataFrame,
    category_map: Mapping[str, str] | None = None,
    default_category: str = "uncategorized",
) -> pd.DataFrame:
    """Assign final categories using existing category/raw_category values and optional mapping."""
    local_df = df.copy()

    if "category" not in local_df.columns:
        local_df["category"] = default_category

    local_df["category"] = local_df["category"].map(normalize_text)

    if category_map:
        normalized_map = {normalize_text(k): normalize_text(v) for k, v in category_map.items()}

        if "raw_category" in local_df.columns:
            raw_series = local_df["raw_category"].map(normalize_text)
            mapped = raw_series.map(normalized_map)
            local_df["category"] = mapped.fillna(local_df["category"])

        local_df["category"] = local_df["category"].map(
            lambda x: normalized_map.get(normalize_text(x), x)
        )

    local_df["category"] = local_df["category"].replace("", pd.NA).fillna(default_category)
    return local_df


def ensure_final_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the dataframe has all required final lexicon columns."""
    local_df = df.copy()

    defaults: dict[str, Any] = {
        "term": "",
        "lemma": "",
        "category": "uncategorized",
        "target_type": "unknown",
        "severity": 1.0,
        "context_dependency": "medium",
        "source": "unknown",
        "notes": "",
        "include_in_queries": False,
        "include_in_classification": True,
    }

    for col in FINAL_LEXICON_COLUMNS:
        if col not in local_df.columns:
            local_df[col] = defaults[col]

    local_df["term"] = local_df["term"].map(normalize_text)
    local_df["lemma"] = local_df["lemma"].map(normalize_text)
    local_df["lemma"] = local_df["lemma"].replace("", pd.NA).fillna(local_df["term"])
    local_df["category"] = local_df["category"].map(normalize_text).replace("", "uncategorized")
    local_df["target_type"] = local_df["target_type"].map(normalize_text).replace("", "unknown")
    local_df["context_dependency"] = (
        local_df["context_dependency"].map(normalize_text).replace("", "medium")
    )
    local_df["source"] = local_df["source"].map(normalize_text).replace("", "unknown")
    local_df["severity"] = pd.to_numeric(local_df["severity"], errors="coerce").fillna(1.0)

    local_df["include_in_queries"] = local_df["include_in_queries"].map(
        lambda x: _coerce_bool(x, False)
    )
    local_df["include_in_classification"] = local_df["include_in_classification"].map(
        lambda x: _coerce_bool(x, True)
    )

    local_df = local_df[local_df["term"] != ""].copy()
    return local_df


def export_processed_lexicon(df: pd.DataFrame, output_path: Path | str) -> Path:
    """Export processed lexicon with the final required schema."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    final_df = ensure_final_schema(df)
    final_df = final_df[FINAL_LEXICON_COLUMNS].copy()
    final_df = final_df.sort_values(["category", "term", "lemma", "source"]).reset_index(drop=True)
    final_df.to_csv(output, index=False)

    return output

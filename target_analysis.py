"""Probable-target analysis for the formal HateCR media-anchored corpus."""

from __future__ import annotations

import re
import textwrap
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


ASSIGNMENT_ORDER = ["high", "medium", "low", "ambiguous", "unassigned"]
ASSIGNMENT_LABELS = {
    "high": "Vinculo directo\nofensa-entidad",
    "medium": "Mencion directa\nsin vinculo",
    "low": "Contexto unico\ndel post madre",
    "ambiguous": "Ambiguo",
    "unassigned": "Sin blanco\ninferible",
}


def normalize_target_text(value: object) -> str:
    """Normalize Spanish text while keeping words from hashtags and mentions."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[#@]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _as_list(value: object) -> list:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    return [item.strip() for item in text.split("|") if item.strip()]


def _pipe(values: Iterable[object]) -> str:
    seen = set()
    output = []
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        key = normalize_target_text(text)
        if key in seen:
            continue
        seen.add(key)
        output.append(text)
    return "|".join(output)


def _validate_columns(frame: pd.DataFrame, required: Sequence[str], name: str) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise KeyError("Faltan columnas en {}: {}".format(name, missing))


def load_target_catalog(
    target_config_path: Union[str, Path],
    media_accounts_path: Optional[Union[str, Path]] = None,
    political_accounts_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Load explicit targets and optionally append verified account handles."""
    config_path = Path(target_config_path)
    if not config_path.exists():
        raise FileNotFoundError("No existe el catalogo de blancos: {}".format(config_path))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    settings = config.get("settings") or {}

    rows = []
    for target in config.get("targets", []):
        target_id = str(target.get("target_id") or "").strip()
        label = str(target.get("target_label") or "").strip()
        target_type = str(target.get("target_type") or "unknown").strip()
        if not target_id or not label:
            raise ValueError("Cada target requiere target_id y target_label")
        aliases = _as_list(target.get("aliases"))
        mention_count_aliases = _as_list(target.get("mention_count_aliases"))
        context_aliases = _as_list(target.get("context_aliases"))
        handles = [handle.lstrip("@") for handle in _as_list(target.get("handles"))]
        aliases.extend(handles)
        if not aliases:
            raise ValueError("El target {} no tiene aliases".format(target_id))
        rows.append(
            {
                "target_id": target_id,
                "target_label": label,
                "target_type": target_type,
                "aliases": _pipe(aliases),
                "mention_count_aliases": _pipe(aliases + mention_count_aliases),
                "context_aliases": _pipe(aliases + context_aliases),
                "handles": _pipe(handles),
                "potential_identity_basis": str(
                    target.get("potential_identity_basis") or "none"
                ),
                "allow_unlinked_direct_assignment": bool(
                    target.get("allow_unlinked_direct_assignment", True)
                ),
                "allow_linked_direct_assignment": bool(
                    target.get("allow_linked_direct_assignment", True)
                ),
                "catalog_source": "target_entities",
                "active": bool(target.get("active", True)),
            }
        )

    catalog = pd.DataFrame(rows)
    if political_accounts_path and Path(political_accounts_path).exists():
        political_config = yaml.safe_load(
            Path(political_accounts_path).read_text(encoding="utf-8")
        ) or {}
        handles_by_id = {
            str(item.get("actor_id")): str(item.get("x_handle") or "").lstrip("@").strip()
            for item in political_config.get("political_accounts", [])
            if item.get("actor_id")
        }
        for index, row in catalog.iterrows():
            handle = handles_by_id.get(str(row["target_id"]), "")
            if not handle:
                continue
            catalog.at[index, "aliases"] = _pipe(_as_list(row["aliases"]) + [handle])
            catalog.at[index, "mention_count_aliases"] = _pipe(
                _as_list(row["mention_count_aliases"]) + [handle]
            )
            catalog.at[index, "context_aliases"] = _pipe(
                _as_list(row["context_aliases"]) + [handle]
            )
            catalog.at[index, "handles"] = _pipe(_as_list(row["handles"]) + [handle])

    include_media = bool(settings.get("include_active_media_accounts", True))
    if include_media and media_accounts_path and Path(media_accounts_path).exists():
        media_config = yaml.safe_load(
            Path(media_accounts_path).read_text(encoding="utf-8")
        ) or {}
        media_rows = []
        for media in media_config.get("media_accounts", []):
            if not bool(media.get("active", False)):
                continue
            handle = str((media.get("x_identity") or {}).get("handle") or "").lstrip("@").strip()
            if not handle:
                continue
            media_rows.append(
                {
                    "target_id": "media_{}".format(media.get("media_id")),
                    "target_label": str(media.get("media_name") or media.get("media_id")),
                    "target_type": "media",
                    "aliases": handle,
                    "mention_count_aliases": handle,
                    "context_aliases": "",
                    "handles": handle,
                    "potential_identity_basis": "none",
                    "allow_unlinked_direct_assignment": False,
                    "allow_linked_direct_assignment": False,
                    "catalog_source": "media_accounts",
                    "active": True,
                }
            )
        if media_rows:
            catalog = pd.concat([catalog, pd.DataFrame(media_rows)], ignore_index=True)

    catalog = catalog.loc[catalog["active"]].copy()
    if catalog["target_id"].duplicated().any():
        duplicated = catalog.loc[
            catalog["target_id"].duplicated(keep=False), "target_id"
        ].tolist()
        raise ValueError("target_id duplicados: {}".format(duplicated))

    alias_owner = {}
    collisions = []
    for row in catalog.itertuples(index=False):
        for alias in _as_list(row.aliases):
            normalized = normalize_target_text(alias)
            if not normalized:
                continue
            owner = alias_owner.get(normalized)
            if owner and owner != row.target_id:
                collisions.append((normalized, owner, row.target_id))
            alias_owner[normalized] = row.target_id
    if collisions:
        raise ValueError("Aliases directos ambiguos: {}".format(collisions))

    return catalog.sort_values(["target_type", "target_label"]).reset_index(drop=True)


def load_validated_offense_patterns(path: Union[str, Path]) -> pd.DataFrame:
    """Load manually validated offense families used by the lexical analysis."""
    offense_path = Path(path)
    if not offense_path.exists():
        raise FileNotFoundError("No existe el archivo de ofensas validadas: {}".format(path))
    frame = pd.read_csv(offense_path)
    _validate_columns(frame, ["offense_family", "regex_pattern"], "offense patterns")
    if "meets_selection_rule" in frame.columns:
        selected = frame["meets_selection_rule"].astype(str).str.lower().isin(
            ["true", "1", "yes", "si", "sí"]
        )
        frame = frame.loc[selected].copy()
    frame = frame.dropna(subset=["offense_family", "regex_pattern"]).copy()
    for pattern in frame["regex_pattern"]:
        re.compile(str(pattern))
    return frame.drop_duplicates("offense_family").reset_index(drop=True)


def try_load_spanish_spacy_model(
    model_candidates: Sequence[str] = ("es_core_news_md", "es_core_news_sm"),
):
    """Return the first available Spanish spaCy pipeline, or None."""
    try:
        import spacy
    except ImportError:
        return None
    for model_name in model_candidates:
        try:
            return spacy.load(model_name)
        except OSError:
            continue
    return None


def _alias_regex(alias: str) -> re.Pattern:
    tokens = [re.escape(token) for token in normalize_target_text(alias).split()]
    if not tokens:
        return re.compile(r"(?!x)x")
    return re.compile(r"(?<!\w)" + r"\s+".join(tokens) + r"(?!\w)")


def _compile_target_patterns(catalog: pd.DataFrame) -> list:
    patterns = []
    for row in catalog.itertuples(index=False):
        direct_aliases = _as_list(row.aliases)
        context_aliases = _as_list(row.context_aliases)
        patterns.append(
            {
                "target_id": row.target_id,
                "target_label": row.target_label,
                "target_type": row.target_type,
                "potential_identity_basis": row.potential_identity_basis,
                "allow_unlinked_direct_assignment": bool(
                    row.allow_unlinked_direct_assignment
                ),
                "allow_linked_direct_assignment": bool(
                    row.allow_linked_direct_assignment
                ),
                "direct": [(alias, _alias_regex(alias)) for alias in direct_aliases],
                "context": [(alias, _alias_regex(alias)) for alias in context_aliases],
            }
        )
    return patterns


def _compile_offense_patterns(frame: pd.DataFrame) -> list:
    return [
        (str(row.offense_family), re.compile(str(row.regex_pattern)))
        for row in frame.itertuples(index=False)
    ]


def _token_starts(text: str) -> list:
    return [match.start() for match in re.finditer(r"\b\w+\b", text)]


def _token_index(starts: list, char_position: int) -> int:
    if not starts:
        return 0
    return max(0, int(np.searchsorted(starts, char_position, side="right") - 1))


def _find_alias_matches(text: str, alias_patterns: list) -> list:
    matches = []
    occupied = []
    ordered = sorted(alias_patterns, key=lambda item: len(item[0]), reverse=True)
    for alias, pattern in ordered:
        for match in pattern.finditer(text):
            span = (match.start(), match.end())
            if any(
                not (span[1] <= previous[0] or span[0] >= previous[1])
                for previous in occupied
            ):
                continue
            occupied.append(span)
            matches.append(
                {
                    "alias": alias,
                    "start": match.start(),
                    "end": match.end(),
                }
            )
    return matches


def count_catalog_mentions(
    prepared: pd.DataFrame,
    catalog: pd.DataFrame,
    target_types: Sequence[str] = (
        "person",
        "party",
        "institution",
        "ideological_group",
    ),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Count direct catalog mentions in every comment, regardless of model labels."""
    _validate_columns(prepared, ["tweet_id", "comment_norm_target"], "prepared corpus")
    _validate_columns(
        catalog,
        ["target_id", "target_label", "target_type", "aliases"],
        "target catalog",
    )
    selected_catalog = catalog.loc[catalog["target_type"].isin(target_types)].copy()
    if "mention_count_aliases" in selected_catalog.columns:
        selected_catalog["aliases"] = selected_catalog["mention_count_aliases"]
    texts = prepared[["tweet_id", "comment_norm_target"]].drop_duplicates("tweet_id")
    corpus_size = int(texts["tweet_id"].nunique())
    hostility_lookup = (
        prepared.drop_duplicates("tweet_id")
        .assign(tweet_id=lambda frame: frame["tweet_id"].astype(str))
        .set_index("tweet_id")["hostility_pred_target"]
        .to_dict()
    )
    hate_lookup = (
        prepared.drop_duplicates("tweet_id")
        .assign(tweet_id=lambda frame: frame["tweet_id"].astype(str))
        .set_index("tweet_id")["hate_pred_target"]
        .to_dict()
    )
    target_patterns = _compile_target_patterns(selected_catalog)
    rows = []
    comments_by_type = {target_type: set() for target_type in target_types}
    hostile_comments_by_type = {target_type: set() for target_type in target_types}
    hate_comments_by_type = {target_type: set() for target_type in target_types}
    all_comments = set()
    all_hostile_comments = set()
    all_hate_comments = set()
    for target in target_patterns:
        occurrence_count = 0
        comment_ids = set()
        for tweet_id, text in texts.itertuples(index=False, name=None):
            matches = _find_alias_matches(str(text or ""), target["direct"])
            if not matches:
                continue
            occurrence_count += len(matches)
            comment_ids.add(str(tweet_id))
        comments_by_type.setdefault(target["target_type"], set()).update(comment_ids)
        hostile_comment_ids = {
            tweet_id for tweet_id in comment_ids if int(hostility_lookup.get(tweet_id, 0)) == 1
        }
        hate_comment_ids = {
            tweet_id for tweet_id in comment_ids if int(hate_lookup.get(tweet_id, 0)) == 1
        }
        hostile_comments_by_type.setdefault(target["target_type"], set()).update(
            hostile_comment_ids
        )
        hate_comments_by_type.setdefault(target["target_type"], set()).update(
            hate_comment_ids
        )
        all_comments.update(comment_ids)
        all_hostile_comments.update(hostile_comment_ids)
        all_hate_comments.update(hate_comment_ids)
        comment_count = len(comment_ids)
        hostile_count = len(hostile_comment_ids)
        hate_count = len(hate_comment_ids)
        if comment_count:
            proportion = hostile_count / comment_count
            z_value = 1.96
            denominator = 1 + z_value**2 / comment_count
            centre = (proportion + z_value**2 / (2 * comment_count)) / denominator
            margin = (
                z_value
                * np.sqrt(
                    proportion * (1 - proportion) / comment_count
                    + z_value**2 / (4 * comment_count**2)
                )
                / denominator
            )
            ci_low = 100 * (centre - margin)
            ci_high = 100 * (centre + margin)
        else:
            ci_low = np.nan
            ci_high = np.nan
        rows.append(
            {
                "target_id": target["target_id"],
                "target_label": target["target_label"],
                "target_type": target["target_type"],
                "potential_identity_basis": target["potential_identity_basis"],
                "mention_occurrences": int(occurrence_count),
                "comments_mentioning_entity": int(len(comment_ids)),
                "hostile_comments_mentioning_entity": int(hostile_count),
                "hostility_pct_of_entity_comments": (
                    100 * hostile_count / comment_count if comment_count else np.nan
                ),
                "hostility_ci95_low": ci_low,
                "hostility_ci95_high": ci_high,
                "hate_comments_mentioning_entity_experimental": int(hate_count),
                "hate_pct_of_entity_comments_experimental": (
                    100 * hate_count / comment_count if comment_count else np.nan
                ),
                "comments_pct_corpus": (
                    100 * len(comment_ids) / corpus_size if corpus_size else np.nan
                ),
                "mentions_per_1000_comments": (
                    1000 * occurrence_count / corpus_size if corpus_size else np.nan
                ),
            }
        )
    target_summary = pd.DataFrame(rows).sort_values(
        ["mention_occurrences", "comments_mentioning_entity", "target_label"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    type_rows = []
    for target_type in target_types:
        type_targets = target_summary.loc[target_summary["target_type"].eq(target_type)]
        comment_ids = comments_by_type.get(target_type, set())
        hostile_comment_ids = hostile_comments_by_type.get(target_type, set())
        hate_comment_ids = hate_comments_by_type.get(target_type, set())
        type_rows.append(
            {
                "target_type": target_type,
                "catalog_entities": int(len(type_targets)),
                "mention_occurrences": int(type_targets["mention_occurrences"].sum()),
                "entity_comment_pairs": int(
                    type_targets["comments_mentioning_entity"].sum()
                ),
                "comments_mentioning_type": int(len(comment_ids)),
                "hostile_comments_mentioning_type": int(len(hostile_comment_ids)),
                "hostility_pct_of_type_comments": (
                    100 * len(hostile_comment_ids) / len(comment_ids)
                    if comment_ids
                    else np.nan
                ),
                "hate_comments_mentioning_type_experimental": int(
                    len(hate_comment_ids)
                ),
                "hate_pct_of_type_comments_experimental": (
                    100 * len(hate_comment_ids) / len(comment_ids)
                    if comment_ids
                    else np.nan
                ),
                "comments_pct_corpus": (
                    100 * len(comment_ids) / corpus_size if corpus_size else np.nan
                ),
            }
        )
    type_summary = pd.DataFrame(type_rows)
    overall_summary = pd.DataFrame(
        [
            {
                "included_target_types": "|".join(target_types),
                "catalog_entities": int(len(target_summary)),
                "mention_occurrences": int(target_summary["mention_occurrences"].sum()),
                "entity_comment_pairs": int(
                    target_summary["comments_mentioning_entity"].sum()
                ),
                "comments_with_any_political_entity": int(len(all_comments)),
                "hostile_comments_with_any_political_entity": int(
                    len(all_hostile_comments)
                ),
                "hostility_pct_of_political_entity_comments": (
                    100 * len(all_hostile_comments) / len(all_comments)
                    if all_comments
                    else np.nan
                ),
                "hate_comments_with_any_political_entity_experimental": int(
                    len(all_hate_comments)
                ),
                "hate_pct_of_political_entity_comments_experimental": (
                    100 * len(all_hate_comments) / len(all_comments)
                    if all_comments
                    else np.nan
                ),
                "comments_pct_corpus": (
                    100 * len(all_comments) / corpus_size if corpus_size else np.nan
                ),
                "corpus_comments": corpus_size,
            }
        ]
    )
    return target_summary, type_summary, overall_summary


def save_entity_mention_hostility_figure(
    mention_summary: pd.DataFrame,
    target_ids: Sequence[str],
    output_path: Union[str, Path],
) -> Path:
    """Plot exact hostility rates among comments mentioning selected entities."""
    _validate_columns(
        mention_summary,
        [
            "target_id",
            "target_label",
            "comments_mentioning_entity",
            "hostile_comments_mentioning_entity",
            "hostility_pct_of_entity_comments",
            "mention_occurrences",
        ],
        "mention summary",
    )
    selected = mention_summary.loc[mention_summary["target_id"].isin(target_ids)].copy()
    selected = selected.sort_values("hostility_pct_of_entity_comments").reset_index(drop=True)
    rates = selected["hostility_pct_of_entity_comments"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor("#FAFAF8")
    ax.set_facecolor("#FAFAF8")
    y = np.arange(len(selected))
    ax.barh(
        y,
        rates,
        color="#B94735",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(
        [textwrap.fill(label, 32) for label in selected["target_label"]]
    )
    for index, row in enumerate(selected.itertuples(index=False)):
        label = "{:.2f}% | {} hostiles / {} comentarios | {} menciones".format(
            row.hostility_pct_of_entity_comments,
            int(row.hostile_comments_mentioning_entity),
            int(row.comments_mentioning_entity),
            int(row.mention_occurrences),
        ).replace(".", ",")
        ax.text(
            101.0,
            index,
            label,
            va="center",
            fontsize=9.5,
            clip_on=False,
        )
    ax.set_xlim(0, 100)
    ax.set_xlabel("Comentarios predichos como hostiles entre los que mencionan la entidad (%)")
    ax.set_title(
        "Hostilidad predicha en comentarios que mencionan entidades politicas\n"
        "Perfil balanceado v2 · valores exactos sobre comentarios unicos",
        fontsize=18,
        pad=18,
    )
    ax.grid(True, axis="x", color="#D9D9D6", linewidth=0.7, alpha=0.65)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    fig.text(
        0.5,
        0.015,
        "Cada comentario cuenta una vez por entidad. Coaparicion con hostilidad no confirma que la entidad sea el blanco del ataque.",
        ha="center",
        fontsize=9.5,
        color="#5E5E5E",
    )
    fig.subplots_adjust(left=0.26, right=0.68, top=0.84, bottom=0.11)
    return _save_figure(fig, output_path)


def _find_offense_matches(text: str, patterns: list) -> list:
    matches = []
    for family, pattern in patterns:
        for match in pattern.finditer(text):
            matches.append(
                {
                    "offense_family": family,
                    "start": match.start(),
                    "end": match.end(),
                }
            )
    return matches


def _nearest_target_offense_pair(
    target_matches: list,
    offense_matches: list,
    starts: list,
) -> Tuple[Optional[dict], Optional[dict], float]:
    best_target = None
    best_offense = None
    best_distance = np.nan
    for target_match in target_matches:
        target_token = _token_index(starts, target_match["start"])
        for offense_match in offense_matches:
            offense_token = _token_index(starts, offense_match["start"])
            distance = abs(target_token - offense_token)
            if pd.isna(best_distance) or distance < best_distance:
                best_target = target_match
                best_offense = offense_match
                best_distance = float(distance)
    return best_target, best_offense, best_distance


def _pick_source_text_column(source_posts: pd.DataFrame) -> str:
    for column in ["source_post_text", "text"]:
        if column in source_posts.columns:
            return column
    raise KeyError("No se encontro source_post_text ni text en posts madre")


def prepare_target_inputs(
    corpus: pd.DataFrame,
    source_posts: pd.DataFrame,
    hostility_prediction_column: str = "ml_hostility_pred_v2_balanced",
    hate_prediction_column: str = "ml_hate_speech_pred_v2_balanced_experimental",
) -> pd.DataFrame:
    """Deduplicate comments, attach source-post context, and normalize predictions."""
    _validate_columns(
        corpus,
        ["tweet_id", "anchor_post_id", "text", hostility_prediction_column],
        "corpus",
    )
    _validate_columns(source_posts, ["source_post_id"], "source posts")
    source_text_column = _pick_source_text_column(source_posts)
    source_lookup = (
        source_posts[["source_post_id", source_text_column]]
        .drop_duplicates("source_post_id")
        .rename(columns={source_text_column: "source_post_text_context"})
    )

    work = corpus.drop_duplicates("tweet_id", keep="first").copy()
    work["tweet_id"] = work["tweet_id"].astype("string")
    work["anchor_post_id"] = work["anchor_post_id"].astype("string")
    source_lookup["source_post_id"] = source_lookup["source_post_id"].astype("string")
    work = work.merge(
        source_lookup,
        left_on="anchor_post_id",
        right_on="source_post_id",
        how="left",
        validate="many_to_one",
    )
    work["comment_norm_target"] = work["text"].fillna("").map(normalize_target_text)
    work["source_norm_target"] = (
        work["source_post_text_context"].fillna("").map(normalize_target_text)
    )
    work["hostility_pred_target"] = pd.to_numeric(
        work[hostility_prediction_column], errors="coerce"
    ).fillna(0).astype(int)
    if hate_prediction_column in work.columns:
        work["hate_pred_target"] = pd.to_numeric(
            work[hate_prediction_column], errors="coerce"
        ).fillna(0).astype(int)
    else:
        work["hate_pred_target"] = 0
    return work


def extract_target_evidence(
    prepared: pd.DataFrame,
    catalog: pd.DataFrame,
    offense_patterns: pd.DataFrame,
    max_token_distance: int = 8,
) -> pd.DataFrame:
    """Create one evidence row per comment-target pair."""
    _validate_columns(
        prepared,
        [
            "tweet_id",
            "anchor_post_id",
            "comment_norm_target",
            "source_norm_target",
            "hostility_pred_target",
            "hate_pred_target",
        ],
        "prepared corpus",
    )
    target_patterns = _compile_target_patterns(catalog)
    compiled_offenses = _compile_offense_patterns(offense_patterns)
    metadata_columns = [
        column
        for column in [
            "event_id",
            "event_name",
            "anchor_media_id",
            "anchor_media_handle",
            "source_type",
            "created_at",
        ]
        if column in prepared.columns
    ]
    rows = []
    iterator_columns = [
        "tweet_id",
        "anchor_post_id",
        "comment_norm_target",
        "source_norm_target",
        "hostility_pred_target",
        "hate_pred_target",
    ] + metadata_columns
    for values in prepared[iterator_columns].itertuples(index=False, name=None):
        record = dict(zip(iterator_columns, values))
        comment = str(record["comment_norm_target"] or "")
        source = str(record["source_norm_target"] or "")
        starts = _token_starts(comment)
        offenses = _find_offense_matches(comment, compiled_offenses)
        offense_families = _pipe(item["offense_family"] for item in offenses)

        for target in target_patterns:
            direct_matches = _find_alias_matches(comment, target["direct"])
            context_matches = _find_alias_matches(source, target["context"])
            if not direct_matches and not context_matches:
                continue

            nearest_target, nearest_offense, distance = _nearest_target_offense_pair(
                direct_matches, offenses, starts
            )
            near_families = []
            if direct_matches and offenses:
                for target_match in direct_matches:
                    target_token = _token_index(starts, target_match["start"])
                    for offense in offenses:
                        offense_token = _token_index(starts, offense["start"])
                        if abs(target_token - offense_token) <= max_token_distance:
                            near_families.append(offense["offense_family"])

            row = {
                "tweet_id": record["tweet_id"],
                "anchor_post_id": record["anchor_post_id"],
                "target_id": target["target_id"],
                "target_label": target["target_label"],
                "target_type": target["target_type"],
                "potential_identity_basis": target["potential_identity_basis"],
                "allow_unlinked_direct_assignment": target[
                    "allow_unlinked_direct_assignment"
                ],
                "allow_linked_direct_assignment": target[
                    "allow_linked_direct_assignment"
                ],
                "direct_mention": bool(direct_matches),
                "anchor_mention": bool(context_matches),
                "direct_aliases_found": _pipe(
                    match["alias"] for match in direct_matches
                ),
                "anchor_aliases_found": _pipe(
                    match["alias"] for match in context_matches
                ),
                "offense_present": bool(offenses),
                "offense_families": offense_families,
                "offense_families_near_target": _pipe(near_families),
                "min_token_distance": distance,
                "target_span_start": (
                    nearest_target["start"] if nearest_target else np.nan
                ),
                "target_span_end": nearest_target["end"] if nearest_target else np.nan,
                "offense_span_start": (
                    nearest_offense["start"] if nearest_offense else np.nan
                ),
                "offense_span_end": (
                    nearest_offense["end"] if nearest_offense else np.nan
                ),
                "nearest_offense_family": (
                    nearest_offense["offense_family"] if nearest_offense else ""
                ),
                "same_sentence": pd.NA,
                "dependency_distance": np.nan,
                "dependency_link": False,
                "direct_offense_link": bool(
                    direct_matches
                    and pd.notna(distance)
                    and distance <= max_token_distance
                ),
                "hostility_pred": int(record["hostility_pred_target"]),
                "hate_pred_experimental": int(record["hate_pred_target"]),
                "max_token_distance": int(max_token_distance),
            }
            row.update({column: record[column] for column in metadata_columns})
            rows.append(row)

    columns = [
        "tweet_id",
        "anchor_post_id",
        "target_id",
        "target_label",
        "target_type",
        "potential_identity_basis",
        "allow_unlinked_direct_assignment",
        "allow_linked_direct_assignment",
        "direct_mention",
        "anchor_mention",
        "direct_aliases_found",
        "anchor_aliases_found",
        "offense_present",
        "offense_families",
        "offense_families_near_target",
        "min_token_distance",
        "target_span_start",
        "target_span_end",
        "offense_span_start",
        "offense_span_end",
        "nearest_offense_family",
        "same_sentence",
        "dependency_distance",
        "dependency_link",
        "direct_offense_link",
        "hostility_pred",
        "hate_pred_experimental",
        "max_token_distance",
    ] + metadata_columns
    return pd.DataFrame(rows, columns=columns)


def _dependency_distance(left_token, right_token) -> Optional[int]:
    left_path = [left_token] + list(left_token.ancestors)
    right_path = [right_token] + list(right_token.ancestors)
    right_positions = {token.i: index for index, token in enumerate(right_path)}
    distances = [
        left_index + right_positions[token.i]
        for left_index, token in enumerate(left_path)
        if token.i in right_positions
    ]
    return min(distances) if distances else None


def enrich_dependency_evidence(
    evidence: pd.DataFrame,
    prepared: pd.DataFrame,
    nlp,
    max_dependency_distance: int = 4,
) -> pd.DataFrame:
    """Use sentence and dependency structure to refine direct offense links."""
    if nlp is None or evidence.empty:
        return evidence.copy()
    result = evidence.copy()
    candidates = result.loc[
        result["direct_mention"]
        & result["offense_present"]
        & result["target_span_start"].notna()
        & result["offense_span_start"].notna(),
        "tweet_id",
    ].drop_duplicates()
    if candidates.empty:
        return result

    text_lookup = (
        prepared.loc[
            prepared["tweet_id"].isin(candidates),
            ["tweet_id", "comment_norm_target"],
        ]
        .drop_duplicates("tweet_id")
        .set_index("tweet_id")["comment_norm_target"]
        .to_dict()
    )
    docs = {
        tweet_id: doc
        for tweet_id, doc in zip(
            text_lookup.keys(),
            nlp.pipe(text_lookup.values(), batch_size=64),
        )
    }
    for index in result.index[result["tweet_id"].isin(candidates)]:
        row = result.loc[index]
        doc = docs.get(row["tweet_id"])
        if doc is None:
            continue
        try:
            target_span = doc.char_span(
                int(row["target_span_start"]),
                int(row["target_span_end"]),
                alignment_mode="expand",
            )
            offense_span = doc.char_span(
                int(row["offense_span_start"]),
                int(row["offense_span_end"]),
                alignment_mode="expand",
            )
        except (TypeError, ValueError):
            continue
        if target_span is None or offense_span is None:
            continue
        same_sentence = target_span.root.sent.start == offense_span.root.sent.start
        dependency_distance = _dependency_distance(target_span.root, offense_span.root)
        dependency_link = bool(
            same_sentence
            and dependency_distance is not None
            and dependency_distance <= max_dependency_distance
        )
        proximity_link = bool(
            same_sentence
            and pd.notna(row["min_token_distance"])
            and float(row["min_token_distance"]) <= float(row["max_token_distance"])
        )
        result.at[index, "same_sentence"] = bool(same_sentence)
        result.at[index, "dependency_distance"] = dependency_distance
        result.at[index, "dependency_link"] = dependency_link
        result.at[index, "direct_offense_link"] = dependency_link or proximity_link
    return result


def build_target_assignments(
    prepared: pd.DataFrame,
    evidence: pd.DataFrame,
    catalog: pd.DataFrame,
) -> pd.DataFrame:
    """Choose at most one probable target per comment and retain ambiguity."""
    catalog_lookup = catalog.set_index("target_id").to_dict(orient="index")
    evidence_groups = {
        tweet_id: group.copy()
        for tweet_id, group in evidence.groupby("tweet_id", sort=False)
    }
    metadata_columns = [
        column
        for column in [
            "event_id",
            "event_name",
            "anchor_media_id",
            "anchor_media_handle",
            "source_type",
            "created_at",
        ]
        if column in prepared.columns
    ]
    rows = []
    iterator_columns = [
        "tweet_id",
        "anchor_post_id",
        "hostility_pred_target",
        "hate_pred_target",
    ] + metadata_columns
    for values in prepared[iterator_columns].itertuples(index=False, name=None):
        record = dict(zip(iterator_columns, values))
        group = evidence_groups.get(record["tweet_id"], pd.DataFrame())
        if group.empty:
            linked_ids = []
            direct_ids = []
            anchor_ids = []
            all_ids = []
            offense_families = ""
        else:
            linked_ids = sorted(
                group.loc[
                    group["direct_offense_link"]
                    & group["allow_linked_direct_assignment"],
                    "target_id",
                ].unique().tolist()
            )
            direct_ids = sorted(
                group.loc[
                    group["direct_mention"]
                    & group["allow_unlinked_direct_assignment"],
                    "target_id",
                ].unique().tolist()
            )
            anchor_ids = sorted(
                group.loc[group["anchor_mention"], "target_id"].unique().tolist()
            )
            all_ids = sorted(group["target_id"].unique().tolist())
            offense_families = _pipe(
                family
                for value in group["offense_families"].dropna()
                for family in _as_list(value)
            )

        primary_id = ""
        if len(linked_ids) == 1:
            confidence = "high"
            basis = "direct_offense_entity_link"
            primary_id = linked_ids[0]
        elif len(linked_ids) > 1:
            confidence = "ambiguous"
            basis = "multiple_direct_offense_links"
        elif len(direct_ids) == 1:
            confidence = "medium"
            basis = "single_direct_mention"
            primary_id = direct_ids[0]
        elif len(direct_ids) > 1:
            confidence = "ambiguous"
            basis = "multiple_direct_mentions"
        elif len(anchor_ids) == 1:
            confidence = "low"
            basis = "single_anchor_context_target"
            primary_id = anchor_ids[0]
        elif len(anchor_ids) > 1:
            confidence = "ambiguous"
            basis = "multiple_anchor_context_targets"
        else:
            confidence = "unassigned"
            basis = "no_catalog_target_detected"

        primary = catalog_lookup.get(primary_id, {})
        candidate_ids = linked_ids or direct_ids or anchor_ids or all_ids
        candidate_labels = [
            catalog_lookup.get(target_id, {}).get("target_label", target_id)
            for target_id in candidate_ids
        ]
        row = {
            "tweet_id": record["tweet_id"],
            "anchor_post_id": record["anchor_post_id"],
            "primary_target_id": primary_id,
            "primary_target_label": primary.get("target_label", ""),
            "primary_target_type": primary.get("target_type", ""),
            "primary_target_potential_identity_basis": primary.get(
                "potential_identity_basis", ""
            ),
            "target_confidence": confidence,
            "target_assignment_basis": basis,
            "target_candidate_ids": "|".join(candidate_ids),
            "target_candidate_labels": "|".join(candidate_labels),
            "direct_linked_target_count": len(linked_ids),
            "direct_target_count": len(direct_ids),
            "anchor_target_count": len(anchor_ids),
            "all_catalog_target_count": len(all_ids),
            "offense_present": bool(offense_families),
            "offense_families": offense_families,
            "hostility_pred": int(record["hostility_pred_target"]),
            "hate_pred_experimental": int(record["hate_pred_target"]),
        }
        row.update({column: record[column] for column in metadata_columns})
        rows.append(row)
    return pd.DataFrame(rows)


def build_assignment_coverage(assignments: pd.DataFrame) -> pd.DataFrame:
    """Summarize assignment confidence for all, hostile, and hate subsets."""
    _validate_columns(
        assignments,
        ["tweet_id", "target_confidence", "hostility_pred", "hate_pred_experimental"],
        "assignments",
    )
    populations = [
        ("all_comments", assignments),
        ("hostility_balanced", assignments.loc[assignments["hostility_pred"].eq(1)]),
        ("hate_balanced_experimental", assignments.loc[assignments["hate_pred_experimental"].eq(1)]),
    ]
    rows = []
    for population, subset in populations:
        total = int(subset["tweet_id"].nunique())
        counts = subset["target_confidence"].value_counts()
        for confidence in ASSIGNMENT_ORDER:
            count = int(counts.get(confidence, 0))
            rows.append(
                {
                    "population": population,
                    "target_confidence": confidence,
                    "n_comments": count,
                    "percentage": 100 * count / total if total else np.nan,
                    "population_total": total,
                }
            )
    return pd.DataFrame(rows)


def build_target_scope_summary(evidence: pd.DataFrame) -> pd.DataFrame:
    """Compare direct mentions, linked offenses, and anchor context separately."""
    scopes = []
    definitions = [
        ("direct_mention", "direct_mention"),
        ("direct_offense_link", "direct_offense_link"),
        ("anchor_context", "anchor_mention"),
    ]
    for scope_name, flag_column in definitions:
        subset = evidence.loc[evidence[flag_column]].copy()
        if scope_name in {"direct_mention", "direct_offense_link"}:
            subset = subset.loc[subset["target_type"].ne("media")].copy()
        if subset.empty:
            continue
        subset["evidence_scope"] = scope_name
        scopes.append(subset)
    if not scopes:
        return pd.DataFrame()
    work = pd.concat(scopes, ignore_index=True).drop_duplicates(
        ["tweet_id", "target_id", "evidence_scope"]
    )
    summary = (
        work.groupby(
            [
                "evidence_scope",
                "target_id",
                "target_label",
                "target_type",
                "potential_identity_basis",
            ],
            as_index=False,
        )
        .agg(
            n_comments=("tweet_id", "nunique"),
            hostile_n=("hostility_pred", "sum"),
            hate_n_experimental=("hate_pred_experimental", "sum"),
        )
    )
    summary["hostility_pct"] = 100 * summary["hostile_n"] / summary["n_comments"]
    summary["hate_pct_experimental"] = (
        100 * summary["hate_n_experimental"] / summary["n_comments"]
    )
    return summary.sort_values(
        ["evidence_scope", "hostile_n", "n_comments"], ascending=[True, False, False]
    ).reset_index(drop=True)


def build_primary_target_summary(assignments: pd.DataFrame) -> pd.DataFrame:
    """Summarize unique primary assignments, retaining confidence composition."""
    assigned = assignments.loc[assignments["primary_target_id"].ne("")].copy()
    if assigned.empty:
        return pd.DataFrame()
    summary = (
        assigned.groupby(
            [
                "primary_target_id",
                "primary_target_label",
                "primary_target_type",
                "primary_target_potential_identity_basis",
            ],
            as_index=False,
        )
        .agg(
            assigned_comments=("tweet_id", "nunique"),
            hostile_n=("hostility_pred", "sum"),
            hate_n_experimental=("hate_pred_experimental", "sum"),
            high_confidence_n=("target_confidence", lambda values: int((values == "high").sum())),
            medium_confidence_n=("target_confidence", lambda values: int((values == "medium").sum())),
            low_confidence_n=("target_confidence", lambda values: int((values == "low").sum())),
        )
    )
    summary["hostility_pct"] = 100 * summary["hostile_n"] / summary["assigned_comments"]
    summary["hate_pct_experimental"] = (
        100 * summary["hate_n_experimental"] / summary["assigned_comments"]
    )
    return summary.sort_values(
        ["hostile_n", "assigned_comments"], ascending=False
    ).reset_index(drop=True)


def build_target_type_summary(assignments: pd.DataFrame) -> pd.DataFrame:
    assigned = assignments.loc[assignments["primary_target_id"].ne("")].copy()
    if assigned.empty:
        return pd.DataFrame()
    summary = (
        assigned.groupby("primary_target_type", as_index=False)
        .agg(
            assigned_comments=("tweet_id", "nunique"),
            hostile_n=("hostility_pred", "sum"),
            hate_n_experimental=("hate_pred_experimental", "sum"),
        )
    )
    summary["hostility_pct"] = 100 * summary["hostile_n"] / summary["assigned_comments"]
    summary["hate_pct_experimental"] = (
        100 * summary["hate_n_experimental"] / summary["assigned_comments"]
    )
    return summary.sort_values("hostile_n", ascending=False).reset_index(drop=True)


def build_offense_target_pairs(evidence: pd.DataFrame) -> pd.DataFrame:
    """Aggregate high-confidence target/offense relationships without raw text."""
    linked = evidence.loc[
        evidence["direct_offense_link"]
        & evidence["allow_linked_direct_assignment"]
        & evidence["offense_families_near_target"].fillna("").ne("")
    ].copy()
    if linked.empty:
        return pd.DataFrame(
            columns=[
                "target_id",
                "target_label",
                "target_type",
                "offense_family",
                "n_comments",
                "hostile_n",
                "hate_n_experimental",
            ]
        )
    linked["offense_family"] = linked["offense_families_near_target"].map(_as_list)
    linked = linked.explode("offense_family").drop_duplicates(
        ["tweet_id", "target_id", "offense_family"]
    )
    return (
        linked.groupby(
            ["target_id", "target_label", "target_type", "offense_family"],
            as_index=False,
        )
        .agg(
            n_comments=("tweet_id", "nunique"),
            hostile_n=("hostility_pred", "sum"),
            hate_n_experimental=("hate_pred_experimental", "sum"),
        )
        .sort_values(["n_comments", "hostile_n"], ascending=False)
        .reset_index(drop=True)
    )


def build_relation_templates(
    prepared: pd.DataFrame,
    evidence: pd.DataFrame,
    max_template_tokens: int = 12,
) -> pd.DataFrame:
    """Build target/offense phrase templates from linked spans."""
    linked = evidence.loc[
        evidence["direct_offense_link"]
        & evidence["allow_linked_direct_assignment"]
        & evidence["target_span_start"].notna()
        & evidence["offense_span_start"].notna()
    ].copy()
    if linked.empty:
        return pd.DataFrame()
    text_lookup = prepared.set_index("tweet_id")["comment_norm_target"].to_dict()
    rows = []
    for row in linked.itertuples(index=False):
        text = text_lookup.get(row.tweet_id, "")
        token_matches = list(re.finditer(r"\b\w+\b", text))
        if not token_matches:
            continue
        target_indices = [
            index
            for index, match in enumerate(token_matches)
            if match.end() > int(row.target_span_start)
            and match.start() < int(row.target_span_end)
        ]
        offense_indices = [
            index
            for index, match in enumerate(token_matches)
            if match.end() > int(row.offense_span_start)
            and match.start() < int(row.offense_span_end)
        ]
        if not target_indices or not offense_indices:
            continue
        target_start, target_end = min(target_indices), max(target_indices)
        offense_start, offense_end = min(offense_indices), max(offense_indices)
        left = min(target_start, offense_start)
        right = max(target_end, offense_end)
        if right - left + 1 > max_template_tokens:
            continue
        tokens = []
        index = left
        while index <= right:
            if index == target_start:
                tokens.append("<TARGET>")
                index = target_end + 1
                continue
            if index == offense_start:
                tokens.append("<OFFENSE>")
                index = offense_end + 1
                continue
            tokens.append(token_matches[index].group(0))
            index += 1
        rows.append(
            {
                "relation_template": " ".join(tokens),
                "template_tokens": len(tokens),
                "target_id": row.target_id,
                "target_label": row.target_label,
                "offense_family": row.nearest_offense_family,
                "tweet_id": row.tweet_id,
                "hostility_pred": row.hostility_pred,
                "hate_pred_experimental": row.hate_pred_experimental,
            }
        )
    if not rows:
        return pd.DataFrame()
    templates = pd.DataFrame(rows).drop_duplicates(
        ["tweet_id", "target_id", "relation_template"]
    )
    return (
        templates.groupby(
            ["relation_template", "template_tokens", "offense_family"],
            as_index=False,
        )
        .agg(
            n_comments=("tweet_id", "nunique"),
            n_targets=("target_id", "nunique"),
            hostile_n=("hostility_pred", "sum"),
            hate_n_experimental=("hate_pred_experimental", "sum"),
        )
        .sort_values(["n_comments", "hostile_n"], ascending=False)
        .reset_index(drop=True)
    )


def discover_uncatalogued_source_entities(
    prepared: pd.DataFrame,
    catalog: pd.DataFrame,
    nlp,
    min_source_posts: int = 3,
) -> pd.DataFrame:
    """Find repeated PER/ORG entities in media source posts for manual catalog review."""
    if nlp is None:
        return pd.DataFrame()
    source_posts = prepared.loc[
        prepared["hostility_pred_target"].eq(1),
        ["anchor_post_id", "source_post_text_context"],
    ].drop_duplicates("anchor_post_id")
    known_aliases = {
        normalize_target_text(alias)
        for value in catalog["context_aliases"]
        for alias in _as_list(value)
    }
    counter = Counter()
    labels = Counter()
    examples = {}
    texts = source_posts["source_post_text_context"].fillna("").astype(str).tolist()
    source_ids = source_posts["anchor_post_id"].tolist()
    seen_pairs = set()
    for source_id, doc in zip(source_ids, nlp.pipe(texts, batch_size=64)):
        for entity in doc.ents:
            if entity.label_ not in {"PER", "ORG"}:
                continue
            normalized = normalize_target_text(entity.text)
            if len(normalized) < 3 or normalized in known_aliases:
                continue
            pair = (source_id, normalized)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            counter[normalized] += 1
            labels[(normalized, entity.label_)] += 1
            examples.setdefault(normalized, entity.text.strip())
    rows = []
    for entity_norm, count in counter.items():
        if count < min_source_posts:
            continue
        entity_label = max(
            (key for key in labels if key[0] == entity_norm),
            key=lambda key: labels[key],
        )[1]
        rows.append(
            {
                "entity_norm": entity_norm,
                "entity_display": examples[entity_norm],
                "spacy_label": entity_label,
                "source_post_count": count,
                "review_status": "pending_manual_review",
                "add_to_catalog": "",
                "notes": "",
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["source_post_count", "entity_norm"], ascending=[False, True]
    ).reset_index(drop=True)


def build_target_manual_review_candidate(
    prepared: pd.DataFrame,
    assignments: pd.DataFrame,
    sample_size: int = 300,
    random_state: int = 42,
) -> pd.DataFrame:
    """Create a separate target-validation sample without touching canonical labels."""
    columns = [
        "tweet_id",
        "text",
        "source_post_text_context",
        "event_id",
        "event_name",
        "anchor_media_handle",
        "anchor_post_id",
        "hostility_pred",
        "hate_pred_experimental",
        "primary_target_id",
        "primary_target_label",
        "primary_target_type",
        "target_confidence",
        "target_assignment_basis",
        "target_candidate_labels",
        "offense_families",
    ]
    merged = assignments.merge(
        prepared[["tweet_id", "text", "source_post_text_context"]],
        on="tweet_id",
        how="left",
        validate="one_to_one",
    )
    for column in columns:
        if column not in merged.columns:
            merged[column] = ""
    pool = merged.loc[
        merged["hostility_pred"].eq(1) | merged["hate_pred_experimental"].eq(1)
    ].copy()
    priority_map = {"high": 0, "ambiguous": 1, "medium": 2, "low": 3, "unassigned": 4}
    pool["_priority"] = pool["target_confidence"].map(priority_map).fillna(5)
    pool["_hate_priority"] = -pool["hate_pred_experimental"]
    pool = pool.sort_values(
        ["_priority", "_hate_priority", "tweet_id"], ascending=[True, True, True]
    )
    deterministic_size = min(len(pool), max(0, sample_size // 2))
    selected = pool.head(deterministic_size)
    remaining = pool.loc[~pool["tweet_id"].isin(selected["tweet_id"])]
    random_size = min(len(remaining), max(0, sample_size - len(selected)))
    if random_size:
        selected = pd.concat(
            [selected, remaining.sample(random_size, random_state=random_state)],
            ignore_index=True,
        )
    selected = selected.drop_duplicates("tweet_id").head(sample_size).copy()
    selected.insert(0, "review_id", ["target_{:04d}".format(i + 1) for i in range(len(selected))])
    selected["manual_target_entity"] = ""
    selected["manual_target_type"] = ""
    selected["manual_target_confirmed"] = ""
    selected["manual_identity_based_attack"] = ""
    selected["manual_target_confidence"] = ""
    selected["target_notes"] = ""
    output_columns = ["review_id"] + columns + [
        "manual_target_entity",
        "manual_target_type",
        "manual_target_confirmed",
        "manual_identity_based_attack",
        "manual_target_confidence",
        "target_notes",
    ]
    return selected[output_columns]


def _save_figure(fig: plt.Figure, output_path: Union[str, Path]) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def save_assignment_coverage_figure(
    coverage: pd.DataFrame,
    output_path: Union[str, Path],
) -> Path:
    """Save assignment-confidence composition for each model-defined population."""
    pivot = coverage.pivot(
        index="population", columns="target_confidence", values="percentage"
    ).reindex(
        index=["all_comments", "hostility_balanced", "hate_balanced_experimental"],
        columns=ASSIGNMENT_ORDER,
    ).fillna(0)
    labels = ["Corpus completo", "Hostilidad balanceada", "Odio balanceado\nexperimental"]
    colors = ["#7A2E2A", "#C75B39", "#E39B55", "#8C8983", "#D7D4CD"]
    fig, ax = plt.subplots(figsize=(13, 6.8))
    fig.patch.set_facecolor("#FAFAF8")
    ax.set_facecolor("#FAFAF8")
    left = np.zeros(len(pivot))
    for confidence, color in zip(ASSIGNMENT_ORDER, colors):
        values = pivot[confidence].to_numpy(dtype=float)
        bars = ax.barh(labels, values, left=left, color=color, label=ASSIGNMENT_LABELS[confidence].replace("\n", " "))
        for bar, value, start in zip(bars, values, left):
            if value < 4:
                continue
            ax.text(
                start + value / 2,
                bar.get_y() + bar.get_height() / 2,
                "{:.1f}%".format(value).replace(".", ","),
                ha="center",
                va="center",
                color="white" if confidence in {"high", "medium"} else "#252525",
                fontsize=9.5,
                fontweight="bold",
            )
        left += values
    ax.set_xlim(0, 100)
    ax.set_xlabel("Porcentaje de comentarios")
    ax.set_title(
        "Cobertura de la inferencia de blancos probables\n"
        "La confianza baja depende exclusivamente del contexto del post madre",
        fontsize=18,
        pad=18,
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False)
    ax.grid(True, axis="x", color="#D9D9D6", linewidth=0.7, alpha=0.6)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.text(
        0.5,
        0.015,
        "Una asignacion automatica es una hipotesis para validacion, no una confirmacion del destinatario.",
        ha="center",
        fontsize=9.5,
        color="#5E5E5E",
    )
    fig.subplots_adjust(left=0.20, right=0.98, top=0.80, bottom=0.27)
    return _save_figure(fig, output_path)


def save_direct_target_mentions_figure(
    scope_summary: pd.DataFrame,
    output_path: Union[str, Path],
    top_n: int = 15,
) -> Path:
    """Compare direct mentions in hostile comments with linked offense evidence."""
    direct = scope_summary.loc[scope_summary["evidence_scope"].eq("direct_mention")].copy()
    linked = scope_summary.loc[
        scope_summary["evidence_scope"].eq("direct_offense_link"),
        ["target_id", "hostile_n"],
    ].rename(columns={"hostile_n": "linked_hostile_n"})
    direct = direct.merge(linked, on="target_id", how="left")
    direct["linked_hostile_n"] = direct["linked_hostile_n"].fillna(0).astype(int)
    direct = direct.sort_values("hostile_n", ascending=False).head(top_n).sort_values("hostile_n")

    fig, ax = plt.subplots(figsize=(13, 8.5))
    fig.patch.set_facecolor("#FAFAF8")
    ax.set_facecolor("#FAFAF8")
    y = np.arange(len(direct))
    ax.barh(y, direct["hostile_n"], color="#D99A56", label="Mencion en comentario hostil")
    ax.barh(y, direct["linked_hostile_n"], color="#7A2E2A", label="Ofensa vinculada directamente")
    ax.set_yticks(y)
    ax.set_yticklabels([textwrap.fill(label, 30) for label in direct["target_label"]])
    for index, row in enumerate(direct.itertuples(index=False)):
        ax.text(row.hostile_n + 0.6, index, str(int(row.hostile_n)), va="center", fontsize=9.5)
    ax.set_xlabel("Comentarios predichos como hostiles")
    ax.set_title(
        "Entidades mencionadas en comentarios hostiles\n"
        "La franja oscura representa evidencia directa ofensa-entidad",
        fontsize=18,
        pad=18,
    )
    ax.legend(loc="lower right", frameon=False)
    ax.grid(True, axis="x", color="#D9D9D6", linewidth=0.7, alpha=0.65)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.text(
        0.5,
        0.015,
        "Mencionar una entidad no confirma que sea el blanco del ataque. Los conteos no son mutuamente excluyentes.",
        ha="center",
        fontsize=9.5,
        color="#5E5E5E",
    )
    fig.subplots_adjust(left=0.30, right=0.96, top=0.84, bottom=0.10)
    return _save_figure(fig, output_path)


def save_anchor_context_rate_figure(
    scope_summary: pd.DataFrame,
    output_path: Union[str, Path],
    min_comments: int = 30,
    top_n: int = 15,
) -> Path:
    """Save normalized hostility rates for source-post target contexts."""
    context = scope_summary.loc[
        scope_summary["evidence_scope"].eq("anchor_context")
        & scope_summary["n_comments"].ge(min_comments)
    ].copy()
    context = context.sort_values("hostility_pct", ascending=False).head(top_n).sort_values("hostility_pct")
    fig, ax = plt.subplots(figsize=(13, 8.5))
    fig.patch.set_facecolor("#FAFAF8")
    ax.set_facecolor("#FAFAF8")
    y = np.arange(len(context))
    ax.barh(y, context["hostility_pct"], color="#3C6874")
    ax.set_yticks(y)
    ax.set_yticklabels([textwrap.fill(label, 30) for label in context["target_label"]])
    for index, row in enumerate(context.itertuples(index=False)):
        ax.text(
            row.hostility_pct + 0.5,
            index,
            "{:.1f}% (n={})".format(row.hostility_pct, int(row.n_comments)).replace(".", ","),
            va="center",
            fontsize=9.3,
        )
    ax.set_xlabel("Comentarios predichos como hostiles (%)")
    ax.set_xlim(0, max(65, float(context["hostility_pct"].max()) + 12) if not context.empty else 65)
    ax.set_title(
        "Hostilidad en conversaciones cuyo post madre menciona cada entidad\n"
        "Contexto tematico, no destinatario confirmado",
        fontsize=18,
        pad=18,
    )
    ax.grid(True, axis="x", color="#D9D9D6", linewidth=0.7, alpha=0.65)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.text(
        0.5,
        0.015,
        "Solo se muestran contextos con al menos {} comentarios. La tasa debe interpretarse junto con la exposicion.".format(min_comments),
        ha="center",
        fontsize=9.5,
        color="#5E5E5E",
    )
    fig.subplots_adjust(left=0.30, right=0.94, top=0.84, bottom=0.10)
    return _save_figure(fig, output_path)

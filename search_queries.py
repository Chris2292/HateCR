from __future__ import annotations

import re
import unicodedata
from typing import Any

from src.x_api import parse_datetime_utc, to_rfc3339_utc


# Default grupos para recuperar posts madre de medios en el piloto.
DEFAULT_SOURCE_TERM_GROUPS = [
    "campaign_general",
    "campaign_events",
    "institutions",
    "candidates_main",
    "additional_candidates",
    "parties_main",
    "parties_additional",
    "government_context",
    "conflict_terms",
    "polarization_terms",
    "security_and_crime",
    "economy_social_policy",
]

# Se mantiene por compatibilidad, pero el pipeline debe priorizar grupos en YAML.
DEFAULT_BASE_QUERY_TERMS: list[str] = []


def _normalize_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _strip_accents(value: str) -> str:
    text = unicodedata.normalize("NFD", str(value or ""))
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", text)


def dedupe_keep_order(values: list[str]) -> list[str]:
    seen = set()
    out: list[str] = []
    for value in values:
        text = _normalize_spaces(str(value or ""))
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def sanitize_handle(handle: str | None) -> str:
    return str(handle or "").strip().lstrip("@")


def quote_term(term: str) -> str:
    return '"' + str(term).replace('"', '\\"').strip() + '"'


def get_terms_from_group(term_groups: dict[str, Any], group_name: str) -> list[str]:
    group = term_groups.get(group_name, {}) if isinstance(term_groups, dict) else {}
    if isinstance(group, dict):
        terms = group.get("terms", [])
        if isinstance(terms, list):
            return [str(t) for t in terms if _normalize_spaces(str(t))]
    return []


def build_base_terms_from_groups(
    term_groups: dict,
    group_names: list[str],
    include_accents_variants: bool = True,
) -> list[str]:
    terms_out: list[str] = []
    seen = set()

    for group_name in group_names:
        for raw_term in get_terms_from_group(term_groups, str(group_name)):
            term = _normalize_spaces(raw_term)
            if not term:
                continue

            key = term.casefold()
            if key not in seen:
                seen.add(key)
                terms_out.append(term)

            if include_accents_variants:
                no_acc = _normalize_spaces(_strip_accents(term))
                if no_acc and no_acc.casefold() not in seen:
                    seen.add(no_acc.casefold())
                    terms_out.append(no_acc)

    return terms_out


def chunk_terms_for_x_query(
    terms: list[str],
    max_terms: int,
    max_chars: int,
) -> list[list[str]]:
    ordered_terms = dedupe_keep_order(terms)
    if not ordered_terms:
        return []

    max_terms = max(1, int(max_terms))
    max_chars = max(1, int(max_chars))

    batches: list[list[str]] = []
    current: list[str] = []
    current_chars = 0

    for term in ordered_terms:
        quoted = quote_term(term)
        quoted_len = len(quoted)

        if not current:
            current = [term]
            current_chars = quoted_len
            continue

        projected_chars = current_chars + len(" OR ") + quoted_len
        if len(current) >= max_terms or projected_chars > max_chars:
            batches.append(current)
            current = [term]
            current_chars = quoted_len
        else:
            current.append(term)
            current_chars = projected_chars

    if current:
        batches.append(current)

    return batches


def _resolve_window_for_event(
    event: dict[str, Any],
    global_start_time: str | None = None,
    global_end_time: str | None = None,
) -> tuple[str | None, str | None] | None:
    start_dt = parse_datetime_utc(event.get("collection_window", {}).get("start"))
    end_dt = parse_datetime_utc(event.get("collection_window", {}).get("end"))

    global_start_dt = parse_datetime_utc(global_start_time) if global_start_time else None
    global_end_dt = parse_datetime_utc(global_end_time) if global_end_time else None

    if global_start_dt:
        if end_dt and end_dt < global_start_dt:
            return None
        if start_dt is None or start_dt < global_start_dt:
            start_dt = global_start_dt

    if global_end_dt:
        if start_dt and start_dt > global_end_dt:
            return None
        if end_dt is None or end_dt > global_end_dt:
            end_dt = global_end_dt

    if start_dt and end_dt and start_dt > end_dt:
        return None

    return to_rfc3339_utc(start_dt), to_rfc3339_utc(end_dt)


def _expand_modes(post_search_mode: str) -> list[str]:
    mode = str(post_search_mode or "both").strip().lower()
    if mode == "both":
        return ["broad", "targeted"]
    if mode in {"broad", "targeted"}:
        return [mode]
    raise ValueError("POST_SEARCH_MODE must be one of: both, broad, targeted")


def build_event_terms(
    event: dict[str, Any],
    term_groups: dict[str, Any],
    base_query_terms: list[str] | None = None,
    default_group_names: list[str] | None = None,
    include_accents_variants: bool = True,
    include_hostility_terms_in_source_search: bool = False,
) -> dict[str, Any]:
    profile = event.get("search_profile", {}) if isinstance(event, dict) else {}
    include_groups = list(profile.get("include_groups", []) or [])
    optional_groups = list(profile.get("optional_groups", []) or [])
    exclude_groups = list(profile.get("exclude_groups", []) or [])

    if not include_groups and default_group_names:
        include_groups = list(default_group_names)

    if not include_hostility_terms_in_source_search:
        include_groups = [g for g in include_groups if str(g) != "hostility_terms"]
        optional_groups = [g for g in optional_groups if str(g) != "hostility_terms"]

    group_names_used = dedupe_keep_order([str(g) for g in include_groups + optional_groups])

    group_terms = build_base_terms_from_groups(
        term_groups=term_groups,
        group_names=group_names_used,
        include_accents_variants=include_accents_variants,
    )
    base_terms = dedupe_keep_order(base_query_terms or [])
    positive_terms = dedupe_keep_order(base_terms + group_terms)

    negative_terms = build_base_terms_from_groups(
        term_groups=term_groups,
        group_names=[str(g) for g in exclude_groups],
        include_accents_variants=False,
    )

    return {
        "positive_terms": positive_terms,
        "negative_terms": negative_terms,
        "term_groups_used": group_names_used,
    }


def build_post_query(
    handle: str,
    mode: str,
    positive_terms: list[str] | None = None,
    negative_terms: list[str] | None = None,
) -> str:
    handle_clean = sanitize_handle(handle)
    if not handle_clean:
        raise ValueError("Cannot build query without a valid handle.")

    mode_clean = str(mode).strip().lower()
    if mode_clean not in {"broad", "targeted"}:
        raise ValueError("mode must be 'broad' or 'targeted'.")

    if mode_clean == "broad":
        return f"from:{handle_clean} lang:es -is:retweet"

    positives = dedupe_keep_order(list(positive_terms or []))
    if not positives:
        raise ValueError("Targeted mode requires positive terms.")

    positive_clause = " OR ".join(quote_term(term) for term in positives)
    query = f"from:{handle_clean} ({positive_clause}) lang:es -is:retweet"

    negatives = dedupe_keep_order(list(negative_terms or []))
    if negatives:
        query = query + " " + " ".join(f"-{quote_term(term)}" for term in negatives)

    return query


def build_event_media_query_specs(
    events: list[dict[str, Any]],
    media_accounts: list[dict[str, Any]],
    term_groups: dict[str, Any],
    post_search_mode: str = "both",
    base_query_terms: list[str] | None = None,
    max_terms: int = 30,
    global_start_time: str | None = None,
    global_end_time: str | None = None,
    max_terms_per_query: int = 20,
    max_query_chars: int = 900,
    include_accents_variants: bool = True,
    include_hostility_terms_in_source_search: bool = False,
    default_group_names: list[str] | None = None,
    collection_scope: str = "media_anchored",
) -> list[dict[str, Any]]:
    modes = _expand_modes(post_search_mode)
    scope = str(collection_scope or "media_anchored").strip().lower()

    specs: list[dict[str, Any]] = []

    for event in events:
        window = _resolve_window_for_event(
            event,
            global_start_time=global_start_time,
            global_end_time=global_end_time,
        )
        if window is None:
            continue
        start_time, end_time = window

        event_terms = build_event_terms(
            event=event,
            term_groups=term_groups,
            base_query_terms=base_query_terms,
            default_group_names=default_group_names or DEFAULT_SOURCE_TERM_GROUPS,
            include_accents_variants=include_accents_variants,
            include_hostility_terms_in_source_search=include_hostility_terms_in_source_search,
        )

        positive_terms = event_terms["positive_terms"]
        negative_terms = event_terms["negative_terms"]
        term_groups_used = event_terms["term_groups_used"]

        for media in media_accounts:
            handle = sanitize_handle(media.get("x_identity", {}).get("handle"))
            if not handle:
                continue

            # Presupuesto de caracteres para la clausula positiva, reservando el resto
            # para from:, lang:, -is:retweet y posibles exclusiones.
            negative_clause = ""
            if negative_terms:
                negative_clause = " " + " ".join(f"-{quote_term(term)}" for term in negative_terms)
            wrapper_text = f"from:{handle} () lang:es -is:retweet{negative_clause}"
            positive_clause_budget = max(1, int(max_query_chars) - len(wrapper_text))

            targeted_batches = chunk_terms_for_x_query(
                terms=positive_terms,
                max_terms=max_terms_per_query if max_terms_per_query else max_terms,
                max_chars=positive_clause_budget,
            )

            for mode in modes:
                mode_clean = str(mode).strip().lower()

                if mode_clean == "broad":
                    query = build_post_query(
                        handle=handle,
                        mode="broad",
                    )
                    if scope == "media_anchored" and f"from:{handle.lower()}" not in query.lower():
                        raise ValueError(
                            f"Media-anchored requiere from:{handle} en query broad. query={query}"
                        )

                    specs.append(
                        {
                            "event_id": str(event.get("event_id")),
                            "event_name": event.get("event_name"),
                            "event_date": event.get("event_date"),
                            "media_id": str(media.get("media_id")),
                            "media_name": media.get("media_name"),
                            "media_handle": handle,
                            "search_mode": "broad",
                            "term_batch_id": "broad",
                            "term_count": 0,
                            "term_group_names": "|".join(term_groups_used),
                            "query": query,
                            "start_time": start_time,
                            "end_time": end_time,
                            "positive_terms": [],
                            "negative_terms": negative_terms,
                            "term_groups_used": term_groups_used,
                        }
                    )
                    continue

                # targeted mode
                if not targeted_batches:
                    continue

                for batch_idx, term_batch in enumerate(targeted_batches, start=1):
                    query = build_post_query(
                        handle=handle,
                        mode="targeted",
                        positive_terms=term_batch,
                        negative_terms=negative_terms,
                    )
                    if scope == "media_anchored" and f"from:{handle.lower()}" not in query.lower():
                        raise ValueError(
                            f"Media-anchored requiere from:{handle} en query targeted. query={query}"
                        )

                    specs.append(
                        {
                            "event_id": str(event.get("event_id")),
                            "event_name": event.get("event_name"),
                            "event_date": event.get("event_date"),
                            "media_id": str(media.get("media_id")),
                            "media_name": media.get("media_name"),
                            "media_handle": handle,
                            "search_mode": "targeted",
                            "term_batch_id": f"targeted_{batch_idx}",
                            "term_count": len(term_batch),
                            "term_group_names": "|".join(term_groups_used),
                            "query": query,
                            "start_time": start_time,
                            "end_time": end_time,
                            "positive_terms": term_batch,
                            "negative_terms": negative_terms,
                            "term_groups_used": term_groups_used,
                        }
                    )

    return specs


def _balanced_terms_from_groups(
    term_groups: dict[str, Any],
    group_names: list[str],
) -> list[str]:
    """Interleave groups so the first terms do not come from one group only."""
    queues = [
        dedupe_keep_order(get_terms_from_group(term_groups, group_name))
        for group_name in group_names
    ]
    selected: list[str] = []
    seen: set[str] = set()
    index = 0

    while True:
        added = False
        for queue in queues:
            if index >= len(queue):
                continue
            term = _normalize_spaces(queue[index])
            key = term.casefold()
            if term and key not in seen:
                selected.append(term)
                seen.add(key)
            added = True
        if not added:
            break
        index += 1

    return selected


def build_formal_event_media_query_specs(
    events: list[dict[str, Any]],
    media_accounts: list[dict[str, Any]],
    term_groups: dict[str, Any],
    max_terms: int = 40,
    max_query_chars: int = 900,
    collection_scope: str = "media_anchored",
) -> list[dict[str, Any]]:
    """Build exactly one broad OR query for each formal event-media pair."""
    max_terms_clean = max(1, int(max_terms))
    max_query_chars_clean = max(100, int(max_query_chars))
    scope = str(collection_scope or "media_anchored").strip().lower()
    specs: list[dict[str, Any]] = []

    for event in events:
        window = _resolve_window_for_event(event)
        if window is None:
            continue
        start_time, end_time = window

        profile = event.get("search_profile", {}) or {}
        include_groups = list(profile.get("include_groups", []) or [])
        optional_groups = list(profile.get("optional_groups", []) or [])
        exclude_groups = list(profile.get("exclude_groups", []) or [])
        group_names = dedupe_keep_order(
            [str(name) for name in include_groups + optional_groups if str(name)]
        )
        candidate_terms = _balanced_terms_from_groups(term_groups, group_names)
        negative_terms = build_base_terms_from_groups(
            term_groups,
            [str(name) for name in exclude_groups],
            include_accents_variants=False,
        )

        for media in media_accounts:
            handle = sanitize_handle(media.get("x_identity", {}).get("handle"))
            if not handle:
                raise ValueError(f"Formal media without valid handle: {media}")

            selected_terms: list[str] = []
            for term in candidate_terms:
                if len(selected_terms) >= max_terms_clean:
                    break
                trial_terms = selected_terms + [term]
                trial_query = build_post_query(
                    handle=handle,
                    mode="targeted",
                    positive_terms=trial_terms,
                    negative_terms=negative_terms,
                )
                if len(trial_query) <= max_query_chars_clean:
                    selected_terms = trial_terms

            if not selected_terms:
                raise ValueError(
                    f"Formal event {event.get('event_id')} produced no search terms."
                )

            query = build_post_query(
                handle=handle,
                mode="targeted",
                positive_terms=selected_terms,
                negative_terms=negative_terms,
            )
            if scope == "media_anchored" and f"from:{handle.lower()}" not in query.lower():
                raise ValueError(
                    f"Formal media-anchored query missing from:{handle}: {query}"
                )

            specs.append(
                {
                    "event_id": str(event.get("event_id")),
                    "event_name": event.get("event_name"),
                    "event_date": event.get("event_date"),
                    "formal_order": event.get("formal_order"),
                    "media_id": str(media.get("media_id")),
                    "media_name": media.get("media_name"),
                    "media_handle": handle,
                    "search_mode": "formal_targeted",
                    "term_batch_id": "formal_1",
                    "term_count": len(selected_terms),
                    "term_group_names": "|".join(group_names),
                    "query": query,
                    "start_time": start_time,
                    "end_time": end_time,
                    "positive_terms": selected_terms,
                    "negative_terms": negative_terms,
                    "term_groups_used": group_names,
                }
            )

    expected = len(events) * len(media_accounts)
    if len(specs) != expected:
        raise ValueError(
            f"Expected {expected} formal queries, built {len(specs)}."
        )
    return specs


def _build_balanced_or_clause(terms: list[str], max_terms: int) -> str:
    selected = dedupe_keep_order(terms)[: max(1, int(max_terms))]
    if not selected:
        return ""
    return " OR ".join(quote_term(term) for term in selected)


def build_organic_event_queries(
    events: list[dict[str, Any]],
    term_groups: dict[str, Any],
    max_terms_per_clause: int = 30,
    global_start_time: str | None = None,
    global_end_time: str | None = None,
) -> list[dict[str, Any]]:
    query_specs: list[dict[str, Any]] = []

    for event in events:
        window = _resolve_window_for_event(
            event,
            global_start_time=global_start_time,
            global_end_time=global_end_time,
        )
        if window is None:
            continue
        start_time, end_time = window

        profile = event.get("search_profile", {}) if isinstance(event, dict) else {}
        include_groups = profile.get("include_groups", []) or []
        optional_groups = profile.get("optional_groups", []) or []

        core_terms: list[str] = []
        for group_name in include_groups:
            core_terms.extend(get_terms_from_group(term_groups, str(group_name)))

        optional_terms: list[str] = []
        for group_name in optional_groups:
            optional_terms.extend(get_terms_from_group(term_groups, str(group_name)))

        core_clause = _build_balanced_or_clause(core_terms, max_terms=max_terms_per_clause)
        optional_clause = _build_balanced_or_clause(optional_terms, max_terms=max_terms_per_clause)

        if core_clause and optional_clause:
            query = f"({core_clause}) ({optional_clause}) lang:es -is:retweet"
        elif core_clause:
            query = f"({core_clause}) lang:es -is:retweet"
        elif optional_clause:
            query = f"({optional_clause}) lang:es -is:retweet"
        else:
            continue

        query_specs.append(
            {
                "event_id": str(event.get("event_id")),
                "event_name": event.get("event_name"),
                "event_date": event.get("event_date"),
                "query": query,
                "start_time": start_time,
                "end_time": end_time,
            }
        )

    return query_specs


def build_candidate_interaction_queries(
    events: list[dict[str, Any]],
    term_groups: dict[str, Any],
    political_accounts: list[dict[str, Any]],
    max_terms_per_clause: int = 20,
    global_start_time: str | None = None,
    global_end_time: str | None = None,
) -> list[dict[str, Any]]:
    query_specs: list[dict[str, Any]] = []

    for event in events:
        window = _resolve_window_for_event(
            event,
            global_start_time=global_start_time,
            global_end_time=global_end_time,
        )
        if window is None:
            continue
        start_time, end_time = window

        profile = event.get("search_profile", {}) if isinstance(event, dict) else {}
        group_names = (profile.get("include_groups", []) or []) + (profile.get("optional_groups", []) or [])

        context_terms: list[str] = []
        for group_name in group_names:
            context_terms.extend(get_terms_from_group(term_groups, str(group_name)))

        context_clause = _build_balanced_or_clause(context_terms, max_terms=max_terms_per_clause)
        if not context_clause:
            context_clause = _build_balanced_or_clause(DEFAULT_BASE_QUERY_TERMS, max_terms=max_terms_per_clause)

        for actor in political_accounts:
            if not bool(actor.get("active", True)):
                continue
            handle = sanitize_handle(actor.get("x_handle"))
            if not handle:
                continue

            to_query = f"to:{handle} ({context_clause}) -is:retweet"
            mention_query = f"@{handle} ({context_clause}) -is:retweet"

            for query_mode, query in [("to", to_query), ("mention", mention_query)]:
                query_specs.append(
                    {
                        "event_id": str(event.get("event_id")),
                        "event_name": event.get("event_name"),
                        "event_date": event.get("event_date"),
                        "target_actor_id": actor.get("actor_id"),
                        "target_actor_name": actor.get("actor_name"),
                        "target_handle": handle,
                        "query_mode": query_mode,
                        "query": query,
                        "start_time": start_time,
                        "end_time": end_time,
                    }
                )

    return query_specs

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import os
import time
from typing import Any, Callable

import requests


DEFAULT_X_API_BASE_URL = "https://api.x.com/2"


@dataclass
class XApiUrls:
    base_url: str
    search_recent_url: str
    search_all_url: str
    counts_all_url: str


def _strip_wrapping_quotes(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        return text[1:-1].strip()
    return text


def sanitize_bearer_token(raw_token: str | None) -> str:
    """Sanitize a bearer token loaded from env vars.

    - Trims spaces
    - Removes accidental wrapping quotes
    - Removes accidental 'Bearer ' prefix to avoid 'Bearer Bearer ...'
    """
    token = "" if raw_token is None else str(raw_token)
    token = _strip_wrapping_quotes(token)
    token = token.strip()

    # Remove accidental prefix one or more times.
    while token.lower().startswith("bearer "):
        token = token[7:].strip()
        token = _strip_wrapping_quotes(token)

    return token


def normalize_x_api_base_url(raw_base_url: str | None) -> str:
    """Normalize API base URL to the exact '/2' base without endpoint paths."""
    base_url = (raw_base_url or DEFAULT_X_API_BASE_URL).strip().rstrip("/")

    if not base_url:
        base_url = DEFAULT_X_API_BASE_URL

    # If a full endpoint was provided by mistake, collapse back to /2.
    marker = "/2/tweets/search/"
    if marker in base_url:
        prefix = base_url.split(marker, 1)[0]
        base_url = f"{prefix}/2"

    if base_url.endswith("/tweets/search"):
        base_url = base_url[: -len("/tweets/search")]

    # Ensure /2 suffix exists.
    if not base_url.endswith("/2"):
        if base_url.endswith("/2/"):
            base_url = base_url[:-1]
        else:
            base_url = f"{base_url}/2"

    return base_url.rstrip("/")


def get_x_api_urls(base_url: str | None = None) -> XApiUrls:
    normalized = normalize_x_api_base_url(base_url or os.getenv("X_API_BASE_URL", DEFAULT_X_API_BASE_URL))
    return XApiUrls(
        base_url=normalized,
        search_recent_url=f"{normalized}/tweets/search/recent",
        search_all_url=f"{normalized}/tweets/search/all",
        counts_all_url=f"{normalized}/tweets/counts/all",
    )


def get_quote_tweets_url(tweet_id: str | int, base_url: str | None = None) -> str:
    urls = get_x_api_urls(base_url=base_url)
    return f"{urls.base_url}/tweets/{tweet_id}/quote_tweets"


def get_x_bearer_token(env_var: str = "X_BEARER_TOKEN") -> str:
    raw = os.getenv(env_var)
    token = sanitize_bearer_token(raw)
    if not token:
        raise ValueError(f"Missing or empty {env_var} in environment.")
    return token


def build_x_auth_headers(token: str | None = None) -> dict[str, str]:
    bearer_token = sanitize_bearer_token(token) if token is not None else get_x_bearer_token()
    return {
        "Authorization": f"Bearer {bearer_token}",
        "Content-Type": "application/json",
    }


def parse_datetime_utc(value: Any) -> datetime | None:
    if value is None:
        return None

    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(timezone.utc)


def to_rfc3339_utc(value: Any) -> str | None:
    dt = parse_datetime_utc(value)
    if dt is None:
        return None
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def recent_window_is_eligible(
    start_time: Any,
    end_time: Any,
    recent_days: int = 7,
    now_utc: datetime | None = None,
) -> bool:
    start_dt = parse_datetime_utc(start_time)
    end_dt = parse_datetime_utc(end_time)
    if start_dt is None or end_dt is None:
        return False

    now_dt = now_utc or datetime.now(timezone.utc)
    threshold = now_dt - timedelta(days=recent_days)

    if start_dt < threshold:
        return False
    if end_dt < threshold:
        return False
    if start_dt > now_dt:
        return False

    # Give a small tolerance for clock skew.
    if end_dt > now_dt + timedelta(minutes=5):
        return False

    return True


def validate_x_api_configuration(print_fn: Callable[[str], None] = print) -> dict[str, Any]:
    """Print safe API diagnostics without exposing secrets."""
    raw_token = os.getenv("X_BEARER_TOKEN", "")
    token = sanitize_bearer_token(raw_token)
    urls = get_x_api_urls()

    diagnostics = {
        "search_recent_url": urls.search_recent_url,
        "search_all_url": urls.search_all_url,
        "counts_all_url": urls.counts_all_url,
        "token_exists": bool(raw_token),
        "token_length": len(token),
        "raw_token_starts_with_bearer": raw_token.strip().lower().startswith("bearer "),
    }

    print_fn(f"SEARCH_RECENT_URL: {diagnostics['search_recent_url']}")
    print_fn(f"SEARCH_ALL_URL: {diagnostics['search_all_url']}")
    print_fn(f"COUNTS_ALL_URL: {diagnostics['counts_all_url']}")
    print_fn(f"X_BEARER_TOKEN exists: {diagnostics['token_exists']}")
    print_fn(f"X_BEARER_TOKEN sanitized length: {diagnostics['token_length']}")
    print_fn(
        "X_BEARER_TOKEN raw starts with 'Bearer': "
        f"{diagnostics['raw_token_starts_with_bearer']}"
    )

    return diagnostics


def safe_json(response: requests.Response) -> dict[str, Any]:
    try:
        payload = response.json()
        if isinstance(payload, dict):
            return payload
        return {"payload": payload}
    except Exception:
        return {"raw_text": (response.text or "")[:1000]}


def summarize_api_error(payload: dict[str, Any], max_len: int = 400) -> str:
    try:
        return json.dumps(payload, ensure_ascii=False)[:max_len]
    except Exception:
        return str(payload)[:max_len]


def smoke_test_recent_search(
    query: str = "debate lang:es -is:retweet",
    max_results: int = 10,
    timeout: int = 60,
) -> dict[str, Any]:
    urls = get_x_api_urls()
    headers = build_x_auth_headers()

    params = {
        "query": query,
        "max_results": max(10, min(100, int(max_results))),
        "tweet.fields": "id,text,created_at,author_id,conversation_id,public_metrics,lang",
    }

    response = requests.get(
        urls.search_recent_url,
        headers=headers,
        params=params,
        timeout=timeout,
    )

    payload = safe_json(response)
    n_rows = len(payload.get("data", []) or []) if isinstance(payload, dict) else 0

    return {
        "endpoint": urls.search_recent_url,
        "status_code": response.status_code,
        "ok": response.status_code == 200,
        "n_rows": n_rows,
        "meta": payload.get("meta", {}) if isinstance(payload, dict) else {},
        "error_summary": "" if response.status_code == 200 else summarize_api_error(payload),
    }


def _compute_wait_seconds(response: requests.Response, max_wait_seconds: int) -> int:
    reset_header = response.headers.get("x-rate-limit-reset")
    if reset_header and str(reset_header).isdigit():
        wait_seconds = max(1, int(reset_header) - int(datetime.now(timezone.utc).timestamp()) + 2)
    else:
        wait_seconds = max_wait_seconds
    return int(min(max_wait_seconds, wait_seconds))


def count_x_posts_full_archive(
    query: str,
    start_time: str,
    end_time: str,
    granularity: str = "hour",
    timeout_seconds: int = 60,
    max_rate_limit_wait_seconds: int = 60,
    max_429_retries: int = 3,
    headers: dict[str, str] | None = None,
    base_url: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return Full-Archive count buckets and a compact request audit."""
    granularity_clean = str(granularity or "hour").strip().lower()
    if granularity_clean not in {"minute", "hour", "day"}:
        raise ValueError("granularity must be minute, hour, or day")

    urls = get_x_api_urls(base_url=base_url)
    req_headers = headers or build_x_auth_headers()
    params = {
        "query": query,
        "start_time": start_time,
        "end_time": end_time,
        "granularity": granularity_clean,
    }

    retries_429 = 0
    while True:
        try:
            response = requests.get(
                urls.counts_all_url,
                headers=req_headers,
                params=params,
                timeout=timeout_seconds,
            )
        except Exception as exc:
            return [], {
                "endpoint": "full_archive_counts",
                "endpoint_url": urls.counts_all_url,
                "status": "request_exception",
                "status_code": None,
                "total_tweet_count": 0,
                "error_summary": str(exc)[:400],
            }

        if response.status_code == 429:
            retries_429 += 1
            if retries_429 > max_429_retries:
                return [], {
                    "endpoint": "full_archive_counts",
                    "endpoint_url": urls.counts_all_url,
                    "status": "rate_limit_retry_exceeded",
                    "status_code": 429,
                    "total_tweet_count": 0,
                    "error_summary": "rate_limit_retry_exceeded",
                }
            time.sleep(_compute_wait_seconds(response, max_rate_limit_wait_seconds))
            continue

        payload = safe_json(response)
        if response.status_code != 200:
            return [], {
                "endpoint": "full_archive_counts",
                "endpoint_url": urls.counts_all_url,
                "status": "error",
                "status_code": response.status_code,
                "total_tweet_count": 0,
                "error_summary": summarize_api_error(payload),
            }

        buckets = payload.get("data", []) or []
        meta = payload.get("meta", {}) or {}
        total = meta.get("total_tweet_count")
        if total is None:
            total = sum(int(bucket.get("tweet_count", 0) or 0) for bucket in buckets)

        return buckets, {
            "endpoint": "full_archive_counts",
            "endpoint_url": urls.counts_all_url,
            "status": "ok",
            "status_code": 200,
            "total_tweet_count": int(total or 0),
            "error_summary": "",
        }


def search_x_posts(
    query: str,
    start_time: str | None,
    end_time: str | None,
    max_results: int,
    max_pages: int,
    endpoint: str = "full_archive",
    tweet_fields: str = "id,text,created_at,author_id,conversation_id,in_reply_to_user_id,referenced_tweets,public_metrics,lang",
    timeout_seconds: int = 60,
    sleep_seconds: float = 0.2,
    max_rate_limit_wait_seconds: int = 30,
    max_429_retries: int = 3,
    headers: dict[str, str] | None = None,
    base_url: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """General X post search wrapper for full archive or recent endpoints."""
    endpoint_name = str(endpoint or "full_archive").strip().lower()
    if endpoint_name not in {"full_archive", "recent"}:
        raise ValueError("endpoint must be 'full_archive' or 'recent'")

    urls = get_x_api_urls(base_url=base_url)
    endpoint_url = urls.search_all_url if endpoint_name == "full_archive" else urls.search_recent_url

    req_headers = headers or build_x_auth_headers()

    rows: list[dict[str, Any]] = []
    next_token: str | None = None
    retries_429 = 0
    status_code: int | None = None
    error_summary = ""
    pages_fetched = 0

    page_limit = max(1, int(max_pages))
    while pages_fetched < page_limit:
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

        try:
            response = requests.get(
                endpoint_url,
                headers=req_headers,
                params=params,
                timeout=timeout_seconds,
            )
        except Exception as exc:
            return rows, {
                "endpoint": endpoint_name,
                "endpoint_url": endpoint_url,
                "status_code": None,
                "n_rows": len(rows),
                "pages_fetched": pages_fetched,
                "error_summary": f"request_exception: {exc}",
            }

        status_code = response.status_code

        if status_code == 429:
            retries_429 += 1
            if retries_429 > max_429_retries:
                return rows, {
                    "endpoint": endpoint_name,
                    "endpoint_url": endpoint_url,
                    "status_code": 429,
                    "n_rows": len(rows),
                    "pages_fetched": pages_fetched,
                    "error_summary": "rate_limit_retry_exceeded",
                }
            wait_seconds = _compute_wait_seconds(response, max_rate_limit_wait_seconds)
            time.sleep(wait_seconds)
            continue

        if status_code != 200:
            payload = safe_json(response)
            error_summary = summarize_api_error(payload)
            return rows, {
                "endpoint": endpoint_name,
                "endpoint_url": endpoint_url,
                "status_code": status_code,
                "n_rows": len(rows),
                "pages_fetched": pages_fetched,
                "error_summary": error_summary,
            }

        payload = safe_json(response)
        data = payload.get("data", []) or []
        meta = payload.get("meta", {}) or {}
        rows.extend(data)
        pages_fetched += 1
        retries_429 = 0

        next_token = meta.get("next_token")
        if not next_token:
            break

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    return rows, {
        "endpoint": endpoint_name,
        "endpoint_url": endpoint_url,
        "status_code": status_code,
        "n_rows": len(rows),
        "pages_fetched": pages_fetched,
        "error_summary": error_summary,
    }

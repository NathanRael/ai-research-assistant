import sys
import textwrap
from typing import Optional, TypedDict

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import settings


def _is_latin_text(text: str, threshold: float = 0.7) -> bool:
    if not text:
        return False
    latin_count = sum(1 for ch in text if ch.isascii() or ch.isalpha() and ch.isascii())
    total = sum(1 for ch in text if ch.isprintable())
    return (latin_count / total) > threshold if total > 0 else False


class SearchResult(TypedDict):
    title: str
    content: str
    source: str


class WebSearchClient:
    search_url = "https://api.langsearch.com/v1"
    headers = {
        "Authorization": f"Bearer {settings.lang_search_api}",
        "Content-Type": "application/json",
        "Accept-Language": "en-US,en;q=0.9,fr;q=0.8",
    }

    def __init__(self):
        pass

    def search(
            self,
            query: str,
            limit: Optional[int] = 6,
            freshness: Optional[str] = None,
    ) -> list[SearchResult]:
        payload = {
            "query": query,
            "summary": True,
            "count": limit,
        }
        if freshness:
            payload["freshness"] = freshness

        response = self._post_with_retry(
            url=f"{self.search_url}/web-search",
            headers=self.headers,
            json=payload,
            timeout=60.0,
        )
        response.raise_for_status()
        data = response.json()
        results = data["data"]["webPages"].get("value", [])
        filtered = [
            r for r in results
            if _is_latin_text(r.get("summary") or r.get("snippet", ""))
        ]
        return [
            {
                "title": result.get("name", "Untitled"),
                "content": result.get("summary") or result.get("snippet", ""),
                "source": result.get("url", ""),
            }
            for result in filtered
        ]

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.NetworkError, httpx.TimeoutException)),
        reraise=True,
    )
    def _post_with_retry(self, **kwargs) -> httpx.Response:
        response = httpx.post(**kwargs)
        if response.status_code == 429:
            # Let tenacity retry with exponential backoff; raise so wait is applied.
            response.raise_for_status()
        # Retry server errors and temporary errors, but not client errors like 401/403.
        if response.status_code >= 500 or response.status_code in {408, 502, 503, 504}:
            response.raise_for_status()
        return response


def run_test():
    def _safe_print(text: str) -> None:
        print(text.encode(sys.stdout.encoding, errors="replace").decode(sys.stdout.encoding))

    def print_results(results: list[SearchResult], max_summary_chars: int = 300) -> None:
        separator = "=" * 80
        for idx, result in enumerate(results, start=1):
            print(f"\n{separator}")
            _safe_print(f"  Result #{idx}")
            print(separator)
            _safe_print(f"  Title : {result['title']}")
            _safe_print(f"  Source: {result['source']}")
            print(f"  {'-' * 76}")
            summary = result["content"]
            if len(summary) > max_summary_chars:
                summary = summary[:max_summary_chars].rsplit(" ", 1)[0] + "..."
            for line in textwrap.wrap(summary, width=76):
                _safe_print(f"  {line}")
        print(f"\n{separator}")
        _safe_print(f"  Total results: {len(results)}")
        print(f"{separator}\n")

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    ws = WebSearchClient()
    results = ws.search("Top programming language for developer in 2026")
    print_results(results)

from __future__ import annotations

import logging
import time

import httpx

from .config import JiraSettings

logger = logging.getLogger(__name__)


class JiraClientError(RuntimeError):
    """Raised when the Jira API request fails."""


class JiraClient:
    def __init__(self, settings: JiraSettings) -> None:
        self.settings = settings

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.settings.token}",
            "Accept": "application/json",
        }

    def _request(self, client: httpx.Client, *, start_at: int) -> dict:
        url = f"{self.settings.base_url}/rest/api/2/search"
        params = {
            "jql": self.settings.jql,
            "startAt": start_at,
            "maxResults": self.settings.page_size,
        }

        last_error: Exception | None = None
        for attempt in range(1, self.settings.max_retries + 1):
            try:
                response = client.get(url, params=params, headers=self._headers())
                response.raise_for_status()
                return response.json()
            except (httpx.HTTPError, ValueError) as exc:
                last_error = exc
                logger.warning("Jira request failed on attempt %s/%s: %s", attempt, self.settings.max_retries, exc)
                if attempt < self.settings.max_retries:
                    time.sleep(min(2**(attempt - 1), 5))

        raise JiraClientError(f"Failed to fetch Jira issues after retries: {last_error}") from last_error

    def fetch_all_issues(self) -> list[dict]:
        if not self.settings.base_url:
            raise JiraClientError("JIRA_BASE_URL is not configured.")
        if not self.settings.token:
            raise JiraClientError("JIRA_TOKEN is not configured.")

        issues: list[dict] = []
        start_at = 0

        with httpx.Client(timeout=self.settings.request_timeout_seconds) as client:
            while True:
                payload = self._request(client, start_at=start_at)
                batch = payload.get("issues") or []
                issues.extend(batch)

                total = int(payload.get("total") or len(issues))
                logger.info("Fetched Jira issues %s/%s", len(issues), total)

                if not batch or len(issues) >= total:
                    break
                start_at += len(batch)

        return issues


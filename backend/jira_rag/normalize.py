from __future__ import annotations

from typing import Any

from .schemas import JiraComment, JiraIssueRecord


def _extract_text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, dict):
        if "content" in value:
            return " ".join(part for part in (_extract_text(item) for item in value["content"]) if part)
        if "text" in value:
            return str(value["text"]).strip()
        return " ".join(part for part in (_extract_text(item) for item in value.values()) if part)

    if isinstance(value, list):
        return " ".join(part for part in (_extract_text(item) for item in value) if part)

    return str(value).strip()


def _display_name(value: dict[str, Any] | None) -> str | None:
    if not value:
        return None
    for key in ("displayName", "name", "emailAddress", "accountId"):
        candidate = value.get(key)
        if candidate:
            return str(candidate)
    return None


def normalize_issue(base_url: str, issue: dict[str, Any]) -> JiraIssueRecord:
    fields = issue.get("fields") or {}
    comments_payload = (((fields.get("comment") or {}).get("comments")) or [])

    comments = [
        JiraComment(
            author=_display_name(comment.get("author")),
            body=_extract_text(comment.get("body")),
            created=comment.get("created"),
            updated=comment.get("updated"),
        )
        for comment in comments_payload
    ]

    return JiraIssueRecord(
        issue_key=str(issue.get("key") or ""),
        project_key=((fields.get("project") or {}).get("key")),
        project_name=((fields.get("project") or {}).get("name")),
        summary=str(fields.get("summary") or ""),
        description=_extract_text(fields.get("description")),
        status=((fields.get("status") or {}).get("name")),
        assignee=_display_name(fields.get("assignee")),
        reporter=_display_name(fields.get("reporter")),
        labels=[str(label) for label in (fields.get("labels") or [])],
        comments=comments,
        created=fields.get("created"),
        updated=fields.get("updated"),
        priority=((fields.get("priority") or {}).get("name")),
        issue_type=((fields.get("issuetype") or {}).get("name")),
        raw_url=f"{base_url}/browse/{issue.get('key')}" if issue.get("key") else None,
    )

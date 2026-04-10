from __future__ import annotations

from typing import Iterable

from .schemas import JiraChunkRecord, JiraIssueRecord


def _chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    cleaned = " ".join(text.split())
    if not cleaned:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + chunk_size)
        chunks.append(cleaned[start:end])
        if end >= len(cleaned):
            break
        start = max(end - overlap, start + 1)
    return chunks


def build_issue_chunks(issues: Iterable[JiraIssueRecord]) -> list[JiraChunkRecord]:
    chunks: list[JiraChunkRecord] = []

    for issue in issues:
        section_map = {
            "summary": issue.summary,
            "description": issue.description,
            "comments": "\n".join(
                f"{comment.author or 'Unknown'}: {comment.body}" for comment in issue.comments if comment.body
            ),
            "metadata": " | ".join(
                [
                    f"status={issue.status or 'Unknown'}",
                    f"assignee={issue.assignee or 'Unassigned'}",
                    f"reporter={issue.reporter or 'Unknown'}",
                    f"priority={issue.priority or 'Unknown'}",
                    f"type={issue.issue_type or 'Unknown'}",
                    f"labels={', '.join(issue.labels) if issue.labels else 'none'}",
                    f"created={issue.created or 'Unknown'}",
                    f"updated={issue.updated or 'Unknown'}",
                ]
            ),
        }

        for section, section_text in section_map.items():
            for index, part in enumerate(_chunk_text(section_text)):
                text = (
                    f"Issue {issue.issue_key}\n"
                    f"Summary: {issue.summary or 'No summary'}\n"
                    f"Section: {section}\n"
                    f"{part}"
                )
                chunks.append(
                    JiraChunkRecord(
                        chunk_id=f"{issue.issue_key}:{section}:{index}",
                        issue_key=issue.issue_key,
                        section=section,
                        text=text,
                        excerpt=part[:320],
                        metadata={
                            "project_key": issue.project_key,
                            "project_name": issue.project_name,
                            "summary": issue.summary,
                            "status": issue.status,
                            "assignee": issue.assignee,
                            "priority": issue.priority,
                            "issue_type": issue.issue_type,
                            "labels": issue.labels,
                            "created": issue.created,
                            "updated": issue.updated,
                            "raw_url": issue.raw_url,
                        },
                    )
                )

    return chunks

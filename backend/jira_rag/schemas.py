from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class JiraComment(BaseModel):
    author: str | None = None
    body: str = ""
    created: str | None = None
    updated: str | None = None


class JiraIssueRecord(BaseModel):
    issue_key: str
    project_key: str | None = None
    project_name: str | None = None
    summary: str = ""
    description: str = ""
    status: str | None = None
    assignee: str | None = None
    reporter: str | None = None
    labels: list[str] = Field(default_factory=list)
    comments: list[JiraComment] = Field(default_factory=list)
    created: str | None = None
    updated: str | None = None
    priority: str | None = None
    issue_type: str | None = None
    raw_url: str | None = None


class JiraChunkRecord(BaseModel):
    chunk_id: str
    issue_key: str
    section: str
    text: str
    excerpt: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class JiraSyncRequest(BaseModel):
    force_rebuild: bool = False


class JiraSyncResponse(BaseModel):
    status: str
    synced_issues: int
    chunk_count: int
    saved_to: str
    jql: str
    synced_at: datetime


class JiraChatRequest(BaseModel):
    message: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=12)
    project_keys: list[str] = Field(default_factory=list)


class JiraProjectOption(BaseModel):
    project_key: str
    project_name: str | None = None


class JiraChatSource(BaseModel):
    issue_key: str
    issue_url: str | None = None
    summary: str | None = None
    issue_type: str | None = None
    status: str | None = None
    priority: str | None = None
    created: str | None = None
    updated: str | None = None
    excerpt: str
    section: str
    score: float


class JiraChatIssueLink(BaseModel):
    issue_key: str
    issue_url: str | None = None


class JiraChatResponse(BaseModel):
    answer: str
    source_issue_keys: list[str]
    source_issues: list[JiraChatIssueLink]
    supporting_excerpts: list[JiraChatSource]
    provider: str

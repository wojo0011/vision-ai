from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone

from .chunking import build_issue_chunks
from .config import get_jira_settings
from .index import JiraVectorStore
from .jira_client import JiraClient
from .normalize import normalize_issue
from .providers import build_provider
from .schemas import JiraChatIssueLink, JiraChatResponse, JiraChatSource, JiraIssueRecord, JiraProjectOption, JiraSyncResponse

logger = logging.getLogger(__name__)


class JiraRagService:
    def __init__(self) -> None:
        self.settings = get_jira_settings()
        self.vector_store = JiraVectorStore(self.settings)
        self._lock = threading.RLock()

    def _save_issues(self, issues: list[JiraIssueRecord]) -> None:
        self.settings.ensure_directories()
        with self.settings.issues_path.open("w", encoding="utf-8") as handle:
            json.dump([issue.model_dump() for issue in issues], handle, ensure_ascii=False, indent=2)

    def sync(self) -> JiraSyncResponse:
        with self._lock:
            self.settings = get_jira_settings()
            client = JiraClient(self.settings)
            raw_issues = client.fetch_all_issues()
            issues = [normalize_issue(self.settings.base_url, item) for item in raw_issues if item.get("key")]
            chunks = build_issue_chunks(issues)

            self._save_issues(issues)
            self.vector_store = JiraVectorStore(self.settings)
            self.vector_store.rebuild(chunks)

            synced_at = datetime.now(timezone.utc)
            logger.info("Completed Jira sync with %s issues and %s chunks", len(issues), len(chunks))

            return JiraSyncResponse(
                status="ok",
                synced_issues=len(issues),
                chunk_count=len(chunks),
                saved_to=str(self.settings.data_dir),
                jql=self.settings.jql,
                synced_at=synced_at,
            )

    def chat(self, message: str, top_k: int, project_keys: list[str] | None = None) -> JiraChatResponse:
        with self._lock:
            self.settings = get_jira_settings()
            results = self.vector_store.search(message, top_k=top_k, project_keys=project_keys)
            provider = build_provider(self.settings)
            chunks = [chunk for chunk, _score in results]
            answer = provider.generate_answer(message, chunks)

            source_issue_keys: list[str] = []
            source_issues: list[JiraChatIssueLink] = []
            supporting_excerpts: list[JiraChatSource] = []
            for chunk, score in results:
                if chunk.issue_key not in source_issue_keys:
                    source_issue_keys.append(chunk.issue_key)
                    source_issues.append(
                        JiraChatIssueLink(
                            issue_key=chunk.issue_key,
                            issue_url=chunk.metadata.get("raw_url"),
                        )
                    )
                supporting_excerpts.append(
                    JiraChatSource(
                        issue_key=chunk.issue_key,
                        issue_url=chunk.metadata.get("raw_url"),
                        summary=chunk.metadata.get("summary"),
                        issue_type=chunk.metadata.get("issue_type"),
                        status=chunk.metadata.get("status"),
                        priority=chunk.metadata.get("priority"),
                        created=chunk.metadata.get("created"),
                        updated=chunk.metadata.get("updated"),
                        excerpt=chunk.excerpt,
                        section=chunk.section,
                        score=score,
                    )
                )

            return JiraChatResponse(
                answer=answer,
                source_issue_keys=source_issue_keys,
                source_issues=source_issues,
                supporting_excerpts=supporting_excerpts,
                provider=provider.provider_name,
            )

    def list_projects(self) -> list[JiraProjectOption]:
        self.settings.ensure_directories()
        if not self.settings.issues_path.exists():
            return []

        with self.settings.issues_path.open("r", encoding="utf-8") as handle:
            raw_issues = json.load(handle)

        seen: dict[str, JiraProjectOption] = {}
        for item in raw_issues:
            issue = JiraIssueRecord.model_validate(item)
            if not issue.project_key:
                continue
            seen[issue.project_key] = JiraProjectOption(
                project_key=issue.project_key,
                project_name=issue.project_name,
            )

        return sorted(seen.values(), key=lambda project: (project.project_name or project.project_key).lower())

    def health(self) -> dict[str, object]:
        loaded = self.vector_store.chunk_count > 0 or self.vector_store.load()
        return {
            "configured": bool(self.settings.base_url and self.settings.token),
            "data_dir": str(self.settings.data_dir),
            "chunk_count": self.vector_store.chunk_count,
            "index_ready": bool(loaded),
            "llm_provider": self.settings.llm_provider,
        }


jira_rag_service = JiraRagService()

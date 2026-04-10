from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class JiraSettings:
    base_url: str
    token: str
    jql: str
    page_size: int = 50
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = 32
    data_dir: Path = Path("data/jira")
    request_timeout_seconds: float = 30.0
    max_retries: int = 3
    llm_provider: str = "extractive"
    llm_model: str = "gpt-4o-mini"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""

    @property
    def issues_path(self) -> Path:
        return self.data_dir / "issues.json"

    @property
    def chunks_path(self) -> Path:
        return self.data_dir / "chunks.json"

    @property
    def metadata_path(self) -> Path:
        return self.data_dir / "index-metadata.json"

    @property
    def faiss_index_path(self) -> Path:
        return self.data_dir / "jira.index"

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)


def get_jira_settings() -> JiraSettings:
    base_url = os.getenv("JIRA_BASE_URL", "").strip().rstrip("/")
    token = os.getenv("JIRA_TOKEN", "").strip()
    jql = os.getenv(
        "JIRA_JQL",
        "assignee=currentUser() AND statusCategory != Done ORDER BY updated DESC",
    ).strip()

    return JiraSettings(
        base_url=base_url,
        token=token,
        jql=jql,
        page_size=int(os.getenv("JIRA_PAGE_SIZE", "50")),
        embedding_model_name=os.getenv(
            "JIRA_EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        ).strip(),
        embedding_batch_size=int(os.getenv("JIRA_EMBEDDING_BATCH_SIZE", "32")),
        data_dir=Path(os.getenv("JIRA_DATA_DIR", "data/jira")).resolve(),
        request_timeout_seconds=float(os.getenv("JIRA_REQUEST_TIMEOUT_SECONDS", "30")),
        max_retries=int(os.getenv("JIRA_MAX_RETRIES", "3")),
        llm_provider=os.getenv("JIRA_LLM_PROVIDER", "extractive").strip().lower(),
        llm_model=os.getenv("JIRA_LLM_MODEL", "gpt-4o-mini").strip(),
        llm_base_url=os.getenv("JIRA_LLM_BASE_URL", "https://api.openai.com/v1").strip(),
        llm_api_key=os.getenv("JIRA_LLM_API_KEY", "").strip(),
    )


from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import httpx

from .config import JiraSettings
from .schemas import JiraChunkRecord

logger = logging.getLogger(__name__)


class BaseLlmProvider(ABC):
    provider_name: str

    @abstractmethod
    def generate_answer(self, question: str, chunks: list[JiraChunkRecord]) -> str:
        raise NotImplementedError


class ExtractiveLlmProvider(BaseLlmProvider):
    provider_name = "extractive"

    def generate_answer(self, question: str, chunks: list[JiraChunkRecord]) -> str:
        if not chunks:
            return "I couldn't find any synced Jira context yet. Run a sync first, then ask again."

        lead = chunks[0]
        joined = " ".join(chunk.excerpt for chunk in chunks[:3]).strip()
        lead_url = lead.metadata.get("raw_url")
        lead_reference = f"{lead.issue_key} ({lead_url})" if lead_url else lead.issue_key
        seen_issue_keys: set[str] = set()
        related_references = []
        for chunk in chunks:
            if chunk.issue_key in seen_issue_keys:
                continue
            seen_issue_keys.add(chunk.issue_key)
            issue_url = chunk.metadata.get("raw_url")
            related_references.append(
                f"{chunk.issue_key} ({issue_url})" if issue_url else chunk.issue_key
            )
        return (
            f"Based on the synced Jira issues, the most relevant thread is {lead_reference}. "
            f"Related issues: {', '.join(related_references[:5])}. "
            f"Here is the strongest supporting context: {joined}"
        )


class OpenAiCompatibleProvider(BaseLlmProvider):
    provider_name = "openai-compatible"

    def __init__(self, settings: JiraSettings) -> None:
        self.settings = settings

    def generate_answer(self, question: str, chunks: list[JiraChunkRecord]) -> str:
        if not self.settings.llm_api_key:
            raise RuntimeError("JIRA_LLM_API_KEY is required for the openai-compatible provider.")

        prompt = "\n\n".join(
            [
                "You are a Jira assistant. Answer only from the provided context.",
                "Whenever you mention a Jira ticket key, include its full ticket URL right beside it.",
                f"Question: {question}",
                "Context:",
                "\n\n".join(
                    (
                        f"[{chunk.issue_key} | {chunk.section} | "
                        f"{chunk.metadata.get('raw_url') or 'no-url'}]\n{chunk.text}"
                    )
                    for chunk in chunks
                ),
                "Answer with a concise response and mention uncertainties when context is incomplete.",
            ]
        )

        payload = {
            "model": self.settings.llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }

        with httpx.Client(timeout=self.settings.request_timeout_seconds) as client:
            response = client.post(
                f"{self.settings.llm_base_url.rstrip('/')}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.settings.llm_api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            body = response.json()

        return body["choices"][0]["message"]["content"].strip()


def build_provider(settings: JiraSettings) -> BaseLlmProvider:
    if settings.llm_provider == "openai-compatible":
        logger.info("Using openai-compatible Jira LLM provider")
        return OpenAiCompatibleProvider(settings)

    logger.info("Using extractive Jira LLM provider")
    return ExtractiveLlmProvider()

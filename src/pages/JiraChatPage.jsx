import { useEffect, useState } from "react";

function formatHumanDate(value) {
  if (!value) {
    return "--";
  }

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return date.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function issueTypeClass(issueType) {
  const normalized = String(issueType || "").toLowerCase();
  if (normalized.includes("epic")) {
    return "jira-type-pill epic";
  }
  if (normalized.includes("story")) {
    return "jira-type-pill story";
  }
  if (normalized.includes("bug")) {
    return "jira-type-pill bug";
  }
  if (normalized.includes("task")) {
    return "jira-type-pill task";
  }
  return "jira-type-pill";
}

function statusClass(status) {
  const normalized = String(status || "").toLowerCase();
  if (normalized.includes("done")) {
    return "jira-meta-pill done";
  }
  if (normalized.includes("progress")) {
    return "jira-meta-pill in-progress";
  }
  if (normalized.includes("review") || normalized.includes("qa") || normalized.includes("test")) {
    return "jira-meta-pill review";
  }
  if (normalized.includes("blocked")) {
    return "jira-meta-pill blocked";
  }
  if (normalized.includes("to do") || normalized.includes("open") || normalized.includes("backlog")) {
    return "jira-meta-pill todo";
  }
  return "jira-meta-pill";
}

function priorityClass(priority) {
  const normalized = String(priority || "").toLowerCase();
  if (normalized.includes("highest") || normalized.includes("critical")) {
    return "jira-meta-pill priority-critical";
  }
  if (normalized.includes("high")) {
    return "jira-meta-pill priority-high";
  }
  if (normalized.includes("medium")) {
    return "jira-meta-pill priority-medium";
  }
  if (normalized.includes("low") || normalized.includes("lowest")) {
    return "jira-meta-pill priority-low";
  }
  return "jira-meta-pill";
}

async function readErrorMessage(response) {
  const contentType = response.headers.get("content-type") || "";

  if (contentType.includes("application/json")) {
    const payload = await response.json();
    return payload.detail || payload.message || "Request failed.";
  }

  const text = await response.text();
  return text || `Request failed with status ${response.status}.`;
}

async function syncJira() {
  const response = await fetch("/jira/sync", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ force_rebuild: true }),
  });

  if (!response.ok) {
    throw new Error(await readErrorMessage(response));
  }

  return response.json();
}

async function fetchJiraProjects() {
  const response = await fetch("/jira/projects");

  if (!response.ok) {
    throw new Error(await readErrorMessage(response));
  }

  return response.json();
}

async function sendJiraChat(message, projectKeys) {
  const response = await fetch("/jira/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ message, top_k: 5, project_keys: projectKeys }),
  });

  if (!response.ok) {
    throw new Error(await readErrorMessage(response));
  }

  return response.json();
}

function SourceList({ sources = [] }) {
  if (!sources.length) {
    return null;
  }

  return (
    <div className="jira-sources">
      <p className="jira-sources-title">Supporting context</p>
      {sources.map((source, index) => (
        <article className="jira-source-card" key={`${source.issue_key}-${source.section}-${index}`}>
          <div className="jira-source-header">
            <div className="jira-source-title-block">
              <strong>{source.issue_key}</strong>
              {source.summary ? <h3>{source.summary}</h3> : null}
              <div className="jira-source-subtitle">
                {source.issue_type ? (
                  <span className={issueTypeClass(source.issue_type)}>{source.issue_type}</span>
                ) : null}
                {source.status ? (
                  <span className={statusClass(source.status)}>{source.status}</span>
                ) : null}
                {source.priority ? (
                  <span className={priorityClass(source.priority)}>{source.priority}</span>
                ) : null}
              </div>
            </div>
            <div className="jira-source-actions">
              <span>{source.section}</span>
              <span>{source.score.toFixed(3)}</span>
              {source.issue_url ? (
                <a
                  className="jira-open-link"
                  href={source.issue_url}
                  target="_blank"
                  rel="noreferrer"
                >
                  Open ticket
                </a>
              ) : null}
            </div>
          </div>
          {source.issue_url ? <code className="jira-url-text">{source.issue_url}</code> : null}
          <div className="jira-source-dates">
            <span><strong>Created:</strong> {formatHumanDate(source.created)}</span>
            <span><strong>Updated:</strong> {formatHumanDate(source.updated)}</span>
          </div>
          <p>{source.excerpt}</p>
        </article>
      ))}
    </div>
  );
}

function renderLinkedAnswer(content, sourceIssues = []) {
  if (!content) {
    return null;
  }

  const issueMap = new Map(
    sourceIssues
      .filter((issue) => issue.issue_key)
      .map((issue) => [issue.issue_key, issue.issue_url])
  );

  const keys = Array.from(issueMap.keys()).sort((left, right) => right.length - left.length);
  if (!keys.length) {
    return content.replace(/(https?:\/\/\S+)/g, "$1 ");
  }

  const pattern = new RegExp(`\\b(${keys.map((key) => key.replace(/[.*+?^${}()|[\\]\\\\]/g, "\\$&")).join("|")})\\b`, "g");
  const parts = content.split(pattern);

  const linkedParts = parts.map((part, index) => {
    const issueUrl = issueMap.get(part);
    if (!issueUrl) {
      return <span key={`text-${index}`}>{part}</span>;
    }

    return (
      <span key={`issue-wrap-${part}-${index}`} className="jira-inline-issue">
        <a href={issueUrl} target="_blank" rel="noreferrer">
          {part}
        </a>
        <code className="jira-url-text">{issueUrl}</code>
      </span>
    );
  });

  return linkedParts.flatMap((part, index) => {
    if (typeof part !== "string" && part?.props?.children) {
      return [part];
    }

    const text = typeof part === "string" ? part : part?.props?.children ?? "";
    const tokens = String(text).split(/(https?:\/\/\S+)/g);
    return tokens.filter(Boolean).map((token, tokenIndex) => {
      if (/^https?:\/\/\S+$/.test(token)) {
        return (
          <a
            key={`url-${index}-${tokenIndex}`}
            href={token}
            target="_blank"
            rel="noreferrer"
          >
            {token}
          </a>
        );
      }

      if (typeof part === "string") {
        return <span key={`text-${index}-${tokenIndex}`}>{token}</span>;
      }

      return token ? <span key={`text-${index}-${tokenIndex}`}>{token}</span> : null;
    });
  });
}

export default function JiraChatPage() {
  const [messages, setMessages] = useState([
    {
      id: "welcome",
      role: "assistant",
      content: "Sync Jira issues, then ask about ownership, blockers, priorities, or recent ticket changes.",
      sourceIssueKeys: [],
      sourceIssues: [],
      supportingExcerpts: [],
    },
  ]);
  const [input, setInput] = useState("");
  const [isSyncing, setIsSyncing] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const [syncMessage, setSyncMessage] = useState("");
  const [errorMessage, setErrorMessage] = useState("");
  const [projects, setProjects] = useState([]);
  const [selectedProjects, setSelectedProjects] = useState([]);

  async function refreshProjects() {
    try {
      const payload = await fetchJiraProjects();
      setProjects(payload);
    } catch (error) {
      setErrorMessage(error.message || "Could not load Jira projects.");
    }
  }

  async function handleSync() {
    setIsSyncing(true);
    setErrorMessage("");

    try {
      const payload = await syncJira();
      setSyncMessage(
        `Synced ${payload.synced_issues} issues into ${payload.chunk_count} searchable chunks.`
      );
      await refreshProjects();
    } catch (error) {
      setErrorMessage(error.message || "Jira sync failed.");
    } finally {
      setIsSyncing(false);
    }
  }

  async function handleSend(event) {
    event.preventDefault();

    const trimmed = input.trim();
    if (!trimmed || isSending) {
      return;
    }

    const userMessage = {
      id: `user-${Date.now()}`,
      role: "user",
      content: trimmed,
      sourceIssueKeys: [],
      sourceIssues: [],
      supportingExcerpts: [],
    };

    setMessages((current) => [...current, userMessage]);
    setInput("");
    setErrorMessage("");
    setIsSending(true);

    try {
      const payload = await sendJiraChat(trimmed, selectedProjects);
      setMessages((current) => [
        ...current,
        {
          id: `assistant-${Date.now()}`,
          role: "assistant",
          content: payload.answer,
          sourceIssueKeys: payload.source_issue_keys,
          sourceIssues: payload.source_issues,
          supportingExcerpts: payload.supporting_excerpts,
        },
      ]);
    } catch (error) {
      setErrorMessage(error.message || "Jira chat failed.");
    } finally {
      setIsSending(false);
    }
  }

  useEffect(() => {
    void refreshProjects();
  }, []);

  return (
    <main className="jira-page-shell">
      <section className="jira-hero-card">
        <div>
          <p className="eyebrow">Jira RAG Chat</p>
          <h1>Jira Assistant</h1>
          <p className="lede">
            Sync your Jira issues into a local FAISS index, then ask grounded questions with issue
            keys and excerpts returned alongside each answer.
          </p>
        </div>
        <div className="jira-hero-actions">
          <button className="secondary-button" onClick={handleSync} type="button" disabled={isSyncing}>
            {isSyncing ? "Syncing..." : "Sync Jira"}
          </button>
          <div className="jira-status-copy">
            <strong>{syncMessage || "No Jira sync has been run in this session yet."}</strong>
            <span>Configured through `JIRA_BASE_URL`, `JIRA_TOKEN`, and `JIRA_JQL`.</span>
          </div>
        </div>
      </section>

      <section className="jira-chat-layout">
        <article className="jira-chat-card">
          <div className="jira-chat-header">
            <div className="jira-chat-header-main">
              <p className="panel-label">Conversation</p>
              <h2>Ask about tickets, owners, updates, or blockers</h2>
            </div>
            <div className="jira-project-filter">
              <label className="field">
                <span className="field-title">Projects</span>
                <select
                  className="jira-project-select"
                  value=""
                  onChange={(event) => {
                    const nextValue = event.target.value;
                    if (!nextValue || selectedProjects.includes(nextValue)) {
                      return;
                    }
                    setSelectedProjects((current) => [...current, nextValue]);
                  }}
                >
                  <option value="">All synced projects</option>
                  {projects
                    .filter((project) => !selectedProjects.includes(project.project_key))
                    .map((project) => (
                      <option key={project.project_key} value={project.project_key}>
                        {project.project_name
                          ? `${project.project_name} (${project.project_key})`
                          : project.project_key}
                      </option>
                    ))}
                </select>
              </label>
              {selectedProjects.length ? (
                <div className="jira-project-chip-row">
                  {selectedProjects.map((projectKey) => {
                    const project = projects.find((item) => item.project_key === projectKey);
                    return (
                      <span className="label-chip jira-project-chip" key={projectKey}>
                        <span>{project?.project_name || projectKey}</span>
                        <button
                          className="jira-chip-remove"
                          type="button"
                          onClick={() =>
                            setSelectedProjects((current) =>
                              current.filter((item) => item !== projectKey)
                            )
                          }
                        >
                          x
                        </button>
                      </span>
                    );
                  })}
                </div>
              ) : (
                <p className="muted compact-line">Searching across all synced Jira projects.</p>
              )}
            </div>
          </div>

          <div className="jira-message-list">
            {messages.map((message) => (
              <div
                className={message.role === "user" ? "jira-message-row user" : "jira-message-row"}
                key={message.id}
              >
                <article
                  className={
                    message.role === "user" ? "jira-message-bubble user" : "jira-message-bubble"
                  }
                >
                  <span className="jira-message-role">
                    {message.role === "user" ? "You" : "Assistant"}
                  </span>
                  <p>{renderLinkedAnswer(message.content, message.sourceIssues)}</p>
                  {message.sourceIssueKeys?.length ? (
                    <div className="jira-key-row">
                      {(message.sourceIssues?.length ? message.sourceIssues : message.sourceIssueKeys.map((key) => ({ issue_key: key }))).map((issue) => (
                        <span className="label-chip jira-ticket-chip" key={`${message.id}-${issue.issue_key}`}>
                          <span>{issue.issue_key}</span>
                          {issue.issue_url ? (
                            <>
                              <a
                                className="jira-open-link"
                                href={issue.issue_url}
                                target="_blank"
                                rel="noreferrer"
                              >
                                Open ticket
                              </a>
                              <code className="jira-url-text">{issue.issue_url}</code>
                            </>
                          ) : null}
                        </span>
                      ))}
                    </div>
                  ) : null}
                  <SourceList sources={message.supportingExcerpts} />
                </article>
              </div>
            ))}

            {isSending ? (
              <div className="jira-message-row">
                <article className="jira-message-bubble">
                  <span className="jira-message-role">Assistant</span>
                  <p>Searching synced Jira context and preparing a response...</p>
                </article>
              </div>
            ) : null}
          </div>

          <form className="jira-input-row" onSubmit={handleSend}>
            <input
              className="text-input jira-chat-input"
              type="text"
              value={input}
              onChange={(event) => setInput(event.target.value)}
              placeholder="What changed recently in my active issues?"
              disabled={isSending}
            />
            <button className="primary-button" type="submit" disabled={isSending || !input.trim()}>
              {isSending ? "Sending..." : "Send"}
            </button>
          </form>

          {errorMessage ? <p className="message error">{errorMessage}</p> : null}
        </article>
      </section>
    </main>
  );
}

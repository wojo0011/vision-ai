import { BrowserRouter, NavLink, Navigate, Route, Routes } from "react-router-dom";

import JiraChatPage from "./pages/JiraChatPage";
import VisionAiPage from "./pages/VisionAiPage";

function HomePage() {
  return (
    <main className="workspace-shell">
      <section className="workspace-hero-card">
        <div>
          <p className="eyebrow">Workspace</p>
          <h1>Vision AI and Jira Assistant in one app shell.</h1>
          <p className="lede">
            The existing Vision AI workflow remains intact on its own route, and Jira RAG chat is
            added as a separate workspace for syncing issues and asking grounded questions.
          </p>
        </div>
        <div className="workspace-grid">
          <NavLink className="workspace-link-card" to="/vision-ai">
            <p className="panel-label">Existing Feature</p>
            <h2>Vision AI</h2>
            <p>
              Collect samples, train locally, and classify live camera frames without changes to
              the core flow.
            </p>
          </NavLink>
          <NavLink className="workspace-link-card" to="/jira-chat">
            <p className="panel-label">New Feature</p>
            <h2>Jira Assistant</h2>
            <p>Sync Jira issues into a local vector index and chat over retrieved ticket context.</p>
          </NavLink>
        </div>
      </section>
    </main>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <div className="app-shell routed-shell">
        <div className="ambient ambient-left" />
        <div className="ambient ambient-right" />

        <header className="top-nav">
          <NavLink className="brand-link" to="/">
            Vision AI
          </NavLink>
          <nav className="top-nav-links" aria-label="Primary">
            <NavLink className={({ isActive }) => (isActive ? "nav-pill active" : "nav-pill")} to="/">
              Home
            </NavLink>
            <NavLink
              className={({ isActive }) => (isActive ? "nav-pill active" : "nav-pill")}
              to="/vision-ai"
            >
              Vision AI
            </NavLink>
            <NavLink
              className={({ isActive }) => (isActive ? "nav-pill active" : "nav-pill")}
              to="/jira-chat"
            >
              Jira Assistant
            </NavLink>
          </nav>
        </header>

        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/vision-ai" element={<VisionAiPage />} />
          <Route path="/jira-chat" element={<JiraChatPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}
